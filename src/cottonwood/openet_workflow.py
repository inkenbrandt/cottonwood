
"""
openet_workflow.py

A streamlined, testable set of functions extracted from a Jupyter-style workflow
for fetching OpenET time series by polygon(s), filtering/aggregating to growing seasons,
computing effective precipitation, and comparing against field-scale diversion data.

Dependencies (install as needed):
    pip install pandas geopandas shapely requests numpy scipy pyproj

Environment:
    - Set OPENET_API_KEY in your environment for authenticated requests.
      You can also pass api_key=... to openet_auth_headers().

Notes:
    - The OpenET API is rate limited and requires authentication.
    - This module avoids notebook-state and side-effects. Everything is functional.
    - Dates are handled in UTC; water-year convenience functions are provided.
"""

from __future__ import annotations

import os
import io
import json
import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, shape
import requests

# Optional, used for orthogonal distance regression
try:
    from scipy import odr
    _HAS_ODR = True
except Exception:
    _HAS_ODR = False
    warnings.warn("scipy.odr not available; odr_regression() will fall back to TLS via SVD")


# --------------------------
# Authentication & Utilities
# --------------------------

OPENET_BASE_URL = "https://openet-api.org"  # Update if your endpoint differs


def openet_auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Build Authorization headers for OpenET API.

    Parameters
    ----------
    api_key : str, optional
        API key string. If None, checks the OPENET_API_KEY environment variable.

    Returns
    -------
    dict
        Headers with Authorization bearer token.
    """
    key = api_key or os.getenv("OPENET_API_KEY")
    if not key:
        raise ValueError("OpenET API key not found. Set OPENET_API_KEY or pass api_key=...")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def to_crs_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure a GeoDataFrame is in WGS84 (EPSG:4326).

    Parameters
    ----------
    gdf : GeoDataFrame

    Returns
    -------
    GeoDataFrame
        Reprojected to EPSG:4326 if necessary.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS. Please set gdf.set_crs(...) first.")
    if str(gdf.crs).lower() in ("epsg:4326", "wgs84", "epsg:4326"):
        return gdf
    return gdf.to_crs(4326)


def water_year(dtindex: pd.DatetimeIndex) -> pd.Series:
    """
    Water year (WY) for each timestamp; WY is the year in which period ends (Oct-Sep).

    Parameters
    ----------
    dtindex : DatetimeIndex

    Returns
    -------
    Series of int
        Water year (e.g., dates in 2024-10-01..2025-09-30 map to WY=2025).
    """
    months = dtindex.month
    years = dtindex.year
    return pd.Series(np.where(months >= 10, years + 1, years), index=dtindex)


def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Coerce a column to datetime and set as index if not already.

    Parameters
    ----------
    df : DataFrame
    date_col : str

    Returns
    -------
    DataFrame
        Copy with datetime index named 'date'.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.set_index(date_col).sort_index()
    out.index.name = "date"
    return out


# ------------------------------
# Build request payloads to OpenET
# ------------------------------

def make_multipolygon_geojson(
    gdf: gpd.GeoDataFrame,
    id_field: str,
    dissolve_by_id: bool = True
) -> Dict:
    """
    Construct a valid GeoJSON FeatureCollection from a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Polygons to request OpenET over; CRS must be EPSG:4326 or reprojectable.
    id_field : str
        Column naming unique polygon IDs (used for joins to time series results).
    dissolve_by_id : bool, default True
        If True, dissolves multipart features that share the same id_field value.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.
    """
    gdf = to_crs_wgs84(gdf)
    if dissolve_by_id:
        gdf = gdf.dissolve(id_field).reset_index()

    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "id": str(row[id_field]),
            "properties": {id_field: row[id_field]},
            "geometry": mapping(geom)
        })
    return {"type": "FeatureCollection", "features": features}


# ------------------------------
# OpenET API Calls
# ------------------------------

def openet_timeseries_multipolygon(
    polygons_fc: Dict,
    variables: Sequence[str],
    start_date: str,
    end_date: str,
    interval: str = "daily",
    model: str = "ensemble",
    units: str = "in",
    base_url: str = OPENET_BASE_URL,
    api_key: Optional[str] = None,
) -> Dict:
    """
    Call OpenET /raster/timeseries/multipolygon endpoint.

    Parameters
    ----------
    polygons_fc : dict
        GeoJSON FeatureCollection of the polygons.
    variables : Sequence[str]
        Variable names to request, e.g., ["et", "eto", "pr"].
    start_date, end_date : str
        ISO date strings, e.g., "2016-03-01".
    interval : {"daily", "monthly", "annual"}
    model : str
        OpenET model key; "ensemble", "ssebop", etc.
    units : {"in","mm"}
    base_url : str
        OpenET base URL.
    api_key : str, optional
        API key; if None, environment variable is used.

    Returns
    -------
    dict
        JSON response from OpenET.
    """
    url = f"{base_url}/raster/timeseries/multipolygon/"
    headers = openet_auth_headers(api_key)
    payload = {
        "polygons": polygons_fc,
        "model": model,
        "variables": list(variables),
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "units": units
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if not resp.ok:
        raise RuntimeError(f"OpenET API error {resp.status_code}: {resp.text}")
    return resp.json()


def parse_openet_timeseries(resp_json: Dict, id_field: str = "id") -> pd.DataFrame:
    """
    Parse OpenET multipolygon time series response into tidy DataFrame.

    Parameters
    ----------
    resp_json : dict
        Response JSON from openet_timeseries_multipolygon().
    id_field : str, default "id"
        Polygon identifier field provided in the GeoJSON Feature "id".

    Returns
    -------
    DataFrame
        Columns: [id_field, date, variable, value]
        Pivot-friendly; use pivot_table to wide-format if desired.
    """
    records = []
    feats = resp_json.get("features") or resp_json.get("data") or []
    for feat in feats:
        pid = str(feat.get("id") or feat.get("properties", {}).get(id_field))
        ts = feat.get("timeseries", {})
        for var, series in ts.items():
            # series is list of {"date":"YYYY-MM-DD","value":float or None}
            for item in series:
                records.append({
                    id_field: pid,
                    "date": item["date"],
                    "variable": var,
                    "value": item["value"]
                })
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", id_field, "variable"]).reset_index(drop=True)


def pivot_openet(df_long: pd.DataFrame, id_field: str = "id") -> pd.DataFrame:
    """
    Pivot tidy OpenET time series to wide format with variables as columns.

    Parameters
    ----------
    df_long : DataFrame
        Output of parse_openet_timeseries().
    id_field : str

    Returns
    -------
    DataFrame
        Index: MultiIndex [id, date]; Columns: variables
    """
    if df_long.empty:
        return df_long.copy()
    wide = (
        df_long.pivot_table(index=[id_field, "date"], columns="variable", values="value", aggfunc="mean")
               .sort_index()
    )
    wide.columns.name = None
    return wide


# ------------------------------
# Season Filtering & Aggregation
# ------------------------------

def filter_growing_season(
    df: pd.DataFrame,
    start_month: int = 3,
    end_month: int = 10,
) -> pd.DataFrame:
    """
    Keep only rows within a growing-season month window (inclusive).

    Parameters
    ----------
    df : DataFrame
        Must be indexed by DatetimeIndex or contain a 'date' column to set as index.
    start_month : int, default 3 (March)
    end_month : int, default 10 (October)

    Returns
    -------
    DataFrame
        Filtered to months in [start_month, end_month].
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out = out.set_index(pd.to_datetime(out["date"]))
        else:
            raise ValueError("DataFrame must have DatetimeIndex or a 'date' column.")
    return out[(out.index.month >= start_month) & (out.index.month <= end_month)]


def add_water_year_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'WY' column inferred from the DatetimeIndex.

    Parameters
    ----------
    df : DataFrame
        Must be DatetimeIndex-indexed.

    Returns
    -------
    DataFrame
        Copy with 'WY' column.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("add_water_year_index: DataFrame must have a DatetimeIndex")
    out = df.copy()
    out["WY"] = water_year(out.index).values
    return out


def aggregate_by_water_year(
    df: pd.DataFrame,
    how: str = "sum",
    cols: Optional[Sequence[str]] = None,
    id_index_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate a DatetimeIndex DataFrame by water year.

    Parameters
    ----------
    df : DataFrame
        Index is either DatetimeIndex or MultiIndex with level [-1] as date.
    how : {"sum","mean","median"}
    cols : sequence of str, optional
        Which value columns to aggregate. If None, all numeric columns are used.
    id_index_name : str, optional
        If df has a MultiIndex (e.g., [id, date]), provide the polygon id level name
        so grouping happens by [id, WY].

    Returns
    -------
    DataFrame
        Aggregated by WY (and id if provided).
    """
    # Reindex for convenience
    if isinstance(df.index, pd.MultiIndex):
        # Expect last level to be date
        date_level = df.index.names[-1]
        tmp = df.copy()
        tmp = tmp.reset_index()
        tmp[date_level] = pd.to_datetime(tmp[date_level])
        tmp = tmp.set_index(date_level).sort_index()
        tmp = add_water_year_index(tmp)
        group_keys = ["WY"]
        if id_index_name and id_index_name in tmp.columns:
            group_keys.insert(0, id_index_name)
        value_cols = cols or tmp.select_dtypes(include=[np.number]).columns.tolist()
        agg = getattr(tmp.groupby(group_keys)[value_cols], how)()
        return agg
    else:
        tmp = add_water_year_index(df)
        value_cols = cols or tmp.select_dtypes(include=[np.number]).columns.tolist()
        return getattr(tmp.groupby("WY")[value_cols], how)()


def aggregate_monthly(
    df: pd.DataFrame,
    how: str = "sum",
    cols: Optional[Sequence[str]] = None,
    id_index_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate a DatetimeIndex DataFrame by calendar month (YYYY-MM).

    Parameters
    ----------
    df : DataFrame
    how : {"sum","mean","median"}
    cols : sequence of str, optional
    id_index_name : str, optional

    Returns
    -------
    DataFrame
        Aggregated by month (and id if provided).
    """
    if isinstance(df.index, pd.MultiIndex):
        date_level = df.index.names[-1]
        tmp = df.copy().reset_index()
        tmp[date_level] = pd.to_datetime(tmp[date_level])
        tmp["YM"] = tmp[date_level].dt.to_period("M").dt.to_timestamp()
        group_keys = ["YM"]
        if id_index_name and id_index_name in tmp.columns:
            group_keys.insert(0, id_index_name)
        value_cols = cols or tmp.select_dtypes(include=[np.number]).columns.tolist()
        agg = getattr(tmp.groupby(group_keys)[value_cols], how)()
        return agg
    else:
        tmp = df.copy()
        if not isinstance(tmp.index, pd.DatetimeIndex):
            raise ValueError("aggregate_monthly: DataFrame must be indexed by DatetimeIndex")
        tmp["YM"] = tmp.index.to_period("M").to_timestamp()
        value_cols = cols or tmp.select_dtypes(include=[np.number]).columns.tolist()
        return getattr(tmp.groupby("YM")[value_cols], how)()


# ------------------------------
# Hydrologic Calculations
# ------------------------------

def compute_effective_precip(
    df: pd.DataFrame,
    et_col: str = "et",
    precip_col: str = "pr",
    out_col: str = "ep"
) -> pd.DataFrame:
    """
    Effective precipitation EP = ET - P (positive implies depletion from diversions).

    Parameters
    ----------
    df : DataFrame
        Source with at least et_col and precip_col.
    et_col : str
    precip_col : str
    out_col : str

    Returns
    -------
    DataFrame
        Copy with an added out_col column.
    """
    out = df.copy()
    if et_col not in out.columns or precip_col not in out.columns:
        raise KeyError(f"Columns {et_col} and {precip_col} must be present")
    out[out_col] = out[et_col] - out[precip_col]
    return out


GALLONS_PER_ACRE_FOOT = 325851.429


def hourly_gpm_to_acft(series_gpm: pd.Series) -> float:
    """
    Convert an hourly series of GPM (gallons per minute) to seasonal acre-feet.

    Parameters
    ----------
    series_gpm : Series
        Hourly timestamps with values in GPM. Missing values are linearly interpolated.

    Returns
    -------
    float
        Total acre-feet represented by the series.
    """
    s = series_gpm.sort_index().astype(float)
    s = s.asfreq("H")
    s = s.interpolate(limit_direction="both")
    gallons = (s * 60).sum()  # gpm -> gph then sum hour by hour
    return float(gallons / GALLONS_PER_ACRE_FOOT)


def compare_totalizer_vs_instantaneous(
    totalizer_acft: Union[pd.Series, float],
    instantaneous_hourly_gpm: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compare yearly totalizer readings (acre-feet) against aggregated instantaneous (hourly GPM).

    Parameters
    ----------
    totalizer_acft : Series or float
        If Series: index should be year (int) or water-year; values are ac-ft.
    instantaneous_hourly_gpm : Series, optional
        Hourly GPM; if provided, it will be aggregated to yearly ac-ft and compared.

    Returns
    -------
    DataFrame
        Columns: ["totalizer_acft", "instantaneous_acft", "difference_acft"]
    """
    if instantaneous_hourly_gpm is None:
        if isinstance(totalizer_acft, (int, float)):
            return pd.DataFrame({"totalizer_acft": [float(totalizer_acft)]})
        else:
            return pd.DataFrame({"totalizer_acft": totalizer_acft})

    # Aggregate instantaneous by water year if index is datetime
    inst = instantaneous_hourly_gpm.copy()
    if isinstance(inst.index, pd.DatetimeIndex):
        wy = water_year(inst.index)
        df = inst.to_frame("gpm").assign(WY=wy.values).groupby("WY")["gpm"].apply(hourly_gpm_to_acft)
        inst_acft = df
    else:
        # Already annual? interpret as hourly series w/o datetimes is ambiguous
        raise ValueError("instantaneous_hourly_gpm must be timestamped (DatetimeIndex, hourly)")

    tot = totalizer_acft
    if not isinstance(tot, pd.Series):
        raise ValueError("For comparison, totalizer_acft must be a pandas Series indexed by year/WY")

    combo = pd.concat([tot.rename("totalizer_acft"), inst_acft.rename("instantaneous_acft")], axis=1)
    combo["difference_acft"] = combo["totalizer_acft"] - combo["instantaneous_acft"]
    return combo


# ------------------------------
# Statistics: ODR & Bland-Altman
# ------------------------------

def odr_regression(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    """
    Orthogonal Distance Regression (ODR) y ~ a*x + b.

    Parameters
    ----------
    x, y : Series
        Must be numeric and of equal length (NaNs dropped pairwise).

    Returns
    -------
    dict
        {'slope', 'intercept', 'slope_se', 'intercept_se', 'r2', 'n'}
    """
    xy = pd.concat([x, y], axis=1).dropna()
    xx, yy = xy.iloc[:, 0].values, xy.iloc[:, 1].values
    n = len(xy)
    if n < 2:
        return {"slope": np.nan, "intercept": np.nan, "slope_se": np.nan, "intercept_se": np.nan, "r2": np.nan, "n": n}

    if _HAS_ODR:
        def f(B, z):  # linear
            return B[0] * z + B[1]
        linear = odr.Model(f)
        data = odr.RealData(xx, yy)
        odr_inst = odr.ODR(data, linear, beta0=[1.0, 0.0])
        out = odr_inst.run()
        a, b = out.beta
        sa, sb = out.sd_beta
    else:
        # TLS via SVD on centered data
        x0 = xx - xx.mean()
        y0 = yy - yy.mean()
        U, svals, Vt = np.linalg.svd(np.vstack([x0, y0]).T, full_matrices=False)
        # TLS slope is -vxy/vxx for smallest singular vector
        v = Vt.T[:, -1]
        a = -v[0] / v[1]
        b = yy.mean() - a * xx.mean()
        sa = np.nan
        sb = np.nan

    r2 = np.corrcoef(xx, yy)[0, 1] ** 2
    return {"slope": float(a), "intercept": float(b), "slope_se": float(sa), "intercept_se": float(sb), "r2": float(r2), "n": int(n)}


def bland_altman(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    """
    Bland-Altman statistics and per-point differences (y - x).

    Parameters
    ----------
    x, y : Series
        Measurements from two methods/instruments.

    Returns
    -------
    DataFrame
        Columns: ["avg", "diff", "mean_diff", "sd_diff", "loa_lower", "loa_upper"]
        avg/diff are per-point; the others are constants (broadcast) summarizing bias.
    """
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=["avg", "diff", "mean_diff", "sd_diff", "loa_lower", "loa_upper"])

    df["avg"] = df[["x", "y"]].mean(axis=1)
    df["diff"] = df["y"] - df["x"]
    mean_diff = df["diff"].mean()
    sd_diff = df["diff"].std(ddof=1)
    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff
    df["mean_diff"] = mean_diff
    df["sd_diff"] = sd_diff
    df["loa_lower"] = loa_lower
    df["loa_upper"] = loa_upper
    return df


# ------------------------------
# Zonal statistics & joins
# ------------------------------

def add_polygon_area_acres(gdf: gpd.GeoDataFrame, area_field: str = "area_acres") -> gpd.GeoDataFrame:
    """
    Add polygon area in acres (projects to equal-area before computing).

    Parameters
    ----------
    gdf : GeoDataFrame
    area_field : str

    Returns
    -------
    GeoDataFrame
        Copy with area_field in acres.
    """
    # Use a US equal-area CRS
    g = gdf.copy()
    crs_orig = g.crs
    g = g.to_crs(5070)  # NAD83 / Conus Albers
    g[area_field] = g.geometry.area / 4046.8564224  # m^2 -> acres
    return g.to_crs(crs_orig)


def join_timeseries_with_polygons(
    ts: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
    id_col_ts: str,
    id_col_poly: str
) -> gpd.GeoDataFrame:
    """
    Join a wide-format timeseries (MultiIndex [id, date]) to polygons by id.

    Parameters
    ----------
    ts : DataFrame
        Wide-format with MultiIndex [id, date].
    polygons : GeoDataFrame
    id_col_ts : str
        Name of the id level in ts.index.
    id_col_poly : str
        Column in polygons with matching ids.

    Returns
    -------
    GeoDataFrame
        One row per polygon (geometry preserved) with no time-series expansion.
        Useful for attaching static attributes like area; for spatiotemporal joins,
        keep ts separate or explode later.
    """
    if not isinstance(ts.index, pd.MultiIndex):
        raise ValueError("ts must have a MultiIndex [id, date]")
    if id_col_ts not in ts.index.names:
        raise KeyError(f"id level '{id_col_ts}' not found in ts.index.names={ts.index.names}")

    # Reduce ts to a per-id statistic (e.g., annual sum already done upstream)
    per_id = ts.groupby(level=id_col_ts).sum(numeric_only=True)
    per_id = per_id.reset_index().rename(columns={id_col_ts: id_col_poly})
    return polygons.merge(per_id, on=id_col_poly, how="left")


# ------------------------------
# High-level convenience wrappers
# ------------------------------

def fetch_openet_timeseries_for_polygons(
    polygons_gdf: gpd.GeoDataFrame,
    id_field: str,
    variables: Sequence[str],
    start_date: str,
    end_date: str,
    interval: str = "daily",
    model: str = "ensemble",
    units: str = "in",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    One-stop: build GeoJSON, call OpenET, parse, and pivot to wide.

    Returns
    -------
    DataFrame
        MultiIndex [id_field, date] wide with variables as columns.
    """
    fc = make_multipolygon_geojson(polygons_gdf, id_field=id_field, dissolve_by_id=True)
    resp = openet_timeseries_multipolygon(
        fc, variables=variables, start_date=start_date, end_date=end_date,
        interval=interval, model=model, units=units, api_key=api_key
    )
    long = parse_openet_timeseries(resp, id_field="id")
    wide = pivot_openet(long, id_field="id")
    # Ensure id level is named id_field for clarity
    wide.index = wide.index.set_names([id_field, "date"])
    return wide


def growing_season_annual_effective_precip(
    ts_wide: pd.DataFrame,
    et_col: str = "et",
    p_col: str = "pr",
    start_month: int = 3,
    end_month: int = 10,
    how: str = "sum",
    id_level: str = "id"
) -> pd.DataFrame:
    """
    Filter to growing season, compute EP=ET-P, then aggregate by WY.

    Parameters
    ----------
    ts_wide : DataFrame
        MultiIndex [id, date] with columns containing et_col and p_col.
    et_col : str
    p_col : str
    start_month, end_month : int
    how : {"sum","mean","median"}
    id_level : str
        Name of polygon id level in index.

    Returns
    -------
    DataFrame
        Aggregated by [id, WY] with ET, P, and EP columns.
    """
    if not isinstance(ts_wide.index, pd.MultiIndex):
        raise ValueError("ts_wide must be MultiIndex [id, date]")
    # Filter to season
    tmp = ts_wide.copy()
    tmp = tmp.reset_index().set_index("date").sort_index()
    tmp = filter_growing_season(tmp, start_month=start_month, end_month=end_month)

    # Compute EP
    needed = [et_col, p_col]
    missing = [c for c in needed if c not in tmp.columns]
    if missing:
        raise KeyError(f"Columns missing for EP calc: {missing}")
    tmp = compute_effective_precip(tmp, et_col=et_col, precip_col=p_col, out_col="ep")

    # Aggregate by WY and id
    tmp[id_level] = tmp[id_level].astype(str)
    tmp = add_water_year_index(tmp)
    agg = getattr(tmp.groupby([id_level, "WY"])[[et_col, p_col, "ep"]], how)()
    return agg


# ------------------------------
# File IO helpers (optional)
# ------------------------------

def read_polygons(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Read polygons from supported vector file (GeoPackage, shapefile, GeoJSON).

    Parameters
    ----------
    path : str
    layer : str, optional
        For GeoPackage, specify layer name.

    Returns
    -------
    GeoDataFrame
    """
    return gpd.read_file(path, layer=layer)


def read_diversions_from_csv(path: str, date_col: str, value_col: str, units: str = "gpm") -> pd.DataFrame:
    """
    Generic read of diversion time series from CSV.

    Parameters
    ----------
    path : str
    date_col : str
    value_col : str
    units : {"gpm","acft"}
        If gpm, assumed hourly instantaneous; if acft, assumed annual totals.

    Returns
    -------
    DataFrame
        If gpm: DatetimeIndex hourly; If acft: annual totals index.
    """
    df = pd.read_csv(path)
    if units.lower() == "gpm":
        df = ensure_datetime(df, date_col)
        df = df.rename(columns={value_col: "gpm"})[["gpm"]]
        # align to hourly
        df = df.asfreq("H").interpolate(limit_direction="both")
        return df
    elif units.lower() == "acft":
        # Expect year column
        df[date_col] = df[date_col].astype(int)
        return df.set_index(date_col)[[value_col]].rename(columns={value_col: "acft"})
    else:
        raise ValueError("units must be 'gpm' or 'acft'")


# End of module
