"""
Utilities to compile weather station data from multiple networks
into tidy pandas/GeoPandas tables.

This module refactors code from `Compile_Weather_Station_Data.ipynb`
into testable, reusable functions with minimal side effects.

Key sources covered
-------------------
1) Utah Climate Center (UCC) API for AgriMet / UAGRIMET-like networks
   Base: https://climate.usu.edu/API/api.php
2) NRCS SNOTEL CSV report generator
   Base: https://wcc.sc.egov.usda.gov/reportGenerator/
3) MesoWest CSV exports (parsed from a local directory) + elevation via
   USGS Elevation Point Query Service (EPQS)
   Base: https://epqs.nationalmap.gov/v1/json

Notes
-----
- To avoid hard-coding secrets, pass `api_key` explicitly.
- Date ranges are chunked to be gentle on APIs.
- Functions avoid writing to disk unless you call the explicit exporters.

Author: refactor from the original notebook
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union, Dict

import io
import math
import pathlib
import datetime as dt

import pandas as pd
import numpy as np

# Optional deps
try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# -----------------------------
# General helpers
# -----------------------------

@dataclass
class DateWindow:
    start: pd.Timestamp
    end: pd.Timestamp


def _ensure_requests():
    if requests is None:
        raise RuntimeError("The 'requests' package is required for this function.")


def _to_ts(x: Union[str, dt.date, dt.datetime, pd.Timestamp]) -> pd.Timestamp:
    return pd.to_datetime(x).tz_localize(None)


def chunk_date_range(start: Union[str, dt.date, dt.datetime],
                     end: Union[str, dt.date, dt.datetime],
                     years_per_chunk: int = 3) -> List[DateWindow]:
    """Split a date range into ~N-year chunks for friendlier API calls.

    Parameters
    ----------
    start, end : str or date-like
        Inclusive date bounds.
    years_per_chunk : int
        Number of years per chunk (approx; ends on Dec 31).

    Returns
    -------
    list[DateWindow]
    """
    s = _to_ts(start)
    e = _to_ts(end)
    if e < s:
        raise ValueError("end must be >= start")

    windows: List[DateWindow] = []
    cur = pd.Timestamp(year=s.year, month=1, day=1)
    if s > cur:
        cur = s
    while cur <= e:
        end_year = min(cur.year + years_per_chunk - 1, e.year)
        wnd_end = pd.Timestamp(year=end_year, month=12, day=31)
        wnd_end = min(wnd_end, e)
        windows.append(DateWindow(start=cur, end=wnd_end))
        cur = pd.Timestamp(year=end_year + 1, month=1, day=1)
    return windows


# -----------------------------
# UCC (Utah Climate Center) API
# -----------------------------

UCC_BASE = "https://climate.usu.edu/API/api.php"


def ucc_list_stations(api_key: str,
                      network: Optional[str] = None,
                      source: Optional[str] = None,
                      version: str = "v3") -> pd.DataFrame:
    """List stations available from the UCC API.

    Parameters
    ----------
    api_key : str
        UCC API key.
    network : str, optional
        Network code (e.g., "UAGRIMET"). If omitted, you can filter downstream.
    source : str, optional
        Source filter (e.g., "UCC"). Either `network` or `source` may be set.
    version : str
        API version (default v3).

    Returns
    -------
    pandas.DataFrame
        One row per station. Includes station_id, name, latitude, longitude, etc.
    """
    _ensure_requests()
    if not (network or source):
        raise ValueError("Provide at least one of network or source.")

    if network and source:
        url = f"{UCC_BASE}/{version}/key={api_key}/station_search/source={source}/network={network}"
    elif network:
        url = f"{UCC_BASE}/{version}/key={api_key}/station_search/network={network}"
    else:  # source only
        url = f"{UCC_BASE}/{version}/key={api_key}/station_search/source={source}"

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    js = resp.json()
    if not js.get("success", False):
        raise RuntimeError(f"UCC API error: {js}")

    df = pd.DataFrame(js["payload"]) if js.get("payload") else pd.DataFrame()
    if not df.empty:
        # Normalize types and tidy indexing
        if "station_id" in df:
            df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce").astype("Int64")
        df = df.drop_duplicates(subset=[c for c in df.columns if c != "geometry"], keep="first")
    return df


def ucc_get_daily(api_key: str,
                   station_ids: Sequence[Union[str, int]],
                   start: Union[str, dt.date, dt.datetime],
                   end: Union[str, dt.date, dt.datetime],
                   network: str = "UAGRIMET",
                   units: str = "e",
                   version: str = "v3",
                   years_per_chunk: int = 3) -> pd.DataFrame:
    """Download daily data for a list of UCC station_ids.

    Parameters
    ----------
    api_key : str
        UCC API key.
    station_ids : sequence of ids
        Numeric station identifiers from `ucc_list_stations`.
    start, end : date-like
        Inclusive range of dates to request.
    network : str
        Network code (e.g., "UAGRIMET").
    units : {"e","m"}
        English or metric units (UCC API convention).
    version : str
        API version (default v3).
    years_per_chunk : int
        Breaks the request into chunks to reduce API load.

    Returns
    -------
    pandas.DataFrame
        Multi-station daily time series with columns like
        [station_id, date_time, element..., quality flags if present].
    """
    _ensure_requests()
    windows = chunk_date_range(start, end, years_per_chunk=years_per_chunk)

    out: List[pd.DataFrame] = []
    for sid in station_ids:
        for w in windows:
            url = (
                f"{UCC_BASE}/{version}/key={api_key}"
                f"/station_search/network={network}"
                f"/station_id={sid}/get_daily"
                f"/start_date={w.start:%Y-%m-%d}/end_date={w.end:%Y-%m-%d}/units={units}"
            )
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            js = r.json()
            if not js.get("success", False) or not js.get("payload"):
                continue
            df = pd.DataFrame(js["payload"]).assign(station_id=int(sid))
            if "date_time" in df:
                df["date_time"] = pd.to_datetime(df["date_time"]).dt.tz_localize(None)
            out.append(df)
            print(f"Fetched station_id={sid} for {w.start:%Y-%m-%d} to {w.end:%Y-%m-%d}, {len(df)} rows.")

    if not out:
        return pd.DataFrame()

    df_all = pd.concat(out, ignore_index=True)
    # Ensure canonical dtypes
    if "station_id" in df_all:
        df_all["station_id"] = pd.to_numeric(df_all["station_id"], errors="coerce").astype("Int64")
    # Sort
    sort_cols = [c for c in ["station_id", "date_time"] if c in df_all.columns]
    if sort_cols:
        df_all = df_all.sort_values(sort_cols).reset_index(drop=True)
    return df_all


# -----------------------------
# SNOTEL via NRCS report generator
# -----------------------------

# Example template from the notebook, generalized here:
# "view_csv/customSingleStationReport/daily/{station}:{state}:SNTL::value,PREC::value,TOBS::value,TMAX::value,TMIN::value,TAVG::value?startDate=2016-01-01&endDate=2020-12-31"
SNOTEL_BASE = "https://wcc.sc.egov.usda.gov/reportGenerator/"


def snotel_get_daily(
    stations: Mapping[Union[int, str], Tuple[str, str]],
    start: Union[str, dt.date, dt.datetime],
    end: Union[str, dt.date, dt.datetime],
) -> Dict[Union[int, str], pd.DataFrame]:
    """Fetch daily SNOTEL tables for multiple stations.

    Parameters
    ----------
    stations : mapping
        Mapping of station code -> (state_abbrev, human_readable_name). Example:
        `{714: ("UT", "Red Pine Ridge")}`.
    start, end : date-like
        Inclusive date range.

    Returns
    -------
    dict
        Keys are the input station codes; values are tidy DataFrames. Columns
        depend on the selected variables in the URL.
    """
    _ensure_requests()
    s = _to_ts(start).date()
    e = _to_ts(end).date()

    # Variables adapted from notebook: snow water equivalent (WTEQ), precipitation, temperatures
    var_list = [
        "WTEQ::value",
        "PREC::value",
        "TOBS::value",
        "TMAX::value",
        "TMIN::value",
        "TAVG::value",
    ]

    out: Dict[Union[int, str], pd.DataFrame] = {}
    for code, (state, _name) in stations.items():
        # Build NRCS custom report URL
        path = (
            f"view_csv/customSingleStationReport/daily/"
            f"{code}:{state}:SNTL::" + ",".join(var_list)
            + f"?start={s:%Y-%m-%d}&end={e:%Y-%m-%d}"
        )
        url = SNOTEL_BASE + path
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        # NRCS CSVs often start with metadata commented by '#'
        df = pd.read_csv(io.StringIO(r.text), comment="#")
        # Try to locate the date column heuristically
        date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
        if date_col is not None:
            df["date_time"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df["snotel_code"] = str(code)
        out[code] = df
    return out


# -----------------------------
# MesoWest directory parsing + EPQS elevation lookup
# -----------------------------

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"


def epqs_elevation(lon: float, lat: float) -> Optional[float]:
    """Query USGS EPQS for elevation (meters) at lon/lat.

    Returns None on failure.
    """
    _ensure_requests()
    try:
        params = {"x": float(lon), "y": float(lat), "wkid": 4326, "units": "Meters", "includeDate": "false"}
        r = requests.get(EPQS_URL, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        return js.get("value") if isinstance(js.get("value"), (int, float)) else None
    except Exception:
        return None


def parse_mesowest_directory(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    """Parse a directory of MesoWest text/CSV files with colon-separated metadata header.

    Expects each file to have a small metadata header in the first 3–4 lines of the form:
    ```
    Station Name: Some Station
    Latitude: 40.123
    Longitude: -111.456
    Elevation: 1729 m   # optional or may be missing
    ```

    This function extracts station metadata only (not the time series payload), because
    formats vary. You can extend this to also parse the data section per your exports.
    """
    path = pathlib.Path(path)
    rows: List[dict] = []

    for file in sorted(path.glob("*")):
        try:
            df = pd.read_csv(file, nrows=4, delimiter=":", header=None, names=["field", "value"], engine="python")
            meta = {str(df.loc[i, "field"]).strip().lower(): str(df.loc[i, "value"]).strip() for i in range(len(df))}
            lat = float(meta.get("latitude", "nan"))
            lon = float(meta.get("longitude", "nan"))
            elev_txt = meta.get("elevation")
            if elev_txt is None or elev_txt.strip() == "" or elev_txt.lower().startswith("nan"):
                elev_m = epqs_elevation(lon, lat)
            else:
                # best-effort parse (numbers at start)
                try:
                    elev_m = float(str(elev_txt).split()[0])
                except Exception:
                    elev_m = epqs_elevation(lon, lat)

            rows.append({
                "meso_file": file.name,
                "name": meta.get("station name") or meta.get("name"),
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev_m,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Station catalog assembly
# -----------------------------

def to_geodataframe(df: pd.DataFrame, lon_col: str = "longitude", lat_col: str = "latitude", crs: str = "EPSG:4326"):
    """Convert a DataFrame with lon/lat columns to a GeoDataFrame (if geopandas available)."""
    if gpd is None:
        raise RuntimeError("geopandas is required for GeoDataFrame operations. Install geopandas.")
    return gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs
    )


def assemble_station_catalog(
    ucc_stations: Optional[pd.DataFrame] = None,
    snotel_station_meta: Optional[pd.DataFrame] = None,
    meso_meta: Optional[pd.DataFrame] = None,
    prefer_cols_to_drop: Sequence[str] = ("state", "country", "source",),
) -> pd.DataFrame:
    """Combine station metadata from multiple sources into one tidy table.

    Parameters
    ----------
    ucc_stations : DataFrame, optional
        Rows from `ucc_list_stations`, possibly filtered to your study stations.
    snotel_station_meta : DataFrame, optional
        DataFrame with SNOTEL station metadata (code, name, lat, lon, etc.).
    meso_meta : DataFrame, optional
        Output from `parse_mesowest_directory`.
    prefer_cols_to_drop : sequence of str
        Columns to drop if present (observed in the original notebook).

    Returns
    -------
    DataFrame
        Concatenated rows with a unified set of columns.
    """
    frames = []
    if ucc_stations is not None and not ucc_stations.empty:
        frames.append(ucc_stations.copy())
    if snotel_station_meta is not None and not snotel_station_meta.empty:
        frames.append(snotel_station_meta.copy())
    if meso_meta is not None and not meso_meta.empty:
        frames.append(meso_meta.copy())

    if not frames:
        return pd.DataFrame()

    cat = pd.concat(frames, ignore_index=True, sort=False)
    for c in prefer_cols_to_drop:
        if c in cat.columns:
            cat = cat.drop(columns=c)
    # Deduplicate by name+coords if ids differ
    key_cols = [c for c in ["primary_id", "station_id", "name", "latitude", "longitude"] if c in cat.columns]
    cat = cat.drop_duplicates(subset=key_cols, keep="first")
    return cat.reset_index(drop=True)


# -----------------------------
# Export helpers
# -----------------------------

def export_to_excel(path: Union[str, pathlib.Path],
                    **sheets: pd.DataFrame) -> pathlib.Path:
    """Write multiple DataFrames to an Excel workbook.

    Example
    -------
    >>> export_to_excel("climate_data.xlsx", ag=ag_df, snotel=snotel_df)
    """
    out = pathlib.Path(path)
    with pd.ExcelWriter(out) as writer:
        for name, df in sheets.items():
            (df or pd.DataFrame()).to_excel(writer, sheet_name=str(name))
    return out


def export_stations_gpkg(path: Union[str, pathlib.Path],
                         stations_df: pd.DataFrame,
                         layer: str = "stations",
                         crs: str = "EPSG:4326") -> Optional[pathlib.Path]:
    """Export a station catalog with geometry to GeoPackage.

    Requires geopandas.
    """
    if gpd is None:
        return None
    gdf = to_geodataframe(stations_df, crs=crs)
    out = pathlib.Path(path)
    gdf.to_file(out, driver="GPKG", layer=layer)
    return out


# -----------------------------
# Convenience high-level workflow
# -----------------------------

def compile_weather_stations(
    api_key: str,
    study_station_names: Optional[Sequence[str]] = None,
    ucc_network: str = "UAGRIMET",
    ucc_daily_range: Optional[Tuple[Union[str, dt.date, dt.datetime], Union[str, dt.date, dt.datetime]]] = None,
    snotel_map: Optional[Mapping[Union[int, str], Tuple[str, str]]] = None,
    mesowest_dir: Optional[Union[str, pathlib.Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """One-stop wrapper approximating the notebook's flow.

    Parameters
    ----------
    api_key : str
        UCC API key.
    study_station_names : list of str, optional
        If given, filter UCC stations to just these names (e.g., ["Elmo","Ferron",...]).
    ucc_network : str
        UCC network to enumerate.
    ucc_daily_range : (start, end), optional
        If provided, will download daily data for filtered UCC station_ids.
    snotel_map : mapping, optional
        Mapping like `{714: ("UT","Red Pine Ridge"), ...}` specifying SNOTEL codes.
    mesowest_dir : str or Path, optional
        Directory of MesoWest files to parse for metadata.

    Returns
    -------
    dict of DataFrames
        Keys: "ucc_stations", "ucc_daily", "snotel_daily", "mesowest_meta", "station_catalog".
    """
    out: Dict[str, pd.DataFrame] = {}

    # UCC station list
    ucc_stations = ucc_list_stations(api_key=api_key, network=ucc_network)
    if study_station_names:
        ucc_stations = ucc_stations[ucc_stations["name"].isin(study_station_names)].copy()
    out["ucc_stations"] = ucc_stations

    # UCC daily
    if ucc_daily_range and not ucc_stations.empty:
        s, e = ucc_daily_range
        station_ids = [sid for sid in ucc_stations.get("station_id", []) if pd.notna(sid)]
        out["ucc_daily"] = ucc_get_daily(api_key=api_key, station_ids=station_ids, start=s, end=e)
    else:
        out["ucc_daily"] = pd.DataFrame()

    # SNOTEL daily
    if snotel_map:
        out["snotel_daily_dict"] = snotel_get_daily(snotel_map, start=ucc_daily_range[0] if ucc_daily_range else "2016-01-01", end=ucc_daily_range[1] if ucc_daily_range else pd.Timestamp.today())
        # Optional: stack dict to a single DataFrame
        out["snotel_daily"] = pd.concat(
            [df.assign(snotel_code=str(k)) for k, df in out["snotel_daily_dict"].items()], ignore_index=True
        ) if out["snotel_daily_dict"] else pd.DataFrame()
    else:
        out["snotel_daily_dict"] = {}
        out["snotel_daily"] = pd.DataFrame()

    # MesoWest meta
    if mesowest_dir:
        out["mesowest_meta"] = parse_mesowest_directory(mesowest_dir)
    else:
        out["mesowest_meta"] = pd.DataFrame()

    # Station catalog
    out["station_catalog"] = assemble_station_catalog(
        ucc_stations=out["ucc_stations"],
        snotel_station_meta=None,  # supply if you have a SNOTEL metadata table
        meso_meta=out["mesowest_meta"],
    )

    return out
