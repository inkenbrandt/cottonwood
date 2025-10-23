"""
Streamlined utilities extracted and refactored from 'GEE Summary-Cottonwood.ipynb'.
"""

from __future__ import annotations
import os, json, math, pathlib, warnings
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    plt = None  # type: ignore
try:
    import statsmodels.api as sm  # noqa: F401
except Exception:
    sm = None  # type: ignore
try:
    import folium  # noqa: F401
except Exception:
    folium = None  # type: ignore
from scipy import odr

ACFT_PER_INCH_PER_ACRE = 1.0 / 12.0


def to_acft(depth_inches: pd.Series | np.ndarray) -> pd.Series:
    """Convert water depth from inches to acre-feet."""
    s = pd.Series(depth_inches)
    return s * ACFT_PER_INCH_PER_ACRE


def water_year(dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Return the water year for each timestamp."""
    idx = pd.to_datetime(dates)
    return (idx + pd.offsets.MonthBegin(-9)).year


def effective_precip(et: pd.Series, p: pd.Series) -> pd.Series:
    """Compute effective precipitation (ET minus precipitation)."""
    return et.sub(p)


def bland_altman(m1: pd.Series, m2: pd.Series, sd_limit: float = 1.96):
    """Calculate Bland–Altman statistics for two measurement series."""
    a = pd.Series(m1).astype(float)
    b = pd.Series(m2).astype(float)
    means = (a + b) / 2.0
    diffs = a - b
    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    return means, diffs, mean_diff, std_diff


def odr_line(x: pd.Series, y: pd.Series):
    """Fit an orthogonal distance regression line and return slope, intercept, and R²."""
    x = pd.Series(x).astype(float).to_numpy()
    y = pd.Series(y).astype(float).to_numpy()

    def f(B, X):
        return B[0] * X + B[1]

    model = odr.Model(f)
    data = odr.RealData(x, y)
    odr_inst = odr.ODR(data, model, beta0=[1.0, 0.0])
    out = odr_inst.run()
    a, b = out.beta
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return float(a), float(b), float(r2)


def call_api(
    endpoint: str, api_key: str, args: Optional[dict] = None, get: bool = True
) -> dict:
    """Call the OpenET API and return the decoded JSON payload."""
    import requests

    base = "https://openet-api.org"
    url = base + endpoint
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if get:
        resp = requests.get(url, headers=headers, params=args or {})
    else:
        resp = requests.post(url, headers=headers, json=args or {})
    resp.raise_for_status()
    return resp.json()


# ---- Functions extracted from the notebook (as-is where possible) ----


def Mapdisplay(center, dicc, Tiles="OpensTreetMap", zoom_start=10):
    """Displays an interactive map
    :param center: Center of the map (Latitude and Longitude).
    :param dicc: Earth Engine Geometries or Tiles dictionary
    :param Tiles: Mapbox Bright,Mapbox Control Room,Stamen Terrain,Stamen Toner,stamenwatercolor,cartodbpositron.
    :zoom_start: Initial zoom level for the map.
    :return: A folium.Map object.
    """
    mapViz = folium.Map(location=center, tiles=Tiles, zoom_start=zoom_start)
    for k, v in dicc.items():
        if ee.image.Image in [type(x) for x in v.values()]:
            folium.TileLayer(
                tiles=v["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                overlay=True,
                name=k,
            ).add_to(mapViz)
        else:
            folium.GeoJson(data=v, name=k).add_to(mapViz)
    mapViz.add_child(folium.LayerControl())
    return mapViz


def bland_altmann(m1, m2, sd_limit=1.96):
    """
     Bland-Altman Plot.

     A Bland-Altman plot is a graphical method to analyze the differences
     between two methods of measurement. The mean of the measures is plotted
     against their difference.

     Parameters
     ----------
     m1, m2: pandas Series or array-like

    Returns
     -------
     ax: matplotlib Axis object
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if len(m1) != len(m2):
        raise ValueError("m1 does not have the same length as m2.")
    if sd_limit < 0:
        raise ValueError("sd_limit ({}) is less than 0.".format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    return means, diffs


def scatterColor(x0, y, w):
    """Creates scatter plot with points colored by variable.
    All input arrays must have matching lengths

    Arg:
        x0 (array):
            array of x values
        y (array):
            array of y values
        w (array):
            array of scalar values

    Returns:
        slope and intercept of best fit line

    """
    import matplotlib as mpl
    import matplotlib.cm as cm

    cmap = plt.cm.get_cmap("RdYlBu")
    norm = mpl.colors.Normalize(vmin=w.min(), vmax=w.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(w)
    sc = plt.scatter(x0, y, label="", color=m.to_rgba(w))

    xa = sm.add_constant(x0)

    est = sm.RLM(y, xa).fit()
    r2 = sm.WLS(y, xa, weights=est.weights).fit().rsquared
    slope = est.params[1]

    x_prime = np.linspace(np.min(x0), np.max(x0), 100)[:, np.newaxis]
    x_prime = sm.add_constant(x_prime)
    y_hat = est.predict(x_prime)

    const = est.params[0]
    y2 = [i * slope + const for i in x0]

    plt.xlabel("Barometric Pressure (ft water)")
    plt.ylabel("Transducer Pressure (ft water)")
    lin = linregress(x0, y)
    x1 = np.arange(np.min(x0), np.max(x0), 0.1)
    y1 = [i * lin[0] + lin[1] for i in x1]
    y2 = [i * slope + const for i in x1]
    plt.plot(
        x1,
        y1,
        c="g",
        label="simple linear regression m = {:.2f} b = {:.0f}, r^2 = {:.2f}".format(
            lin[0], lin[1], lin[2] ** 2
        ),
    )
    plt.plot(
        x1,
        y2,
        c="r",
        label="rlm regression m = {:.2f} b = {:.0f}, r2 = {:.2f}".format(
            slope, const, r2
        ),
    )
    plt.legend()
    cbar = plt.colorbar(m)

    cbar.set_label("Julian Date")

    return slope, const, r2


def getlutrend(wrluy, crop, irr=None, adj=True):
    """
    wrluy = yearly et by landuse with year as index
    crop = type of landuse; ex 'Corn'
    irr = irr method; ex 'Sprinkler'
    """
    if irr:
        crp = wrluy[wrluy["IRR_Method"] == irr].dropna(subset=["et_acft_adj"])
    else:
        crp = wrluy.dropna(subset=["et_acft_adj"])
    if adj:
        crpyr = (
            crp.groupby(["Description", crp.index])
            .sum()
            .loc[crop, "et_acft_adj"]
            .sort_index()
        )
        # alfyr.plot()
    else:
        crpyr = (
            crp.groupby(["Description", crp.index])
            .sum()
            .loc[crop, "et_acft"]
            .sort_index()
        )
        # alfyr.plot(color='red')
    return crpyr


def plotlu(crpyr, crop, ylab="ET (acft)", axis=None):
    """Plot a crop-specific time series and annotate it with Mann–Kendall trends."""
    if axis:
        pass
    else:
        fig, axis = plt.subplots(1, 1, figsize=(12, 9))
    axis.plot(crpyr.index, crpyr, label=crop)
    plotmk(crpyr, axis)
    axis.set_ylabel(ylab)
    axis.set_title(crop)
    axis.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


def plotmk(crpyr, axis, units="ac-ft/yr", color=None):
    """This function plots the Mann Kendall trend
    crpyr = crop year
    axis = matplotlib plotting axis
    """
    seas = mk.yue_wang_modification_test(crpyr)
    mklist = [
        seas.trend,
        seas.h,
        seas.p,
        seas.z,
        seas.Tau,
        seas.s,
        seas.slope,
        seas.intercept,
    ]
    if mklist[1]:
        frst = crpyr.first_valid_index()
        lst = crpyr.last_valid_index()
        x = range(frst, lst)
        lab = f"MK {mklist[-2]:.2f} {units}"
        if color:
            clr = color
        else:
            if mklist[0] == "increasing":
                clr = "red"
            else:
                clr = "green"
        axis.plot(
            x, [(i - 1985) * mklist[-2] + mklist[-1] for i in x], label=lab, color=clr
        )


def matchdiv(df, hucbasin):
    """matches dataset to hydro div information and adds hydrodiv information
    df = dataset
    hucbasin = huc geodataframe
    """
    df["huc_ind"] = df["system:index"].apply(lambda x: x.split("_")[0][-5:], 1)
    df = df.set_index("huc_ind")
    match_dict = hucbasin.reset_index().set_index("huc_ind")["huc_12"].to_dict()
    df["huc12"] = df.index.map(match_dict)
    df["hucname"] = df["huc12"].map(hucbasin["hu_12_name"].to_dict())
    return df


def grphuc(x):
    """function to be applied to groupby technique on hucs and SEEBOP data"""
    d = {}
    d["ValleySide"] = x["ValleySide"][0]
    d["Topography"] = x["Topography"][0]
    d["wateryear"] = x["wateryear"][0]
    d["AreaAcres"] = x["AreaAcres"].max()
    d["et_acft"] = x["et_acft"].mean()
    d["MEAN"] = x["MEAN"].sum()
    return pd.Series(
        d,
        index=["AreaAcres", "Topography", "ValleySide", "et_acft", "MEAN", "wateryear"],
    )


def grpdaymet(x):
    """function to be applied to groupby technique on hucs and DAYMET data"""
    d = {}
    d["ppt_daymet"] = x["PR_acft"].sum()
    # SWE gives snapshot of snowpack for current day
    d["swe_daymet"] = x["SWE_acft"].median()
    return pd.Series(d, index=["ppt_daymet", "swe_daymet"])


def plotcomparion(
    df, dataset_1, dataset_2, datatype, slope_adj=1.0, offset_adj=0, plot=True
):
    """Compare two datasets with regression and Bland–Altman diagnostics."""
    df1 = (
        df[[f"{dataset_1}_{datatype}", f"{dataset_2}_{datatype}"]]
        .interpolate(method="time")
        .dropna()
    )

    y = df1[f"{dataset_2}_{datatype}"]
    x = df1[f"{dataset_1}_{datatype}"] * slope_adj - offset_adj
    res = sm.OLS(y, x).fit()

    y1 = df1[f"{dataset_2}_{datatype}"]
    x1 = df1[f"{dataset_1}_{datatype}"] * res.params[0]
    res1 = sm.OLS(y1, x1).fit()
    means1, diffs1 = bland_altmann(x1, y1)

    means, diffs = bland_altmann(x, y)

    if plot:
        paramsig, ax = plt.subplots(2, 1, figsize=(12, 10))

        ax[0].scatter(x, y)
        ax[0].plot(x, res.fittedvalues, "r--", label="OLS")
        ax[0].plot(
            np.linspace(x.min(), x.max(), 1000),
            np.linspace(x.min(), x.max(), 1000),
            color="black",
            label="1-to-1 Line",
        )
        ax[0].legend(loc="best")
        ax[0].grid()

        if datatype == "ppt":
            labelpart = "Precipitation"
        elif datatype == "aet":
            labelpart = "Evapotranspiration"

        ax[0].set_xlabel(f"{dataset_1.upper()} {labelpart} (acre-ft/mo)")
        ax[0].set_ylabel(f"{dataset_2.upper()} {labelpart} (acre-ft/mo)")

        j = sm.graphics.mean_diff_plot(x, y, ax=ax[1])

    if res.params[0] > 1:
        relsize = "bigger"
    else:
        relsize = "smaller"
    print(
        f"{dataset_2} is {res.params[0]:0.3f} times {relsize} than {dataset_1} (r2 = {res.rsquared:0.2f})"
    )
    print(f"Mean diff is {np.mean(diffs):0.2f} and offset is {np.mean(diffs1):0.2f}")
    print(f"Model MSE = {res.mse_resid}")

    return np.mean(diffs), np.mean(diffs1), res.params[0]


def call_api(endpoint, api_key, args=None, get=True):
    """Using user specified inputs, returns data from OpenET Raster API.

    Args:
        endpoint (str): Raster API endpoint

        api_key (str): Required api access key

        args (dictionary): User specified arguments for api call

        get (bool): use True if a get request and False if a post request

    Returns:
        result (object): An object of Raster API results
    """

    if args:
        args = args
    else:
        args = {}

    # api server address
    server = "https://openet.dri.edu/"

    # initialize request url
    url = server + endpoint

    # create header
    header = {"Authorization": api_key}

    if get:
        # make api get request
        resp = requests.get(url=url, headers=header, params=args, verify=False)

    else:
        # make api post request
        resp = requests.post(
            url=url, headers=header, data=json.dumps(args), verify=False
        )

    # view results
    # print(resp.url)
    # print(resp.content)
    return resp


def pull_point_et(
    api_key,
    variable="et",
    yearst=2016,
    yearend=2022,
    latitude=39.22,
    longitude=-111.0698,
):
    """Retrieve a point time series from the OpenET Raster API and return it as a DataFrame."""
    # import time
    args = {
        "start_date": f"{yearst}-01-01",  # inclusive starting date
        "end_date": f"{yearend}-12-31",  # inclusive completion date
        # spatial options
        "lon": longitude,  # longitude,latitude region
        "lat": latitude,
        "interval": "daily",
        "output_file_format": "json",
        # OpenET options
        "variable": variable,  # variable to retrieve (ndvi, etof, eto, et, pr)
        "model": "ensemble",  # model selection (ensemble, geesebal, ssebop, eemetric, sims, disalexi, ptjpl)
        "ref_et_source": "gridmet",  # reference et collection (cimis, gridmet)
        "provisional": True,
        # data processing options
        "pixel_aggregation": "sum",  # spatial aggregation method
        "units": "english",
    }

    # query result
    resp = call_api("raster/timeseries/point?", api_key=api_key, args=args, get=True)
    print(resp.text)
    # time.sleep(5)
    etdf = pd.DataFrame(resp.json())
    etdf["time"] = pd.to_datetime(etdf["time"])
    etdf = etdf.set_index("time")
    return etdf


def pull_daymet_point(yearst=2016, yearend=2022, latitude=39.22, longitude=-111.0698):
    """Download DAYMET single-pixel data and return it as a time-indexed DataFrame."""
    from requests import Session, Request

    urlbase = "https://daymet.ornl.gov/single-pixel/api/data?"
    args = {
        "lat": latitude,
        "lon": longitude,
        "vars": "dayl,prcp,srad,swe,tmax,tmin,vp",
        "start": f"{yearst}-01-01",
        "end": f"{yearend}-12-31",
    }
    s = Session()
    p = Request("GET", urlbase, params=args).prepare()
    df = pd.read_csv(p.url, skiprows=7)
    df["datetime"] = pd.to_datetime(df["year"] * 1000 + df["yday"], format="%Y%j")
    df = df.set_index("datetime")
    return df


def summarizeoet(key, dfs, add_area=False):
    """Summarizes and transposes data from the OpenET API"""
    colnm = key.split("_")[-2:]
    colnm[0] = colnm[0][:3]
    col_name = "_".join(colnm)

    df = dfs[key]
    df["time"] = pd.to_datetime(df["time"])
    dfgm = df[df["time"].dt.month.isin([3, 4, 5, 6, 7, 8, 9, 10])]
    dfgm["year"] = dfgm["time"].dt.year
    # dfgm['month'] = dfgm['time'].dt.month
    dfyr = dfgm.groupby(["ID", "year"]).sum(numeric_only=True)  # .unstack(1)
    dfyr = dfyr.drop(["area_hectares"], axis=1)  # .droplevel(0,axis=1)

    dfmo = df.groupby(["ID", "time"]).sum(numeric_only=True)  # .unstack(1)
    dfmo = dfmo.drop(["area_hectares"], axis=1)  # .droplevel(0,axis=1)

    # return dfyr
    return dfmo, dfyr


def grpterra(x):
    """function to be applied to groupby technique on hucs and Terra data"""
    d = {}
    d["terra_et"] = x["aet_volume_af"].sum()
    d["terra_ppt"] = x["pr_volume_af"].sum()
    d["terra_ro"] = x["ro_volume_af"].sum()
    d["terra_def"] = x["def_volume_af"].sum()
    d["terra_soil"] = x["soil_volume_af"].max()
    d["terra_swe"] = x["swe_diff"].sum()
    d["terra_swe_max"] = x["swe_diff"].sum()
    d["wateryear"] = x["wateryear"].max()
    return pd.Series(
        d,
        index=[
            "terra_et",
            "terra_ppt",
            "terra_ro",
            "terra_soil",
            "terra_def",
            "terra_swe",
            "wateryear",
        ],
    )


def grpdaymet(x):
    """function to be applied to groupby technique on hucs and DAYMET data"""
    d = {}
    d["ppt_daymet"] = x["PR_acft"].sum()
    # SWE gives snapshot of snowpack for current day
    d["swe_daymet"] = x["SWE_acft"].median()
    return pd.Series(d, index=["ppt_daymet", "swe_daymet"])


def f(x):
    """Summarize PRISM precipitation into descriptive statistics."""
    d = {}
    fld = "prism_ppt_in"
    # fld = 'prism_ppt_volume_af'
    d["95th %tile"] = x[fld].quantile(0.95)
    d["median"] = x[fld].median()
    d["mean"] = x[fld].mean()
    d["5th %tile"] = x[fld].quantile(0.05)
    return pd.Series(d, index=["95th %tile", "median", "mean", "5th %tile"])


def g(x):
    """Aggregate PRISM precipitation to mean depth and associated water year."""
    d = {}
    fld = "prism_ppt_in"
    d["mean"] = x[fld].mean()
    d["wateryear"] = x["wateryear"].min()
    return pd.Series(d, index=["mean", "wateryear"])


def grpirrmo(x):
    """function to be applied to groupby technique on hucs and SEEBOP data"""
    d = {}
    for col in x:
        if col == "SubsurfaceSM" or col == "SurfaceSM":
            d[col] = x[col].max()
        elif col == "sensor" or col == "year":
            pass
        else:
            d[col] = x[col].sum()

    return pd.Series(d, index=list(d.keys()))


def odr_line(p, x):
    """The line of best fit."""
    # unpack the parameters:
    y = p * x
    return y


def perform_odr(x, y, xerr, yerr):
    """Finds the ODR for data {x, y} and returns the result"""
    linear = odr.Model(odr_line)
    mydata = odr.Data(x, y, wd=None, we=None)
    myodr = odr.ODR(mydata, linear, beta0=[0])
    output = myodr.run()
    return output


def line(x, a):
    """The line of best fit."""
    # unpack the parameters:
    y = a * x
    return y


# Backwards-compatible alias with a corrected name
try:

    def plot_comparison(*args, **kwargs):
        return plotcomparion(*args, **kwargs)

except NameError:
    pass
