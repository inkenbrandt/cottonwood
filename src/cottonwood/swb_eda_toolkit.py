# Re-run using Polars (which has a built-in Parquet reader), then convert to pandas for plotting.
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import polars as pl

# ---------- Helpers (reuse minimal copies to keep cell compact) ----------
def guess_date_col_cols(cols) -> str:
    candidates = ["date","time","timestamp","DATE","Date","TIMESTAMP","datetime"]
    for c in candidates:
        if c in cols:
            return c
    return cols[0]

def water_year_pd(ts: pd.Series) -> pd.Series:
    return ts.dt.year + (ts.dt.month >= 10).astype(int)

def safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.true_divide(a, b)
        out[~np.isfinite(out)] = np.nan
    return out

def mk_output_dirs():
    figs = Path("/mnt/data/figures")
    figs.mkdir(parents=True, exist_ok=True)
    return figs

def pick_col(cols, names):
    for n in names:
        if n in cols:
            return n
    return None

# ---------- Load SWB via Polars ----------
swb_path = "/mnt/data/swb_wrlu_dly_data.parquet"
pl_df = pl.read_parquet(swb_path)
cols = pl_df.columns
date_col = guess_date_col_cols(cols)

# Coerce to pandas for downstream flexible ops/plots
df = pl_df.to_pandas()

# Date handling
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
df = df.sort_values(date_col).reset_index(drop=True)
df["water_year"] = water_year_pd(df[date_col])
df["month"] = df[date_col].dt.month
df["year"] = df[date_col].dt.year
df["doy"] = df[date_col].dt.dayofyear

# Identify columns
col_precip   = pick_col(df.columns, ["precip","ppt","precip_mm","P","prcp","daymet_prcp"])
col_aet      = pick_col(df.columns, ["aet","ETa","actual_et","et_actual","et","aet_mm"])
col_pet      = pick_col(df.columns, ["pet","PET","potential_et","pet_mm"])
col_recharge = pick_col(df.columns, ["recharge","rech","net_infiltration","recharge_mm"])
col_runoff   = pick_col(df.columns, ["runoff","qrunoff","runoff_mm"])
col_storage  = pick_col(df.columns, ["soil_storage","soil_moisture","soilwater","soil_water","sw","soil_storage_mm"])
col_id       = pick_col(df.columns, ["huc12","HUC12","cell_id","grid_id","hru_id","watershed","poly_id"])

core_cols = [c for c in [col_precip,col_aet,col_pet,col_recharge,col_runoff,col_storage] if c]

for c in core_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# dS
if col_storage:
    if col_id:
        df["dS"] = df.groupby(col_id, dropna=False)[col_storage].diff()
    else:
        df["dS"] = df[col_storage].diff()
else:
    df["dS"] = np.nan

# Water balance residual
P = df[col_precip] if col_precip else 0.0
A = df[col_aet] if col_aet else 0.0
Q = df[col_runoff] if col_runoff else 0.0
R = df[col_recharge] if col_recharge else 0.0
dS = df["dS"]
df["wb_residual"] = P - (A + Q + R + dS)

# Efficiencies
df["et_fraction"] = np.nan if not (col_aet and col_pet) else safe_div(df[col_aet].values, df[col_pet].values)
df["recharge_eff"] = np.nan if not (col_recharge and col_precip) else safe_div(df[col_recharge].values, df[col_precip].values)

# Aggregations
numeric_cols = [c for c in core_cols + ["dS","wb_residual","et_fraction","recharge_eff"] if c]

wy_summary = df.groupby("water_year", dropna=False)[numeric_cols].agg(["sum","mean","median","std","min","max","count"])
wy_summary.columns = ['_'.join([c for c in col if c]) for col in wy_summary.columns.to_flat_index()]
wy_summary = wy_summary.reset_index()

if col_id:
    wy_spatial_summary = df.groupby([col_id,"water_year"], dropna=False)[numeric_cols].agg(["sum","mean","median","std","min","max","count"])
    wy_spatial_summary.columns = ['_'.join([c for c in col if c]) for col in wy_spatial_summary.columns.to_flat_index()]
    wy_spatial_summary = wy_spatial_summary.reset_index()
else:
    wy_spatial_summary = None

monthly_summary = df.groupby(["water_year","month"], dropna=False)[numeric_cols].agg(["sum","mean","median","std","min","max","count"])
monthly_summary.columns = ['_'.join([c for c in col if c]) for col in monthly_summary.columns.to_flat_index()]
monthly_summary = monthly_summary.reset_index()

# Try OpenET parquet with Polars
openet_path = "/mnt/data/openet_all_dly_data_wrlu.parquet"
oe_merge_info = {"loaded": False, "on_cols": None, "n_rows_merged": 0}
merged = None
if Path(openet_path).exists():
    oe_pl = pl.read_parquet(openet_path)
    oe_cols = oe_pl.columns
    oe_date = guess_date_col_cols(oe_cols)
    oe_id = pick_col(oe_cols, ["huc12","HUC12","cell_id","grid_id","hru_id","watershed","poly_id"])
    oe_et = pick_col(oe_cols, ["et_ensemble","et","ET_ensemble","ETa","aet","openet_et"])
    oe_p  = pick_col(oe_cols, ["precip","ppt","gridmet_ppt","gridmet_precip"])
    oe_refet = pick_col(oe_cols, ["refet","reference_et","eto"])

    # Minimal columns to Pandas
    keep_cols = [c for c in [oe_date, oe_id, oe_et, oe_p, oe_refet] if c]
    oe = oe_pl.select([pl.col(k) for k in keep_cols]).to_pandas()
    oe[oe_date] = pd.to_datetime(oe[oe_date], errors="coerce", infer_datetime_format=True)
    oe = oe.sort_values(oe_date)

    oe_slim = oe.rename(columns={
        oe_date: "date_oe",
        (oe_id if oe_id else "id_oe"): "id_oe",
        (oe_et if oe_et else "et_oe"): "et_oe",
        (oe_p if oe_p else "p_oe"): "p_oe",
        (oe_refet if oe_refet else "refet_oe"): "refet_oe"
    })

    df_merge = df[[date_col] + ([col_id] if col_id else []) + numeric_cols].copy()
    df_merge = df_merge.rename(columns={date_col: "date_oe"})
    if col_id:
        df_merge = df_merge.rename(columns={col_id: "id_oe"})

    on_cols = ["date_oe"]
    if col_id and oe_id:
        on_cols.append("id_oe")

    merged = pd.merge(df_merge, oe_slim, on=on_cols, how="inner")
    if "et_oe" in merged.columns and col_aet:
        merged["et_diff"] = merged["et_oe"] - merged[col_aet]
        merged["et_mean"] = (merged["et_oe"] + merged[col_aet]) / 2.0
    if "p_oe" in merged.columns and col_precip:
        merged["p_diff"] = merged["p_oe"] - merged[col_precip]
        merged["p_mean"] = (merged["p_oe"] + merged[col_precip]) / 2.0

    oe_merge_info.update({"loaded": True, "on_cols": on_cols, "n_rows_merged": len(merged)})

# Save Excel
out_xlsx = "/mnt/data/swb_eda_summary.xlsx"
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
    df.head(1000).to_excel(xw, sheet_name="sample_rows", index=False)
    wy_summary.to_excel(xw, sheet_name="water_year_summary", index=False)
    monthly_summary.to_excel(xw, sheet_name="wy_month_summary", index=False)
    if wy_spatial_summary is not None:
        wy_spatial_summary.head(50000).to_excel(xw, sheet_name="wy_x_spatial_head", index=False)
    if merged is not None:
        merged.head(50000).to_excel(xw, sheet_name="swb_openet_merged_head", index=False)

# Display tables for quick inspection
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Water Year Summary (SWB)", wy_summary)
display_dataframe_to_user("Monthly Summary by Water Year (SWB)", monthly_summary.head(2000))
if wy_spatial_summary is not None:
    display_dataframe_to_user("Water Year x Spatial (head)", wy_spatial_summary.head(5000))
if merged is not None:
    display_dataframe_to_user("SWB ↔ OpenET daily merged (head)", merged.head(5000))

# Figures
fig_dir = mk_output_dirs()

if col_aet:
    ts = df.groupby(date_col, dropna=False)[col_aet].mean().reset_index()
    plt.figure()
    plt.plot(ts[date_col], ts[col_aet])
    plt.title("Regional mean daily AET (SWB)")
    plt.xlabel("Date")
    plt.ylabel(col_aet)
    plt.tight_layout()
    plt.savefig(fig_dir / "timeseries_mean_AET_SWB.png", dpi=150)
    plt.close()

if col_precip:
    ts = df.groupby(date_col, dropna=False)[col_precip].mean().reset_index()
    plt.figure()
    plt.plot(ts[date_col], ts[col_precip])
    plt.title("Regional mean daily Precipitation (SWB)")
    plt.xlabel("Date")
    plt.ylabel(col_precip)
    plt.tight_layout()
    plt.savefig(fig_dir / "timeseries_mean_P_SWB.png", dpi=150)
    plt.close()

plt.figure()
plt.plot(wy_summary["water_year"], wy_summary.get("wb_residual_sum", pd.Series([np.nan]*len(wy_summary))))
plt.title("Water-balance residual (sum) by water year")
plt.xlabel("Water Year")
plt.ylabel("Residual sum (units of input data)")
plt.tight_layout()
plt.savefig(fig_dir / "wb_residual_by_wy.png", dpi=150)
plt.close()

if col_id and "et_fraction" in df.columns:
    data_for_box = [grp.dropna().values for _, grp in df.groupby(col_id)["et_fraction"]]
    labels = [str(k) for k, _ in df.groupby(col_id)]
    data_for_box = data_for_box[:20]
    labels = labels[:20]
    if len(data_for_box) > 0:
        plt.figure()
        plt.boxplot(data_for_box, showfliers=False)
        plt.title("ET fraction (AET/PET) by spatial unit (first 20)")
        plt.xlabel("Spatial Unit")
        plt.ylabel("AET/PET")
        plt.xticks(range(1, len(labels)+1), labels, rotation=90)
        plt.tight_layout()
        plt.savefig(fig_dir / "et_fraction_by_spatial_box_first20.png", dpi=150)
        plt.close()

if merged is not None and "et_diff" in merged.columns and "et_mean" in merged.columns:
    plt.figure()
    plt.scatter(merged["et_mean"], merged["et_diff"], s=4)
    plt.title("Bland-Altman: OpenET vs SWB AET (daily)")
    plt.xlabel("Mean of methods (ET)")
    plt.ylabel("Difference (OpenET - SWB)")
    plt.tight_layout()
    plt.savefig(fig_dir / "bland_altman_openet_vs_swb_aet.png", dpi=150)
    plt.close()

if merged is not None and "p_diff" in merged.columns and "p_mean" in merged.columns:
    plt.figure()
    plt.scatter(merged["p_mean"], merged["p_diff"], s=4)
    plt.title("Bland-Altman: OpenET vs SWB Precip (daily)")
    plt.xlabel("Mean of methods (P)")
    plt.ylabel("Difference (OpenET - SWB)")
    plt.tight_layout()
    plt.savefig(fig_dir / "bland_altman_openet_vs_swb_p.png", dpi=150)
    plt.close()

if col_recharge and col_precip:
    tmp = df.groupby("water_year", dropna=False)[[col_precip, col_recharge]].sum().reset_index()
    tmp["rech_eff_wy"] = safe_div(tmp[col_recharge].values, tmp[col_precip].values)
    plt.figure()
    plt.scatter(tmp[col_precip], tmp["rech_eff_wy"])
    plt.title("Recharge efficiency vs total precipitation (by water year)")
    plt.xlabel("Total P (WY)")
    plt.ylabel("Recharge / P")
    plt.tight_layout()
    plt.savefig(fig_dir / "recharge_eff_vs_precip_wy.png", dpi=150)
    plt.close()

# README
readme_txt = f"""\
SWB exploratory outputs
=======================

Source parquet: {swb_path}
OpenET parquet loaded: {bool(Path(openet_path).exists())}
OpenET merge rows: {oe_merge_info['n_rows_merged'] if oe_merge_info['loaded'] else 0}
Merge keys (if used): {oe_merge_info['on_cols']}

Primary columns detected:
- date column: {date_col}
- spatial id: {col_id}
- precip: {col_precip}
- AET: {col_aet}
- PET: {col_pet}
- recharge: {col_recharge}
- runoff: {col_runoff}
- storage: {col_storage}

Derived:
- dS: daily storage change (diff by spatial id where available)
- wb_residual = P - (AET + Runoff + Recharge + dS)
- et_fraction = AET / PET
- recharge_eff = Recharge / P

Outputs:
- Excel summary: /mnt/data/swb_eda_summary.xlsx
- Figures directory: /mnt/data/figures
"""
with open("/mnt/data/SWB_EDA_README.txt", "w") as f:
    f.write(readme_txt)

readme_txt
