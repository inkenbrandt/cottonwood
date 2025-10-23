"""
download_big_rasters.py

Utilities extracted from `download_big_rasters.ipynb` into reusable functions.
"""
# Detected in notebook:
# URLs (sample):
#   - https://utah.openet-api.org/raster/export/stack"
#   - https://utah.openet-api.org/account/status").json()

from __future__ import annotations

import os, math, json, hashlib, logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, List

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

import rasterio as rio
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

LOG = logging.getLogger("download_big_rasters")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

def make_session(retries: int = 5, 
                 backoff: float = 0.5, 
                 timeout: int = 60) -> requests.Session:
    s = requests.Session()
    r = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff,
              status_forcelist=(429,500,502,503,504), allowed_methods=["GET","HEAD"])
    a = HTTPAdapter(max_retries=r, pool_connections=32, pool_maxsize=32)
    s.mount("http://", a); s.mount("https://", a)
    _req = s.request
    def req(method, url, **kw):
        kw.setdefault("timeout", timeout)
        return _req(method, url, **kw)
    s.request = req
    return s

def resumable_download(url: str, 
                       dest: str, 
                       session: Optional[requests.Session] = None, 
                       chunk: int = 2**20) -> str:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    session = session or make_session()
    temp = dest + ".part"
    pos = os.path.getsize(temp) if os.path.exists(temp) else 0
    headers = {"Range": f"bytes={pos}-"} if pos>0 else {}
    with session.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        mode = "ab" if pos>0 else "wb"
        total = int(r.headers.get("Content-Length", "0")) + pos
        with open(temp, mode) as f, tqdm(total=total, initial=pos, unit="B", unit_scale=True, desc=os.path.basename(dest)) as pbar:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if chunk_bytes:
                    f.write(chunk_bytes)
                    pbar.update(len(chunk_bytes))
    os.replace(temp, dest)
    return dest

@dataclass
class FetchSpec:
    direct_url: Optional[str] = None  # For COGs: same URL for all tiles, downloaded whole if used here
    wcs_url: Optional[str] = None     # For server-side tiling via WCS
    layer: Optional[str] = None
    format: str = "image/tiff"
    res: Optional[Tuple[float,float]] = None
    epsg: int = 3857

def make_tile_grid(aoi: gpd.GeoDataFrame, 
                   tile_size_m: float, 
                   out_crs: int|str = 3857) -> gpd.GeoDataFrame:
    if aoi.crs is None:
        raise ValueError("AOI has no CRS")
    g = aoi.to_crs(out_crs)
    geom = unary_union(g.geometry)
    minx, miny, maxx, maxy = geom.bounds
    nx, ny = math.ceil((maxx-minx)/tile_size_m), math.ceil((maxy-miny)/tile_size_m)
    tiles = []
    for ix in range(nx):
        for iy in range(ny):
            x0, y0 = minx + ix*tile_size_m, miny + iy*tile_size_m
            x1, y1 = min(x0+tile_size_m, maxx), min(y0+tile_size_m, maxy)
            b = box(x0,y0,x1,y1)
            if b.intersects(geom):
                tiles.append(b.intersection(geom))
    return gpd.GeoDataFrame({"tile_id": range(len(tiles))}, geometry=tiles, crs=out_crs)

def _bbox(geom) -> Tuple[float,float,float,float]:
    minx, miny, maxx, maxy = geom.bounds
    return float(minx), float(miny), float(maxx), float(maxy)

def _wcs_url(spec: FetchSpec, 
             bbox: Tuple[float,float,float,float]) -> str:
    if not (spec.wcs_url and spec.layer and spec.res):
        raise ValueError("WCS fetch requires wcs_url, layer, and res")
    xres, yres = spec.res
    return (
        f"{spec.wcs_url}?service=WCS&version=2.0.1&request=GetCoverage"
        f"&coverageId={spec.layer}&subset=Long({bbox[0]},{bbox[2]})"
        f"&subset=Lat({bbox[1]},{bbox[3]})&format={spec.format}"
        f"&RESX={xres}&RESY={yres}&CRS=EPSG:{spec.epsg}"
    )

def fetch_tile(geom, 
               spec: FetchSpec, 
               out_dir: str = "./tiles", 
               session: Optional[requests.Session] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    bbox = _bbox(geom)
    if spec.direct_url:
        h = hashlib.sha1(json.dumps(bbox).encode()).hexdigest()[:10]
        dest = os.path.join(out_dir, f"tile_{h}.tif")
        resumable_download(spec.direct_url, dest, session=session)
        return dest
    else:
        url = _wcs_url(spec, bbox)
        fname = "wcs_{:.2f}_{:.2f}_{:.2f}_{:.2f}.tif".format(*bbox).replace(".","_")
        dest = os.path.join(out_dir, fname)
        resumable_download(url, dest, session=session)
        return dest

def mosaic_tiles(tile_paths: Sequence[str], out_path: str, nodata: float|None = None) -> str:
    if not tile_paths:
        raise ValueError("No tiles to mosaic")
    srcs = [rio.open(p) for p in tile_paths]
    try:
        arr, transform = rio_merge(srcs, nodata=nodata)
        meta = srcs[0].meta.copy()
        meta.update(dict(height=arr.shape[1], width=arr.shape[2], transform=transform))
        if nodata is not None:
            meta["nodata"] = nodata
        with rio.open(out_path, "w", **meta) as dst:
            dst.write(arr)
    finally:
        for s in srcs: s.close()
    return out_path

def download_large_raster_over_aoi(
    aoi: gpd.GeoDataFrame,
    fetch_spec: FetchSpec,
    tile_size_m: float = 2048,
    out_dir: str = "./tiles",
    mosaic_out: str|None = "./mosaic.tif",
    max_tiles: int|None = None
) -> Dict[str,str]:
    grid = make_tile_grid(aoi, tile_size_m=tile_size_m, out_crs=fetch_spec.epsg)
    if max_tiles is not None:
        grid = grid.iloc[:max_tiles].copy()
    session = make_session()
    tile_paths: List[str] = []
    for _, row in tqdm(grid.iterrows(), total=len(grid), desc="Fetching tiles"):
        tile_paths.append(fetch_tile(row.geometry, fetch_spec, out_dir=out_dir, session=session))
    out = {"tiles_dir": os.path.abspath(out_dir)}
    if mosaic_out:
        out["mosaic"] = os.path.abspath(mosaic_tiles(tile_paths, mosaic_out))
    return out
