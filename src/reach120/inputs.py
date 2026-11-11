"""Readers for the shared datasets: friction surfaces, boundaries, facilities, population."""
from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd

from . import geo, tiffzip
from .config import norm_name, outputs_root, settings, shared_path

# ----------------------------------------------------------------------------- rasters

def write_raster(path, array: np.ndarray, transform, crs: str, nodata=None, predictor: int | None = None) -> None:
    """Write one band as a tiled, deflate-compressed GeoTIFF."""
    import rasterio
    from affine import Affine

    profile = {
        "driver": "GTiff", "height": array.shape[0], "width": array.shape[1], "count": 1,
        "dtype": array.dtype.name, "crs": crs, "transform": Affine(*transform[:6]),
        "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    if predictor:
        profile["predictor"] = predictor
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def read_raster(path):
    """(array, transform as a 6-tuple, crs string, nodata) for a one-band raster."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1), tuple(src.transform)[:6], src.crs.to_string(), src.nodata


# ----------------------------------------------------------------------------- friction

def friction_source(mode: str):
    cfg = settings()["paths"]["friction"][mode]
    zip_path = shared_path(cfg["zip"])
    if not zip_path.exists():
        raise FileNotFoundError(f"friction surface not found: {zip_path}. Run: python scripts/get_data.py")
    return zip_path, tiffzip.find_member(zip_path, cfg["member"])


def friction_layout(mode: str) -> tiffzip.TiffLayout:
    zip_path, member = friction_source(mode)
    return tiffzip.read_layout(zip_path, member, cache_dir=outputs_root() / "_cache" / "layouts")


def friction_window(mode: str, bounds_utm):
    """(col0, row0, ncols, nrows, transform) of the grid window covering `bounds_utm`."""
    lay = friction_layout(mode)
    if lay.epsg not in (None, 32632):
        raise ValueError(f"expected the friction surface in EPSG:32632, found EPSG:{lay.epsg}")
    xmin, ymin, xmax, ymax = bounds_utm
    pw, ph = lay.pixel_width, lay.pixel_height
    col0 = max(0, math.floor((xmin - lay.origin_x) / pw))
    col1 = min(lay.width, math.ceil((xmax - lay.origin_x) / pw))
    row0 = max(0, math.floor((lay.origin_y - ymax) / ph))
    row1 = min(lay.height, math.ceil((lay.origin_y - ymin) / ph))
    transform = (pw, 0.0, lay.origin_x + col0 * pw, 0.0, -ph, lay.origin_y - row0 * ph)
    return col0, row0, col1 - col0, row1 - row0, transform


def cut_friction(mode: str, bounds_utm) -> tuple[np.ndarray, tuple]:
    """The friction window covering `bounds_utm` (xmin, ymin, xmax, ymax), snapped to the grid.

    Returns float32 minutes per metre with NaN for impassable or missing cells, and the
    window's affine transform.
    """
    zip_path, member = friction_source(mode)
    lay = friction_layout(mode)
    col0, row0, ncols, nrows, transform = friction_window(mode, bounds_utm)
    arr = tiffzip.read_window(zip_path, member, lay, col0, row0, ncols, nrows).astype(np.float32, copy=False)
    bad = ~np.isfinite(arr) | (arr <= 0)
    if lay.nodata is not None:
        bad |= arr == np.float32(lay.nodata)
    arr[bad] = np.nan
    return arr, transform


def raster_grid(path):
    """(transform as a 6-tuple, (rows, cols)) of a raster, without reading its pixels."""
    import rasterio

    with rasterio.open(path) as src:
        return tuple(src.transform)[:6], (src.height, src.width)


# ----------------------------------------------------------------------------- boundaries

def require(paths) -> None:
    """Stop with the fix when national input files are missing."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("input data missing:\n  " + "\n  ".join(missing)
                                + "\nDownload it with: python scripts/get_data.py (see data/README.md)")


def national_inputs() -> list:
    """Every national file the prepare step reads."""
    p = settings()["paths"]
    return ([shared_path(p[k]) for k in ("facilities", "states", "wards", "women", "population_total")]
            + [shared_path(p["friction"][m]["zip"]) for m in settings()["access"]["modes"]])


def _read_vector(path, where: str | None = None):
    import geopandas as gpd

    return gpd.read_file(path, where=where, engine="pyogrio")


@functools.lru_cache(maxsize=1)
def states():
    path = shared_path(settings()["paths"]["states"])
    require([path])
    gdf = _read_vector(path)
    gdf = gdf.to_crs(geo.WGS84)
    gdf["key"] = gdf["statename"].map(norm_name)
    return gdf


def state_record(state: str):
    gdf = states()
    hit = gdf[gdf["key"] == norm_name(state)]
    if hit.empty:
        names = ", ".join(sorted(gdf["statename"]))
        raise KeyError(f"state {state!r} not found. Known states: {names}")
    return hit.iloc[0]


def state_names() -> list[str]:
    return sorted(states()["statename"].tolist())


def wards(state: str):
    """Ward polygons of a state, numbered 1..n in a stable order."""
    rec = state_record(state)
    path = shared_path(settings()["paths"]["wards"])
    name = str(rec["statename"]).replace("'", "''")
    gdf = _read_vector(path, where=f"state = '{name}'")
    if gdf.empty:     # spelling differs: fall back to normalised matching
        gdf = _read_vector(path)
        gdf = gdf[gdf["state"].map(norm_name) == norm_name(state)]
    if gdf.empty:
        raise KeyError(f"no wards found for {state}")
    gdf = gdf.to_crs(geo.WGS84)
    gdf = gdf.sort_values(["lga", "ward"], kind="stable").reset_index(drop=True)
    gdf["ward_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    keep = ["ward_id", "ward", "lga", "state", "statecode", "area_sqkm", "geometry"]
    return gdf[[c for c in keep if c in gdf.columns]]


# ----------------------------------------------------------------------------- facilities

FACILITY_COLUMNS = ["unique_id", "latitude", "longitude", "state_standard", "lga_standard", "ward_standard",
                    "facility_name", "facility_level", "facility_type", "facility_ownership", "functional",
                    "nhfr_facility_code"]


@functools.lru_cache(maxsize=1)
def facilities_all() -> pd.DataFrame:
    df = pd.read_csv(shared_path(settings()["paths"]["facilities"]), usecols=FACILITY_COLUMNS, low_memory=False)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(3.5, 14.5)) & (df["longitude"].between(2.5, 15.0))].copy()
    for col in FACILITY_COLUMNS:
        if col not in ("latitude", "longitude"):
            df[col] = df[col].fillna("").astype(str).str.strip()
    df["state_key"] = df["state_standard"].map(norm_name)
    return df.reset_index(drop=True)


def facilities_without_location(state: str) -> pd.DataFrame:
    """A state's facility records whose coordinates are missing or outside Nigeria."""
    df = pd.read_csv(shared_path(settings()["paths"]["facilities"]), usecols=FACILITY_COLUMNS, low_memory=False)
    df = df[df["state_standard"].map(norm_name) == norm_name(state)].copy()
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    bad = lat.isna() | lon.isna() | ~lat.between(3.5, 14.5) | ~lon.between(2.5, 15.0)
    out = df[bad].copy()
    for col in FACILITY_COLUMNS:
        if col not in ("latitude", "longitude"):
            out[col] = out[col].fillna("").astype(str).str.strip()
    return out


# ----------------------------------------------------------------------------- population

def read_population_window(path, bounds_lonlat):
    """Cells of a population grid covering (west, south, east, north). Nodata becomes 0."""
    import rasterio
    from rasterio.windows import Window

    w, s, e, n = bounds_lonlat
    with rasterio.open(path) as src:
        t = src.transform
        col0 = max(0, math.floor((w - t.c) / t.a))
        col1 = min(src.width, math.ceil((e - t.c) / t.a))
        row0 = max(0, math.floor((n - t.f) / t.e))
        row1 = min(src.height, math.ceil((s - t.f) / t.e))
        win = Window(col0, row0, col1 - col0, row1 - row0)
        arr = src.read(1, window=win).astype(np.float32)
        nodata = src.nodata
        transform = tuple(src.window_transform(win))[:6]
        crs = src.crs.to_string()
    bad = ~np.isfinite(arr) | (arr < 0)
    if nodata is not None:
        bad |= arr == np.float32(nodata)
    arr[bad] = 0.0
    return arr, transform, crs
