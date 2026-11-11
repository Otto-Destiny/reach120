"""Step 2: travel time to the nearest capable facility, for every scenario and transport mode.

For each run the 30 m result is saved as whole minutes (about 20 MB compressed) and moved onto
the population grid (about 92 m) by taking the **median** of the 30 m cells in each population
cell. Not the mean: the friction surface treats rivers as barriers, which leaves a few 30 m cells
enclosed with travel times in the thousands of minutes, and a mean lets one of them push a whole
population cell beyond two hours. In Kano that artefact alone put 4,379 women "beyond two hours
by vehicle", most of them in cells whose typical travel time was under 20 minutes.

Existing 30 m rasters are reused, so changing the aggregation does not repeat the searches.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd

from . import dijkstra, inputs, scenarios
from .config import settings
from .prepare import log
from .store import StateStore, read_json, write_json

UNREACHABLE = 65535


def write_minutes(path, minutes: np.ndarray, transform, crs: str) -> None:
    """Whole minutes as uint16; 65535 marks unreachable or missing."""
    q = np.where(np.isfinite(minutes), np.clip(np.rint(minutes), 0, UNREACHABLE - 1), UNREACHABLE).astype(np.uint16)
    inputs.write_raster(path, q, transform, crs, nodata=UNREACHABLE, predictor=2)


def read_minutes(path) -> tuple[np.ndarray, tuple, str]:
    """Minutes as float32 with NaN for unreachable, plus transform and CRS."""
    arr, transform, crs, _ = inputs.read_raster(path)
    out = arr.astype(np.float32)
    out[arr == UNREACHABLE] = np.nan
    return out, transform, crs


def to_pop_grid(dist: np.ndarray, src_transform, src_crs: str, dst_transform, dst_shape, dst_crs: str) -> np.ndarray:
    """Median of the 30 m travel times inside each population cell (see the module docstring).
    Unreached cells are ignored; a population cell with no reachable 30 m cell is NaN."""
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    src = np.where(np.isfinite(dist), dist, np.nan).astype(np.float32)
    dst = np.full(tuple(dst_shape), np.nan, dtype=np.float32)
    reproject(source=src, destination=dst, src_transform=Affine(*src_transform[:6]), src_crs=src_crs,
              src_nodata=np.nan, dst_transform=Affine(*dst_transform[:6]), dst_crs=dst_crs, dst_nodata=np.nan,
              resampling=Resampling.med, num_threads=4)
    return dst


def source_cells(fac: pd.DataFrame, mask: pd.Series, mode: str, cols: int) -> np.ndarray:
    use = mask & fac[f"ok_{mode}"]
    return (fac.loc[use, f"row_{mode}"].to_numpy(np.int64) * cols + fac.loc[use, f"col_{mode}"].to_numpy(np.int64))


def run(state: str, scenario_list=None, modes=None, reuse: bool = True) -> list[dict]:
    store = StateStore(state)
    win = read_json(store.window)
    acc = settings()["access"]
    scenario_list = scenario_list or scenarios.names()
    modes = modes or acc["modes"]
    keep = acc.get("keep_30m", "all")
    pop = win["pop"]
    fac = pd.read_parquet(store.facilities)
    results = []
    for mode in modes:
        costs = transform = crs = None
        for s in scenario_list:
            src = source_cells(fac, fac[f"cap_{s}"], mode, int(win["shape"][1]))
            fingerprint = hashlib.sha1(np.sort(src).tobytes()).hexdigest()
            path30, meta_path = store.tt30(s, mode), store.run_meta(s, mode)
            old = read_json(meta_path) if meta_path.exists() else {}
            same_sources = old.get("sources_hash", fingerprint if old.get("sources") == src.size else None) == fingerprint
            if (reuse and path30.exists() and same_sources
                    and path30.stat().st_mtime > store.friction(mode).stat().st_mtime):
                dist, transform30, crs30 = read_minutes(path30)
                meta = {**old, "sources_hash": fingerprint}
                write_json(meta_path, meta)
                secs, reused = meta["seconds"], True
            else:
                if costs is None:
                    costs, transform, crs, _ = inputs.read_raster(store.friction(mode))
                    costs = np.ascontiguousarray(costs, dtype=np.float32)
                t0 = time.time()
                dist = dijkstra.travel_time(costs, src, (abs(transform[4]), transform[0]))
                secs, reused, transform30, crs30 = time.time() - t0, False, transform, crs
                if keep == "all" or f"{s}/{mode}" in keep:
                    write_minutes(path30, dist, transform30, crs30)
                reached = int(np.isfinite(dist).sum())
                passable = int(np.count_nonzero(costs > 0))
                meta = {"state": win["state"], "scenario": s, "mode": mode, "sources": int(src.size),
                        "sources_hash": fingerprint, "seconds": round(secs, 1), "cells": int(costs.size), "passable_cells": passable,
                        "reached_share_of_passable": reached / max(passable, 1),
                        "max_minutes": float(np.nanmax(np.where(np.isfinite(dist), dist, np.nan))) if reached else None,
                        "computed": time.strftime("%Y-%m-%d %H:%M")}
                write_json(store.run_meta(s, mode), meta)
            ttpop = to_pop_grid(dist, transform30, crs30, pop["transform"], pop["shape"], pop["crs"])
            write_minutes(store.ttpop(s, mode), ttpop, pop["transform"], pop["crs"])
            results.append(meta)
            log(f"  {s:12} {mode:6} {meta['sources']:4} sources, "
                + ("reused the 30 m result" if reused else f"{secs:6.1f} s, reached {meta['reached_share_of_passable']:.1%} of passable cells"))
            del dist, ttpop
        del costs
    return results
