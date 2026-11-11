"""Step 1: cut everything one state needs out of the national datasets.

Writes, under outputs/<state>/inputs/:
  friction_<mode>.tif   the state plus a margin, from the zipped 30 m friction surfaces
  women_15_49.tif       women aged 15-49 on the WorldPop grid (about 92 m), state bounding box
  population_total.tif  total population on the same grid, for checks
  ward_ids.tif          ward number of every population cell (0 = outside the state)
  wards.parquet         ward names and LGAs
  wards.geojson         simplified ward outlines for maps
  facilities.parquet    every facility in the friction window, with capability flags and grid cells
  window.json           grids, bounds and settings used
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import geo, inputs, scenarios
from .config import norm_name, settings
from .store import StateStore, write_json


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snap_cells(costs: np.ndarray, rows, cols, radius: int):
    """Move each point onto the nearest passable cell within `radius` cells.

    Returns (rows, cols, ok, moved_cells). Points with no passable cell nearby get ok=False.
    """
    h, w = costs.shape
    rows = np.asarray(rows, dtype=np.int64).copy()
    cols = np.asarray(cols, dtype=np.int64).copy()
    ok = np.zeros(rows.size, dtype=bool)
    moved = np.full(rows.size, np.nan)
    for i in range(rows.size):
        r, c = rows[i], cols[i]
        if 0 <= r < h and 0 <= c < w and costs[r, c] > 0:      # NaN compares False
            ok[i], moved[i] = True, 0.0
            continue
        r0, r1, c0, c1 = max(0, r - radius), min(h, r + radius + 1), max(0, c - radius), min(w, c + radius + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        patch = costs[r0:r1, c0:c1]
        rr, cc = np.nonzero(patch > 0)
        if rr.size == 0:
            continue
        d2 = (rr + r0 - r) ** 2 + (cc + c0 - c) ** 2
        k = int(np.argmin(d2))
        rows[i], cols[i], ok[i], moved[i] = rr[k] + r0, cc[k] + c0, True, float(np.sqrt(d2[k]))
    return rows, cols, ok, moved


def run(state: str) -> dict:
    import geopandas as gpd
    import shapely
    from affine import Affine
    from rasterio.features import rasterize

    cfg = settings()
    acc = cfg["access"]
    inputs.require(inputs.national_inputs())
    store = StateStore(state)
    rec = inputs.state_record(state)
    name = str(rec["statename"])
    poly = rec.geometry
    log(f"preparing {name}")

    # Friction windows: the state plus a margin, in the surface's own grid.
    margin_m = float(acc["margin_km"]) * 1000
    poly_utm = gpd.GeoSeries([poly], crs=geo.WGS84).to_crs(geo.UTM)
    bounds_utm = tuple(poly_utm.buffer(margin_m).total_bounds)
    transform = shape = None
    for mode in acc["modes"]:
        *_, ncols, nrows, t = inputs.friction_window(mode, bounds_utm)
        path = store.friction(mode)
        existing = inputs.raster_grid(path) if path.exists() else None
        if existing and existing[1] == (nrows, ncols) and np.allclose(existing[0], t, rtol=0, atol=1e-6):
            log(f"the {mode} friction window is already cut: reusing it")
        else:
            log(f"cutting the {mode} friction window from the zipped surface")
            arr, t = inputs.cut_friction(mode, bounds_utm)
            inputs.write_raster(path, arr, t, geo.UTM, nodata=np.nan, predictor=3)
            log(f"  {arr.shape[1]:,} x {arr.shape[0]:,} cells, {np.isnan(arr).mean():.1%} impassable")
            del arr
        if transform is not None and (t != transform or (nrows, ncols) != shape):
            raise ValueError("friction surfaces are not on the same grid")
        transform, shape = t, (nrows, ncols)

    # Population grids for the state's bounding box.
    w, s, e, n = poly.bounds
    bbox = (w - 0.01, s - 0.01, e + 0.01, n + 0.01)
    women, ptransform, pcrs = inputs.read_population_window(inputs.shared_path(cfg["paths"]["women"]), bbox)
    total, ptransform2, _ = inputs.read_population_window(inputs.shared_path(cfg["paths"]["population_total"]), bbox)
    if ptransform2 != ptransform or total.shape != women.shape:
        raise ValueError("the population layers are not on the same grid")
    inputs.write_raster(store.women, women, ptransform, pcrs, predictor=3)
    inputs.write_raster(store.population, total, ptransform, pcrs, predictor=3)

    # Wards, burnt onto the population grid.
    wards = inputs.wards(name)
    ward_ids = rasterize(((g, int(i)) for g, i in zip(wards.geometry, wards["ward_id"])),
                         out_shape=women.shape, transform=Affine(*ptransform), fill=0, dtype="int32")
    inputs.write_raster(store.ward_ids, ward_ids, ptransform, pcrs, nodata=0)
    pd.DataFrame(wards.drop(columns="geometry")).to_parquet(store.wards, index=False)
    simple = wards[["ward_id", "ward", "lga", "geometry"]].copy()
    simple["geometry"] = shapely.set_precision(simple.geometry.simplify(0.0008, preserve_topology=True).values, 1e-5)
    store.wards_geojson.write_text(simple.to_json(drop_id=True), encoding="utf-8")
    state_shape = gpd.GeoDataFrame({"state": [name]}, geometry=[poly.simplify(0.0005)], crs=geo.WGS84)
    store.state_geojson.write_text(state_shape.to_json(drop_id=True), encoding="utf-8")
    in_wards = float(women[ward_ids > 0].sum())
    log(f"  {len(wards)} wards; {in_wards:,.0f} women aged 15-49 inside them")

    # Facilities inside the friction window, with grid cells snapped onto passable ground.
    fac = inputs.facilities_all()
    x, y = geo.to_utm(fac["longitude"].to_numpy(), fac["latitude"].to_numpy())
    rows, cols = geo.rowcol(transform, x, y)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    fac = fac.loc[inside].copy()
    fac["x"], fac["y"] = x[inside], y[inside]
    fac["in_state"] = fac["state_key"] == norm_name(name)
    fac = scenarios.add_flags(fac)
    fac["operating"] = scenarios.operating(fac).to_numpy()
    for mode in acc["modes"]:
        costs, *_ = inputs.read_raster(store.friction(mode))
        r, c, ok, moved = snap_cells(costs, rows[inside], cols[inside], int(acc["snap_radius_cells"]))
        fac[f"row_{mode}"], fac[f"col_{mode}"], fac[f"ok_{mode}"], fac[f"snap_cells_{mode}"] = r, c, ok, moved
        del costs
    fac = fac.reset_index(drop=True)
    fac.to_parquet(store.facilities, index=False)
    caps = {s: int((fac[f"cap_{s}"] & fac["ok_mixed"]).sum()) for s in scenarios.names()}
    log(f"  {len(fac):,} facilities in the window ({int(fac['in_state'].sum()):,} in {name}); capable: {caps}")
    unplaced = scenarios.add_flags(inputs.facilities_without_location(name))
    unplaced_caps = {s: int(unplaced[f"cap_{s}"].sum()) for s in scenarios.names()}
    if len(unplaced):
        log(f"  {len(unplaced)} {name} facility records have no usable coordinates and are left out; "
            f"capable among them: {unplaced_caps}")

    window = {
        "state": name, "statecode": str(rec.get("statecode", "")), "crs": geo.UTM,
        "transform": list(transform), "shape": list(shape), "cell_size": [abs(transform[4]), transform[0]],
        "bounds_utm": list(bounds_utm), "margin_km": acc["margin_km"],
        "pop": {"crs": pcrs, "transform": list(ptransform), "shape": list(women.shape), "bounds": list(bbox)},
        "women_in_wards": in_wards, "wards": int(len(wards)), "facilities_in_window": int(len(fac)),
        "facilities_in_state": int(fac["in_state"].sum()), "capable_sources": caps,
        "state_records_without_coordinates": int(len(unplaced)), "capable_without_coordinates": unplaced_caps,
        "prepared": time.strftime("%Y-%m-%d %H:%M"),
    }
    write_json(store.window, window)
    log(f"prepared {name}")
    return window
