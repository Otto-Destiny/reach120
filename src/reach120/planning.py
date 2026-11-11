"""Step 4: the inputs to upgrade planning.

* **Demand points.** Women more than two hours from care, summed into blocks of 5 x 5
  population cells (about 460 m), with a weight for every planner scenario. Each block stands
  at one 30 m cell, taken from the block's most populous uncovered cell, that is itself beyond
  two hours on the 30 m grid. That keeps the planner consistent with the coverage tables: a
  stand-in on the near side of a river barrier would make the women behind it look easy to reach.
* **Candidates.** Facilities that could be upgraded: hospitals and other secondary-level
  facilities not yet counted as capable, plus the best-placed public primary health centre
  in each LGA. Each has a cost tier (see config/costs.yaml).
* **Reach.** For each candidate, which demand points would be within two hours if it were
  upgraded. One capped, pruned search per candidate on the 30 m grid (see dijkstra.py).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import dijkstra, geo, inputs, scenarios
from .access import read_minutes
from .config import settings
from .prepare import log, snap_cells
from .store import StateStore, read_json, write_json

HOSPITAL_TYPES = {"Hospital", "General Hospital", "Specialist Hospital", "Cottage Hospital",
                  "Federal Medical Center", "Teaching/Tertiary Hospital", "Teaching Hospital"}
NOT_CANDIDATES = {"Eye Hospital", "Dental Hospital", "Eye Clinic", "Dental Clinic",
                  "Diagnostic / Laboratory Center", "Nursing / Convalescent Home", "Unclassified", "Unknown"}
PHC_TYPES = {"Primary Health Center", "Primary Health Clinic", "Health Center"}


def build_demand(state: str, mode: str) -> pd.DataFrame:
    store = StateStore(state)
    plan_cfg = settings()["planning"]
    acc = settings()["access"]
    block, limit = int(plan_cfg["block_cells"]), float(acc["target_min"])
    names = plan_cfg["scenarios"]
    women, ptransform, _, _ = inputs.read_raster(store.women)
    ward_ids, *_ = inputs.read_raster(store.ward_ids)
    wards = pd.read_parquet(store.wards).set_index("ward_id")
    base = (ward_ids > 0) & (women > 0)
    uncovered, minutes = {}, {}
    for s in names:
        m, _, _ = read_minutes(store.ttpop(s, mode))
        minutes[s] = m
        uncovered[s] = base & ~(m <= limit)
    union = np.logical_or.reduce([uncovered[s] for s in names])
    rr, cc = np.nonzero(union)
    if rr.size == 0:
        log("  no women are beyond the target in any planner scenario")
        df = pd.DataFrame({c: pd.Series(dtype=float) for c in ["demand_id", "lon", "lat", "ward_id", "cell", "women",
                                                               *[f"w_{s}" for s in names]]})
        df.to_parquet(store.demand(mode), index=False)
        return df
    w = women[rr, cc].astype(np.float64)
    n_bc = -(-women.shape[1] // block)
    key = (rr // block) * n_bc + (cc // block)
    _, inv = np.unique(key, return_inverse=True)
    df = pd.DataFrame({"women": np.bincount(inv, weights=w)})
    for s in names:
        u = uncovered[s][rr, cc]
        df[f"w_{s}"] = np.bincount(inv, weights=w * u)
        t = minutes[s][rr, cc]
        fin = u & np.isfinite(t)
        tw = np.bincount(inv, weights=np.where(fin, w * np.nan_to_num(t), 0.0))
        ww = np.bincount(inv, weights=np.where(fin, w, 0.0))
        df[f"minutes_{s}"] = np.where(ww > 0, tw / np.where(ww > 0, ww, 1), np.nan)

    # Each block stands at its most populous uncovered cell...
    order = np.lexsort((-w, inv))
    first = order[np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]]
    top_r, top_c = rr[first], cc[first]
    df["ward_id"] = ward_ids[top_r, top_c]
    df["ward"] = wards["ward"].reindex(df["ward_id"]).to_numpy()
    df["lga"] = wards["lga"].reindex(df["ward_id"]).to_numpy()
    # ...and, within that cell's footprint, at a passable 30 m cell beyond the target (nearest the
    # centre), using the strictest scenario's 30 m travel times.
    costs, ftransform, _, _ = inputs.read_raster(store.friction(mode))
    field, _, _ = read_minutes(store.tt30(names[0], mode))
    a, _, c0, _, e, f0 = ptransform[:6]
    offsets = [(dy, dx) for dy in (0.5, 1 / 6, 5 / 6) for dx in (0.5, 1 / 6, 5 / 6)]     # centre first
    best_cell = np.full(len(df), -1, dtype=np.int64)
    best_far = np.zeros(len(df), dtype=bool)
    rows30, cols30 = costs.shape
    for dy, dx in offsets:
        lon = c0 + (top_c + dx) * a
        lat = f0 + (top_r + dy) * e
        x, y = geo.to_utm(lon, lat)
        r30, c30 = geo.rowcol(ftransform, x, y)
        inside = (r30 >= 0) & (r30 < rows30) & (c30 >= 0) & (c30 < cols30)
        flat = np.where(inside, r30 * cols30 + c30, 0)
        passable = inside & (costs.ravel()[flat] > 0)
        t30 = np.where(passable, field.ravel()[flat], np.nan)
        far = passable & ~(t30 <= limit)
        take = (best_cell < 0) & passable | (~best_far & far)
        best_cell[take] = flat[take]
        best_far[take] = far[take]
    centre_lon, centre_lat = geo.cell_centres(ptransform, top_r, top_c)
    df["lon"], df["lat"], df["cell"] = centre_lon, centre_lat, best_cell
    df["consistent"] = best_far
    bad = best_cell < 0
    if bad.any():
        log(f"  {int(bad.sum())} demand points ({df.loc[bad, 'women'].sum():,.0f} women) have no passable ground; dropped")
    if (~best_far & ~bad).any():
        log(f"  {int((~best_far & ~bad).sum())} demand points have no 30 m cell beyond the target in their footprint "
            f"({df.loc[~best_far & ~bad, 'women'].sum():,.0f} women); kept at their best cell")
    df = df[~bad].reset_index(drop=True)
    df.insert(0, "demand_id", np.arange(len(df), dtype=np.int64))
    df.to_parquet(store.demand(mode), index=False)
    log(f"  {mode}: {len(df):,} demand points; women beyond {int(limit)} min: "
        + ", ".join(f"{s} {df[f'w_{s}'].sum():,.0f}" for s in names))
    return df


def build_candidates(state: str, demand: pd.DataFrame, mode: str) -> pd.DataFrame:
    store = StateStore(state)
    plan_cfg = settings()["planning"]
    names = plan_cfg["scenarios"]
    strictest = names[0]
    fac = pd.read_parquet(store.facilities)
    fac = fac[fac["in_state"] & fac["operating"] & fac[f"ok_{mode}"] & ~fac["facility_type"].isin(NOT_CANDIDATES)]
    not_capable = ~fac[f"cap_{strictest}"]
    hosp = fac[not_capable & fac["facility_type"].isin(HOSPITAL_TYPES)].assign(tier="hospital")
    sec = fac[not_capable & ~fac["facility_type"].isin(HOSPITAL_TYPES) & (fac["facility_level"] == "Secondary")].assign(tier="secondary")
    pool = pd.concat([hosp, sec])

    dlon, dlat = demand["lon"].to_numpy(), demand["lat"].to_numpy()
    dw = demand[f"w_{strictest}"].to_numpy() if len(demand) else np.zeros(0)

    def women_within(lons, lats, km):
        if dw.size == 0 or len(lons) == 0:
            return np.zeros(len(lons))
        out = np.empty(len(lons))
        for i in range(0, len(lons), 64):
            d = geo.haversine_km(np.asarray(lons[i:i + 64])[:, None], np.asarray(lats[i:i + 64])[:, None], dlon[None, :], dlat[None, :])
            out[i:i + 64] = (d <= km) @ dw
        return out

    phc = fac[fac["facility_type"].isin(PHC_TYPES) & (fac["facility_ownership"] == "Public") & ~fac["unique_id"].isin(pool["unique_id"])]
    picks = []
    for _, group in phc.groupby("lga_standard"):
        score = women_within(group["longitude"].to_numpy(), group["latitude"].to_numpy(), float(plan_cfg["phc_radius_km"]))
        if score.size and score.max() > 0:
            picks.append(group.iloc[[int(np.argmax(score))]])
    if picks:
        pool = pd.concat([pool, pd.concat(picks).assign(tier="phc")])
    pool = pool.drop_duplicates("unique_id").copy()
    pool["straight_score"] = women_within(pool["longitude"].to_numpy(), pool["latitude"].to_numpy(),
                                          float(plan_cfg["straight_line_radius_km"]))
    cap = int(plan_cfg["max_candidates"])
    if len(pool) > cap:
        log(f"  {len(pool)} candidates; keeping the {cap} with most uncovered women within "
            f"{plan_cfg['straight_line_radius_km']} km")
        pool = pool.sort_values("straight_score", ascending=False).head(cap)
    cols = int(read_json(store.window)["shape"][1])
    pool["cell"] = pool[f"row_{mode}"].astype(np.int64) * cols + pool[f"col_{mode}"].astype(np.int64)
    keep = ["unique_id", "facility_name", "facility_type", "facility_level", "facility_ownership", "functional",
            "lga_standard", "ward_standard", "latitude", "longitude", "tier", "straight_score", "cell",
            *[f"cap_{s}" for s in names]]
    pool = pool[keep].rename(columns={"lga_standard": "lga", "ward_standard": "ward"})
    pool = pool.sort_values(["tier", "lga", "facility_name"]).reset_index(drop=True)
    pool.insert(0, "cand_id", np.arange(len(pool), dtype=np.int64))
    pool.to_parquet(store.candidates(mode), index=False)
    log(f"  {mode}: {len(pool)} candidates: " + ", ".join(f"{t} {n}" for t, n in pool["tier"].value_counts().items()))
    return pool


def build_reach(state: str, mode: str, verify: int = 3) -> dict:
    store = StateStore(state)
    plan_cfg = settings()["planning"]
    limit = float(settings()["access"]["target_min"])
    strictest = plan_cfg["scenarios"][0]
    problems = scenarios.nested_check()
    if problems:
        raise ValueError("planner scenarios must be nested: " + "; ".join(problems))
    demand = pd.read_parquet(store.demand(mode))
    cands = pd.read_parquet(store.candidates(mode))
    costs, ftransform, _, _ = inputs.read_raster(store.friction(mode))
    costs = np.ascontiguousarray(costs, dtype=np.float32)
    cell = (abs(ftransform[4]), ftransform[0])
    rep = demand["cell"].to_numpy(np.int64)
    field_path = store.tt30(strictest, mode)
    floor = offset = None
    if field_path.exists():
        field, _, _ = read_minutes(field_path)
        field = np.where(np.isfinite(field), field, np.inf).astype(np.float32)
        at_rep = field.ravel()[rep] if rep.size else np.zeros(0)
        finite = at_rep[np.isfinite(at_rep)]
        floor = float(finite.min()) if finite.size else limit
        offset = floor - limit - 1.0     # the field holds whole minutes: allow for rounding
        search = dijkstra.ReachSearch(costs, cell, prune_field=field, prune_offset=offset)
        del field
    else:
        log(f"  {mode}: no 30 m baseline ({field_path.name}), so searches run unpruned within the cap")
        search = dijkstra.ReachSearch(costs, cell)
        verify = 0

    indptr, indices, minutes, explored = [0], [], [], []
    t0 = time.time()
    for j, src in enumerate(cands["cell"].to_numpy(np.int64)):
        if rep.size == 0:                 # nobody beyond the target: nothing to reach
            indices.append(np.zeros(0, np.int32))
            minutes.append(np.zeros(0, np.float32))
            indptr.append(indptr[-1])
            explored.append(0)
            continue
        d = search.run([src], limit)
        v = d[rep]
        hit = np.flatnonzero(v <= limit)
        indices.append(hit.astype(np.int32))
        minutes.append(v[hit].astype(np.float32))
        indptr.append(indptr[-1] + hit.size)
        explored.append(search.last_expanded)
        if (j + 1) % 50 == 0:
            log(f"  {mode} reach: {j + 1}/{len(cands)} candidates, {time.time() - t0:.0f} s")
    secs = time.time() - t0

    checks = []
    if verify and len(cands):
        plain = dijkstra.ReachSearch(costs, cell)
        for j in np.linspace(0, len(cands) - 1, min(verify, len(cands))).astype(int):
            d_plain = plain.run([int(cands["cell"].iloc[j])], limit)[rep].copy()
            got = set(indices[j].tolist())
            want = set(np.flatnonzero(d_plain <= limit).tolist())
            checks.append({"cand_id": int(j), "pruned_hits": len(got), "unpruned_hits": len(want),
                           "identical": got == want, "unpruned_cells_explored": plain.last_expanded,
                           "pruned_cells_explored": explored[j]})
        if not all(c["identical"] for c in checks):
            raise RuntimeError(f"pruned and unpruned reach disagree: {checks}")

    np.savez_compressed(store.reach(mode), indptr=np.asarray(indptr, np.int64),
                        indices=np.concatenate(indices) if indices else np.zeros(0, np.int32),
                        minutes=np.concatenate(minutes) if minutes else np.zeros(0, np.float32),
                        n_demand=len(demand), n_cand=len(cands), limit=limit)
    meta = {"state": state, "mode": mode, "limit_min": limit, "candidates": len(cands), "demand_points": len(demand),
            "seconds": round(secs, 1), "median_cells_explored": float(np.median(explored)) if explored else 0,
            "max_cells_explored": int(max(explored)) if explored else 0, "grid_cells": int(costs.size),
            "pruned": offset is not None, "prune_floor_min": floor, "prune_offset_min": offset,
            "pairs_within_limit": int(indptr[-1]), "verification": checks, "computed": time.strftime("%Y-%m-%d %H:%M")}
    write_json(store.reach_meta(mode), meta)
    log(f"  {mode} reach: {len(cands)} candidates in {secs:.0f} s; median {meta['median_cells_explored']:,.0f} cells "
        f"explored of {costs.size:,}; {indptr[-1]:,} candidate-demand pairs within {int(limit)} min"
        + (f"; pruning verified on {len(checks)} candidates" if checks else ""))
    return meta


def run(state: str, verify: int = 3, modes=None) -> dict:
    out = {}
    for mode in modes or settings()["planning"]["modes"]:
        log(f"planning inputs, {mode}")
        demand = build_demand(state, mode)
        build_candidates(state, demand, mode)
        out[mode] = build_reach(state, mode, verify=verify)
    return out
