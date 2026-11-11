"""Step 3: how many women aged 15-49 live within 30, 60 and 120 minutes of care.

Counts are made per population cell and summed to wards, LGAs and the state. Nothing is
reported below ward level, as WorldPop advises, because single-cell values are fractional
estimates.
"""
from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pandas as pd

from . import display, inputs, scenarios
from .access import read_minutes
from .config import norm_name, settings
from .prepare import log
from .store import StateStore, read_json, write_json


def ward_table(women: np.ndarray, minutes: np.ndarray, ward_ids: np.ndarray, wards: pd.DataFrame,
               thresholds=(30, 60, 120), target: int = 120) -> pd.DataFrame:
    ids = ward_ids.ravel()
    keep = ids > 0
    ids = ids[keep].astype(np.int64)
    w = women.ravel()[keep].astype(np.float64)
    t = minutes.ravel()[keep]
    n = int(max(ids.max(initial=0), int(wards["ward_id"].max()))) + 1
    idx = wards["ward_id"].to_numpy(np.int64)
    out = wards[["ward_id", "ward", "lga"]].copy()
    out["women"] = np.bincount(ids, weights=w, minlength=n)[idx]
    for th in thresholds:
        out[f"within_{th}"] = np.bincount(ids, weights=w * (t <= th), minlength=n)[idx]
    reach = np.isfinite(t)
    out["no_route"] = np.bincount(ids, weights=w * ~reach, minlength=n)[idx]
    out["beyond_target"] = out["women"] - out[f"within_{target}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["share_beyond_target"] = np.where(out["women"] > 0, out["beyond_target"] / out["women"], np.nan)
        wr = np.bincount(ids, weights=w * reach, minlength=n)[idx]
        tw = np.bincount(ids, weights=np.where(reach, w * np.nan_to_num(t), 0.0), minlength=n)[idx]
        out["mean_minutes"] = np.where(wr > 0, tw / wr, np.nan)
    return out


from .tables import lga_table, totals  # noqa: E402  (kept here for callers of coverage.*)


def _worldpop_state_total(state: str):
    path = inputs.shared_path("worldpop/total/NGA_population_v3_0_admin.zip")
    try:
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith("states_pop_total_scaled.csv"))
            df = pd.read_csv(io.BytesIO(zf.read(name)))
        hit = df[df["statename"].map(norm_name) == norm_name(state)]
        return float(hit["population"].iloc[0]) if len(hit) else None
    except (OSError, StopIteration, KeyError):
        return None


def run(state: str) -> dict:
    store = StateStore(state)
    win = read_json(store.window)
    acc = settings()["access"]
    thresholds = tuple(acc["thresholds_min"])
    target = int(acc["target_min"])
    women, ptransform, pcrs, _ = inputs.read_raster(store.women)
    total_pop, *_ = inputs.read_raster(store.population)
    ward_ids, *_ = inputs.read_raster(store.ward_ids)
    wards = pd.read_parquet(store.wards)
    inside = ward_ids > 0
    fac = pd.read_parquet(store.facilities)

    runs, ward_props, display_runs = {}, {}, {}
    for s in scenarios.names():
        for mode in acc["modes"]:
            path = store.ttpop(s, mode)
            if not path.exists():
                continue
            minutes, _, _ = read_minutes(path)
            table = ward_table(women, minutes, ward_ids, wards, thresholds, target)
            table.to_parquet(store.coverage(s, mode), index=False)
            meta = read_json(store.run_meta(s, mode))
            key = f"{s}/{mode}"
            runs[key] = {"scenario": s, "mode": mode, "label": scenarios.label(s), "sources": meta["sources"],
                         **totals(table, thresholds)}
            ward_props[key] = table.set_index("ward_id")[["share_beyond_target", "beyond_target", "women"]]
            png = store.display_png(s, mode)
            bounds = display.write_png(display.classify(minutes, inside), ptransform, pcrs, png)
            display_runs[key] = {"png": png.name, "bounds": bounds}
            # Travel time at each facility's own location (used by the assistant's lookup).
            r = np.floor((fac["latitude"].to_numpy() - ptransform[5]) / ptransform[4]).astype(int)
            c = np.floor((fac["longitude"].to_numpy() - ptransform[2]) / ptransform[0]).astype(int)
            ok = (r >= 0) & (r < minutes.shape[0]) & (c >= 0) & (c < minutes.shape[1])
            vals = np.full(len(fac), np.nan)
            vals[ok] = minutes[r[ok], c[ok]]
            fac[f"tt_{s}_{mode}"] = vals
            log(f"  {key:18} {runs[key]['share_beyond_target']:.1%} of women more than {target} minutes away")
    if not runs:
        raise FileNotFoundError("no travel-time runs found: run the travel-time step first")
    fac.to_parquet(store.facilities, index=False)

    # Ward outlines with every run's share attached, for the app and the brief.
    gj = json.loads(store.wards_geojson.read_text(encoding="utf-8"))
    for feat in gj["features"]:
        wid = int(feat["properties"]["ward_id"])
        for key, props in ward_props.items():
            tag = key.replace("/", "_")
            if wid in props.index:
                row = props.loc[wid]
                feat["properties"][f"share_{tag}"] = None if pd.isna(row["share_beyond_target"]) else round(float(row["share_beyond_target"]), 4)
                feat["properties"][f"beyond_{tag}"] = round(float(row["beyond_target"]), 1)
                feat["properties"]["women"] = round(float(row["women"]), 1)
    store.display_wards.write_text(json.dumps(gj), encoding="utf-8")
    write_json(store.display_meta, {"runs": display_runs,
                                    "legend": [[b[0], b[2]] for b in display.BANDS] + [list(display.NO_ROUTE)]})

    default = f"{scenarios.default_scenario()}/{settings()['planning']['mode']}"
    shares = [r["share_beyond_target"] for r in runs.values() if r["share_beyond_target"] is not None]
    lgas = lga_table(pd.read_parquet(store.coverage(*default.split("/")))) if default in runs else None
    worldpop_total = _worldpop_state_total(win["state"])
    in_state_total = float(total_pop[inside].sum())
    summary = {
        "state": win["state"], "statecode": win.get("statecode"), "computed": time.strftime("%Y-%m-%d %H:%M"),
        "target_min": target, "thresholds": list(thresholds), "default_run": default,
        "women_total": float(women[inside].sum()), "runs": runs,
        "range_share_beyond_target": [min(shares), max(shares)] if shares else None,
        "worst_lgas": lgas.head(10).to_dict(orient="records") if lgas is not None else [],
        "checks": {
            "women_in_wards_from_tables": float(next(iter(runs.values()))["women"]),
            "women_in_wards_from_raster": float(women[inside].sum()),
            "population_in_wards": in_state_total,
            "worldpop_state_population": worldpop_total,
            "population_ratio_to_worldpop": in_state_total / worldpop_total if worldpop_total else None,
        },
    }
    write_json(store.summary, summary)
    return summary
