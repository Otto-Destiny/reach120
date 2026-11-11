"""Step 6: evidence that the numbers can be trusted, and where they cannot.

* ``ontime_travel_times``: in any state holding one of OnTIME's 15 city regions, modelled time
  to the nearest verified facility against Google's driving times for the same cells.
* ``osrm_check``: optional, opt-in. Rural spot-check against road routing on OpenStreetMap,
  through the public OSRM demo server (one request a second, a few dozen in total).
* ``dhs_check``: if Nigeria DHS 2023-24 files are present, home-birth share by modelled
  travel-time band.
* ``sensitivity``: every headline across capability scenarios and transport modes.
* ``write_report``: validation_report.md, stating plainly what was and was not run.
"""
from __future__ import annotations

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dijkstra, display, geo, inputs, scenarios
from .access import read_minutes
from .config import project_path, settings
from .prepare import log, snap_cells
from .store import StateStore, national, read_json, write_json


def _window_lonlat(win) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = win["bounds_utm"]
    lon, lat = geo.to_lonlat([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def _metrics(model: np.ndarray, ref: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    ok = np.isfinite(model) & np.isfinite(ref)
    m, r = model[ok], ref[ok]
    if m.size < 5:
        return {"cells": int(m.size)}
    err = m - r
    return {"cells": int(m.size), "spearman": float(spearmanr(m, r).statistic), "median_model_min": float(np.median(m)),
            "median_reference_min": float(np.median(r)), "median_error_min": float(np.median(err)),
            "median_abs_error_min": float(np.median(np.abs(err))), "mean_error_min": float(err.mean()),
            "same_side_of_30": float(np.mean((m <= 30) == (r <= 30))), "same_side_of_60": float(np.mean((m <= 60) == (r <= 60)))}


def ontime_travel_times(state: str) -> dict | None:
    from .scorecard import load_ontime

    store = StateStore(state)
    win = read_json(store.window)
    w, s, e, n = _window_lonlat(win)
    cells_path = store.validation("ontime_cells.parquet")
    if cells_path.exists():
        cells = pd.read_parquet(cells_path)
    else:
        parts = []
        cols = ["center_lat_lng", "facility_type", "departure_time", "min_travel_time_1_minutes"]
        for chunk in pd.read_csv(project_path(settings()["paths"]["ontime_travel_times"]), usecols=cols, chunksize=500_000):
            chunk = chunk[chunk["facility_type"] == "both"]
            ll = chunk["center_lat_lng"].str.split(expand=True).astype(float)
            chunk = chunk.assign(lat=ll[0], lon=ll[1])
            parts.append(chunk[(chunk["lon"].between(w, e)) & (chunk["lat"].between(s, n))])
        cells = pd.concat(parts)[["lat", "lon", "departure_time", "min_travel_time_1_minutes"]].rename(
            columns={"min_travel_time_1_minutes": "google_min"})
        cells.to_parquet(cells_path, index=False)
    if cells.empty:
        return None
    if "model_min" in cells.columns and cells_path.stat().st_mtime > store.friction("mixed").stat().st_mtime:
        return _ontime_summary(store, cells, int(cells.attrs.get("sources", 0)) or None, win["state"])

    transform, shape = win["transform"], win["shape"]
    costs, *_ = inputs.read_raster(store.friction("mixed"))
    costs = np.ascontiguousarray(costs, dtype=np.float32)
    ont = load_ontime()
    ox, oy = geo.to_utm(ont["longitude"].to_numpy(), ont["latitude"].to_numpy())
    r, c = geo.rowcol(transform, ox, oy)
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    r, c, ok, _ = snap_cells(costs, r[inside], c[inside], int(settings()["access"]["snap_radius_cells"]))
    sources = (r[ok] * shape[1] + c[ok]).astype(np.int64)
    log(f"  OnTIME: {len(cells[['lat', 'lon']].drop_duplicates()):,} cells and {sources.size} verified facilities in the window")
    dist = dijkstra.travel_time(costs, sources, win["cell_size"])
    cx, cy = geo.to_utm(cells["lon"].to_numpy(), cells["lat"].to_numpy())
    cr, cc = geo.rowcol(transform, cx, cy)
    good = (cr >= 0) & (cr < shape[0]) & (cc >= 0) & (cc < shape[1])
    model = np.full(len(cells), np.nan)
    model[good] = dist[cr[good], cc[good]]
    model[~np.isfinite(model)] = np.nan
    cells = cells.assign(model_min=model)
    cells.to_parquet(cells_path, index=False)
    write_json(store.validation("ontime_sources.json"), {"verified_facilities_in_window": int(sources.size)})
    return _ontime_summary(store, cells, int(sources.size), win["state"])


def _ontime_summary(store: StateStore, cells: pd.DataFrame, sources: int | None, state: str) -> dict:
    import json

    import shapely
    from shapely.geometry import shape

    if sources is None:
        src = store.validation("ontime_sources.json")
        sources = read_json(src)["verified_facilities_in_window"] if src.exists() else 0
    # The OnTIME file repeats some cells (identical rows, up to 7 times): count each cell once.
    cells = cells.drop_duplicates(["lat", "lon", "departure_time"])
    # Compare only cells inside the state: near the window's edge, a cell's nearest facility can lie outside it.
    poly = shape(json.loads(store.state_geojson.read_text(encoding="utf-8"))["features"][0]["geometry"])
    inside = shapely.contains_xy(poly, cells["lon"].to_numpy(), cells["lat"].to_numpy())
    cells = cells[inside]
    by_dep = {dep: _metrics(g["model_min"].to_numpy(), g["google_min"].to_numpy()) for dep, g in cells.groupby("departure_time")}
    result = {"verified_facilities_in_window": sources, "cells": int(len(cells[["lat", "lon"]].drop_duplicates())), "by_departure": by_dep,
              "note": "Google times are by car with traffic; the model is a no-traffic, dry-season friction surface. "
                      "Cells outside the state are left out."}
    write_json(store.validation("ontime_travel_times.json"), result)
    _scatter(cells[cells["departure_time"] == "weekday1-3am"], store.validation("ontime_scatter.png"), state)
    return result


def _scatter(df: pd.DataFrame, path: Path, state: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    fig.patch.set_facecolor(display.SURFACE)
    ax.set_facecolor(display.SURFACE)
    top = float(np.nanpercentile(np.r_[df["google_min"], df["model_min"]], 99))
    ax.plot([0, top], [0, top], color=display.MUTED, lw=1, zorder=1)
    ax.scatter(df["google_min"], df["model_min"], s=5, color=display.SERIES_LIGHT[0], alpha=0.35, lw=0, zorder=2)
    ax.text(top * 0.98, top * 0.92, "equal times", color=display.INK_SECONDARY, ha="right", fontsize=8)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Google driving time, weekday 1-3 am (minutes)", color=display.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Modelled travel time (minutes)", color=display.INK_SECONDARY, fontsize=9)
    ax.set_title(f"Time to the nearest verified facility, {state} city cells", color=display.INK, fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(display.AXIS)
    ax.tick_params(colors=display.MUTED, labelsize=8)
    ax.grid(color=display.GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def osrm_check(state: str, pairs: int | None = None, seed: int = 0) -> dict:
    import requests

    store = StateStore(state)
    val = settings()["validation"]
    pairs = pairs or int(val["osrm_pairs"])
    scen, mode = scenarios.default_scenario(), "mixed"
    minutes, ptransform, _ = read_minutes(store.ttpop(scen, mode))
    women, *_ = inputs.read_raster(store.women)
    ward_ids, *_ = inputs.read_raster(store.ward_ids)
    ok = (ward_ids > 0) & (women > 0.5) & (minutes >= 20) & (minutes <= 240)
    rr, cc = np.nonzero(ok)
    rng = np.random.default_rng(seed)
    pick = rng.choice(rr.size, size=min(pairs, rr.size), replace=False)
    lon, lat = geo.cell_centres(ptransform, rr[pick], cc[pick])
    fac = pd.read_parquet(store.facilities)
    fac = fac[fac[f"cap_{scen}"]]
    session = requests.Session()
    session.headers["User-Agent"] = "reach120-validation/0.1 (portfolio research tool)"
    rows = []
    for i in range(pick.size):
        d = geo.haversine_km(lon[i], lat[i], fac["longitude"].to_numpy(), fac["latitude"].to_numpy())
        near = fac.iloc[np.argsort(d)[:5]]
        coords = ";".join([f"{lon[i]:.6f},{lat[i]:.6f}"] + [f"{a:.6f},{b:.6f}" for a, b in zip(near["longitude"], near["latitude"])])
        url = f"{val['osrm_url']}/table/v1/driving/{coords}?sources=0&annotations=duration"
        try:
            resp = session.get(url, timeout=30)
            durations = resp.json().get("durations", [[None]])[0][1:]
            secs = [x for x in durations if x is not None]
            osrm_min = min(secs) / 60 if secs else np.nan
        except (requests.RequestException, ValueError, KeyError, IndexError):
            osrm_min = np.nan
        rows.append({"lon": lon[i], "lat": lat[i], "model_min": float(minutes[rr[pick[i]], cc[pick[i]]]), "osrm_min": osrm_min})
        time.sleep(float(val["osrm_pause_s"]))
    df = pd.DataFrame(rows)
    df.to_csv(store.validation("osrm_pairs.csv"), index=False)
    result = {"pairs": int(len(df)), "scenario": scen, **_metrics(df["model_min"].to_numpy(), df["osrm_min"].to_numpy()),
              "note": "OSRM times are free-flow car times from the nearest road; they leave out the walk to the road."}
    write_json(store.validation("osrm_check.json"), result)
    return result


def dhs_check(state: str) -> dict | None:
    folder = project_path(settings()["paths"]["dhs"])
    gps = sorted(glob.glob(str(folder / "**" / "*GE*FL*.shp"), recursive=True))
    births = sorted(glob.glob(str(folder / "**" / "*BR*FL*.DTA"), recursive=True))
    if not gps or not births:
        return None
    import geopandas as gpd

    store = StateStore(state)
    clusters = gpd.read_file(gps[0])
    clusters = clusters[(clusters["LATNUM"] != 0) | (clusters["LONGNUM"] != 0)]
    br = pd.read_stata(births[0], columns=None, convert_categoricals=False)
    br.columns = [c.lower() for c in br.columns]
    recent = br["b19"] < 60 if "b19" in br.columns else (br["v008"] - br["b3"]) < 60
    br = br[recent & br["m15"].notna() & ~br["m15"].isin([96, 99])]
    br = br.assign(home=br["m15"].between(10, 19))
    per = br.groupby("v001").agg(births=("home", "size"), home=("home", "mean")).reset_index()
    minutes, ptransform, _ = read_minutes(store.ttpop(scenarios.default_scenario(), "mixed"))
    r = np.floor((clusters["LATNUM"].to_numpy() - ptransform[5]) / ptransform[4]).astype(int)
    c = np.floor((clusters["LONGNUM"].to_numpy() - ptransform[2]) / ptransform[0]).astype(int)
    inside = (r >= 0) & (r < minutes.shape[0]) & (c >= 0) & (c < minutes.shape[1])
    cl = pd.DataFrame({"v001": clusters["DHSCLUST"].to_numpy()[inside], "minutes": minutes[r[inside], c[inside]]})
    df = cl.merge(per, on="v001")
    if df.empty:
        return {"clusters": 0, "note": "no DHS clusters fall inside this state's window"}
    bands = pd.cut(df["minutes"], [0, 30, 60, 120, np.inf], labels=["0-30", "30-60", "60-120", "120+"], include_lowest=True)
    table = df.groupby(bands, observed=False).apply(lambda g: pd.Series({
        "clusters": len(g), "births": int(g["births"].sum()),
        "home_birth_share": float(np.average(g["home"], weights=g["births"])) if g["births"].sum() else np.nan}))
    result = {"clusters": int(len(df)), "bands": table.reset_index().to_dict(orient="records"),
              "note": "DHS displaces cluster locations by up to 2 km (urban) and 5 km (rural), occasionally 10 km."}
    write_json(store.validation("dhs_check.json"), result)
    return result


def sensitivity(state: str) -> dict:
    store = StateStore(state)
    summary = read_json(store.summary)
    rows = [{"scenario": r["scenario"], "label": r["label"], "mode": r["mode"], "sources": r["sources"],
             "women_beyond_target": r["beyond_target"], "share_beyond_target": r["share_beyond_target"]}
            for r in summary["runs"].values()]
    budget = float(settings()["planning"]["default_budget_naira"])
    plans = []
    for mode in settings()["planning"]["modes"]:
        for s in settings()["planning"]["scenarios"]:
            path = store.curve(s, mode)
            if path.exists():
                curve = pd.read_parquet(path)
                at = curve.iloc[(curve["budget_naira"] - budget).abs().argsort().iloc[0]]
                plans.append({"scenario": s, "mode": mode, "budget_naira": float(at["budget_naira"]), "exact": float(at["exact"]),
                              "greedy": float(at["greedy"]), "straight_line": float(at["straight_line"])})
    result = {"runs": rows, "plans_at_default_budget": plans, "range_share_beyond_target": summary.get("range_share_beyond_target")}
    write_json(store.validation("sensitivity.json"), result)
    return result


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.1f}%"


def write_report(state: str, results: dict) -> Path:
    store = StateStore(state)
    summary = read_json(store.summary)
    target = summary["target_min"]
    lines = [f"# Reach120 validation report: {summary['state']}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Default run: `{summary['default_run']}`.", ""]

    lines += ["## 1. Travel times against Google (OnTIME)", ""]
    ot = results.get("ontime")
    if ot is None:
        lines += ["Not run: none of OnTIME's 15 city regions falls inside this state's window.", ""]
    else:
        lines += [f"{ot['cells']:,} city cells, {ot['verified_facilities_in_window']} verified facilities as destinations. "
                  "Error is model minus Google.", "",
                  "| Departure | Cells | Rank correlation | Median model | Median Google | Median error | Same side of 30 min | Same side of 60 min |",
                  "|---|---|---|---|---|---|---|---|"]
        for dep, m in sorted(ot["by_departure"].items()):
            if "spearman" in m:
                lines.append(f"| {dep} | {m['cells']:,} | {m['spearman']:.2f} | {m['median_model_min']:.1f} | "
                             f"{m['median_reference_min']:.1f} | {m['median_error_min']:+.1f} | {_pct(m['same_side_of_30'])} | {_pct(m['same_side_of_60'])} |")
        lines += ["", f"_{ot['note']}_", "", "![Model against Google](ontime_scatter.png)", ""]

    lines += ["## 2. Rural spot-check against road routing (OSRM)", ""]
    osrm = results.get("osrm")
    if osrm is None:
        lines += ["Not run. It calls the public OSRM demo server, so it is opt-in: `reach120 validate --state "
                  f"{summary['state']} --osrm`.", ""]
    elif "spearman" in osrm:
        lines += [f"{osrm['cells']} of {osrm['pairs']} origins answered. Rank correlation {osrm['spearman']:.2f}, "
                  f"median error {osrm['median_error_min']:+.1f} min (model minus OSRM).", "", f"_{osrm['note']}_", ""]
    else:
        lines += [f"Too few answers from the server ({osrm.get('cells', 0)}).", ""]

    lines += ["## 3. Is facility type a fair stand-in for C-section capability?", ""]
    sc_path = national("capability_scorecard.json")
    if sc_path.exists():
        sc = read_json(sc_path)
        lines += [f"OnTIME verified {sc['verified_total']:,} facilities in 15 city regions ({_pct(sc['verified_private_share'])} "
                  f"private); {sc['verified_matched']:,} were matched to GRID3 facilities.", "",
                  "| Scenario | Catch rate | Capable in study area | Not on the verified list | Balance (F1) |",
                  "|---|---|---|---|---|"]
        for r in sc["scenarios"]:
            lines.append(f"| {r['label']} | {_pct(r['catch_rate'])} | {r['capable_in_study_area']:,} | "
                         f"{_pct(r['false_inclusion_rate'])} | {r['balance_f1']:.2f} |" if r["balance_f1"] is not None else
                         f"| {r['label']} | {_pct(r['catch_rate'])} | {r['capable_in_study_area']:,} | {_pct(r['false_inclusion_rate'])} | n/a |")
        lines += ["", f"Best balance among planner scenarios: **{sc['recommended']}**. _{sc['note']}_", ""]
    else:
        lines += ["Not run yet: `reach120 scorecard`.", ""]

    lines += ["## 4. Sensitivity", "", f"Share of women aged 15-49 more than {target} minutes from care:", "",
              "| Scenario | Mode | Capable facilities used | Women beyond | Share beyond |", "|---|---|---|---|---|"]
    sens = results["sensitivity"]
    for r in sens["runs"]:
        lines.append(f"| {r['label']} | {r['mode']} | {r['sources']} | {r['women_beyond_target']:,.0f} | {_pct(r['share_beyond_target'])} |")
    if sens["plans_at_default_budget"]:
        lines += ["", "Women brought within reach at the default budget:", "",
                  "| Scenario | Mode | Budget | Exact optimum | Greedy | Straight-line ranking |", "|---|---|---|---|---|---|"]
        for p in sens["plans_at_default_budget"]:
            lines.append(f"| {p['scenario']} | {p['mode']} | {display.format_naira(p['budget_naira'])} | {p['exact']:,.0f} | "
                         f"{p['greedy']:,.0f} | {p['straight_line']:,.0f} |")
    lines += [""]

    lines += ["## 5. Does modelled travel time predict home births? (DHS)", ""]
    dhs = results.get("dhs")
    if dhs is None:
        lines += ["Not run: the Nigeria DHS 2023-24 files are not on this machine. Register with the DHS Program, request "
                  "the births recode and GPS files, and save them under `data/raw/dhs-2023-24/`.", ""]
    else:
        lines += [f"{dhs['clusters']} clusters. _{dhs['note']}_", "", "| Travel-time band | Clusters | Births | Home-birth share |", "|---|---|---|---|"]
        for b in dhs.get("bands", []):
            lines.append(f"| {b['minutes']} | {int(b['clusters'])} | {int(b['births'])} | {_pct(b['home_birth_share'])} |")
        lines += [""]

    checks = summary.get("checks", {})
    lines += ["## 6. Data checks", "",
              f"- Women in the ward tables: {checks.get('women_in_wards_from_tables', 0):,.0f}; in the raster inside wards: "
              f"{checks.get('women_in_wards_from_raster', 0):,.0f}.",
              f"- Total population inside the wards: {checks.get('population_in_wards', 0):,.0f}; WorldPop's own state total: "
              + (f"{checks['worldpop_state_population']:,.0f} (ratio {checks['population_ratio_to_worldpop']:.3f})."
                 if checks.get("worldpop_state_population") else "not found."), ""]
    for mode in settings()["planning"]["modes"]:
        reach = store.reach_meta(mode)
        if reach.exists():
            rm = read_json(reach)
            ver = rm.get("verification", [])
            check = ("pruning checked against unpruned searches on " + str(len(ver)) + " candidates: "
                     + ("identical." if all(v["identical"] for v in ver) else "MISMATCH.")) if ver else \
                    ("searches ran unpruned within the cap." if not rm.get("pruned") else "pruning not re-checked.")
            lines.append(f"- Planner reach ({mode}): {rm['candidates']} candidates, {rm['demand_points']:,} demand points, "
                         f"median {rm['median_cells_explored']:,.0f} cells explored per candidate out of {rm['grid_cells']:,}; {check}")
    win = read_json(store.window)
    if win.get("state_records_without_coordinates"):
        lines.append(f"- {win['state_records_without_coordinates']} of the state's facility records have no usable coordinates "
                     f"and are left out; capable among them: {win.get('capable_without_coordinates')}.")
    lines.append("")
    path = store.validation("validation_report.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(state: str, osrm: bool = False) -> dict:
    results = {}
    if not national("capability_scorecard.json").exists():
        from . import scorecard
        log("scoring the capability stand-in against OnTIME (national, once)")
        scorecard.run()
    log("travel times against OnTIME")
    results["ontime"] = ontime_travel_times(state)
    if osrm:
        log("rural spot-check against OSRM road routing")
        results["osrm"] = osrm_check(state)
    else:
        existing = StateStore(state).validation("osrm_check.json")
        results["osrm"] = read_json(existing) if existing.exists() else None
    results["dhs"] = dhs_check(state)
    results["sensitivity"] = sensitivity(state)
    path = write_report(state, results)
    log(f"  wrote {path}")
    return results
