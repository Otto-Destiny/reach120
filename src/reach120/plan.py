"""Step 5: upgrade plans within a budget, and the budget curve.

Used by the command line, the app's planner page and the assistant. It reads only small
files (parquet, npz, json), so it runs without the geospatial stack.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from . import optimize, scenarios
from .config import load, settings, slug
from .store import StateStore, read_json, write_json


@dataclass
class PlanInputs:
    state: str
    mode: str
    demand: pd.DataFrame
    candidates: pd.DataFrame
    cover: csr_matrix
    minutes: csr_matrix
    summary: dict


def load_inputs(state: str, mode: str | None = None) -> PlanInputs:
    store = StateStore(state)
    mode = mode or settings()["planning"]["mode"]
    demand = pd.read_parquet(store.demand(mode))
    cands = pd.read_parquet(store.candidates(mode))
    with np.load(store.reach(mode)) as z:
        indptr, indices, mins = z["indptr"], z["indices"], z["minutes"]
        shape = (int(z["n_cand"]), int(z["n_demand"]))
    cover = csr_matrix((np.ones(indices.size, dtype=bool), indices, indptr), shape=shape)
    minutes = csr_matrix((mins.astype(np.float32), indices, indptr), shape=shape)
    return PlanInputs(state, mode, demand, cands, cover, minutes, read_json(store.summary))


def cost_settings() -> dict:
    return load("costs")


def default_tier_costs() -> dict[str, float]:
    return {k: float(v["cost_naira"]) for k, v in cost_settings()["tiers"].items()}


def candidate_costs(cands: pd.DataFrame, tier_costs: dict | None = None, overrides: dict | None = None) -> np.ndarray:
    tiers = {**default_tier_costs(), **(tier_costs or {})}
    per_site = {**(cost_settings().get("overrides") or {}), **(overrides or {})}
    costs = cands["tier"].map(tiers).astype(float).to_numpy()
    if per_site:
        ids = cands["unique_id"].astype(str).to_numpy()
        for i, uid in enumerate(ids):
            if uid in per_site:
                costs[i] = float(per_site[uid])
    return costs


def problem_for(inp: PlanInputs, scenario: str, tier_costs=None, overrides=None) -> optimize.Problem:
    if scenario not in settings()["planning"]["scenarios"]:
        raise ValueError(f"the planner runs the nested scenarios {settings()['planning']['scenarios']}, not {scenario!r}")
    return optimize.Problem(
        cover=inp.cover,
        weights=inp.demand[f"w_{scenario}"].to_numpy(np.float64) if len(inp.demand) else np.zeros(0),
        costs=candidate_costs(inp.candidates, tier_costs, overrides),
        allowed=~inp.candidates[f"cap_{scenario}"].to_numpy(bool),
    )


def _baseline(inp: PlanInputs, scenario: str) -> dict:
    run = inp.summary["runs"].get(f"{scenario}/{inp.mode}", {})
    return {"women": run.get("women"), "beyond_target": run.get("beyond_target"),
            "share_beyond_target": run.get("share_beyond_target")}


def _site_rows(inp: PlanInputs, problem: optimize.Problem, selected, credit: dict | None = None) -> list[dict]:
    rows = []
    for j in selected:
        c = inp.candidates.iloc[j]
        alone = optimize.covered_women(problem, [j])
        rows.append({"cand_id": int(c["cand_id"]), "unique_id": str(c["unique_id"]), "facility_name": c["facility_name"],
                     "facility_type": c["facility_type"], "ownership": c["facility_ownership"], "lga": c["lga"],
                     "ward": c["ward"], "tier": c["tier"], "cost_naira": float(problem.costs[j]),
                     "women_newly_within_target": round((credit or {}).get(j, alone), 1),
                     "women_reachable_alone": round(alone, 1),
                     "latitude": float(c["latitude"]), "longitude": float(c["longitude"])})
    rows.sort(key=lambda r: -r["women_newly_within_target"])
    return rows


def run_plan(inp: PlanInputs, scenario: str, budget: float, tier_costs=None, overrides=None,
             time_limit: float | None = None, save: bool = True) -> dict:
    plan_cfg = settings()["planning"]
    problem = problem_for(inp, scenario, tier_costs, overrides)
    greedy = optimize.solve_greedy(problem, budget)
    exact = optimize.solve_exact(problem, budget, time_limit or plan_cfg["solver_time_limit_s"],
                                 plan_cfg["solver_workers"], hint=greedy.selected)
    straight = optimize.solve_ranked(problem, budget, inp.candidates["straight_score"].to_numpy(float))
    best = exact if exact.women >= greedy.women else greedy        # exact can time out early
    credit = optimize.attribute(problem, inp.minutes, best.selected)
    base = _baseline(inp, scenario)
    beyond = base["beyond_target"] or 0.0
    reached = best.women
    tiers = {**default_tier_costs(), **(tier_costs or {})}
    fingerprint = json.dumps({"tiers": tiers, "overrides": overrides or {}}, sort_keys=True)
    run_id = f"{slug(scenario)}-{inp.mode}-{int(round(budget / 1e6))}m-{hashlib.sha1(fingerprint.encode()).hexdigest()[:6]}"
    result = {
        "run_id": run_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "state": inp.state,
        "scenario": scenario, "scenario_label": scenarios.label(scenario), "mode": inp.mode,
        "target_min": inp.summary.get("target_min", 120), "budget_naira": float(budget),
        "costs_are_placeholders": bool(cost_settings().get("placeholder", True)),
        "tier_costs_naira": tiers, "cost_overrides": overrides or {},
        "baseline": base,
        "plan": {"method": best.method, "status": exact.status, "women_newly_within_target": round(reached, 1),
                 "cost_naira": best.cost, "sites": _site_rows(inp, problem, best.selected, credit),
                 "solver_bound": exact.bound, "solver_seconds": round(exact.seconds, 2)},
        "greedy": {"women_newly_within_target": round(greedy.women, 1), "cost_naira": greedy.cost,
                   "ranked_sites": [{"facility_name": inp.candidates.iloc[j]["facility_name"], "lga": inp.candidates.iloc[j]["lga"],
                                     "cost_naira": float(problem.costs[j]), "women_added": round(g, 1)} for j, g in greedy.order]},
        "straight_line": {"women_newly_within_target": round(straight.women, 1), "cost_naira": straight.cost,
                          "sites": [inp.candidates.iloc[j]["facility_name"] for j in straight.selected]},
        "after": {"beyond_target": round(beyond - reached, 1),
                  "share_beyond_target": (beyond - reached) / base["women"] if base["women"] else None},
        "reachable_by_any_candidate": round(optimize.covered_women(problem, list(np.flatnonzero(problem.allowed))), 1),
    }
    if save:
        write_json(StateStore(inp.state).plan(run_id), result)
    return result


def budget_points(problem: optimize.Problem, points: int, extra=()) -> list[float]:
    """From zero to the budget that reaches everyone any candidate can reach."""
    everything = optimize.solve_greedy(problem, float(problem.costs[problem.allowed].sum()) if problem.allowed.any() else 0.0)
    top = everything.cost
    if top <= 0:
        return sorted({0.0, *map(float, extra)})
    grid = np.linspace(0, top, points)
    return sorted({float(round(b, -6)) for b in grid} | {float(b) for b in extra if 0 <= b})


def sweep(inp: PlanInputs, scenario: str, tier_costs=None, overrides=None, points: int | None = None) -> pd.DataFrame:
    plan_cfg = settings()["planning"]
    problem = problem_for(inp, scenario, tier_costs, overrides)
    rows = []
    for b in budget_points(problem, points or plan_cfg["sweep_points"], [plan_cfg["default_budget_naira"]]):
        greedy = optimize.solve_greedy(problem, b)
        exact = optimize.solve_exact(problem, b, min(5.0, plan_cfg["solver_time_limit_s"]), plan_cfg["solver_workers"],
                                     hint=greedy.selected)
        straight = optimize.solve_ranked(problem, b, inp.candidates["straight_score"].to_numpy(float))
        rows.append({"budget_naira": b, "exact": max(exact.women, greedy.women), "exact_status": exact.status,
                     "greedy": greedy.women, "straight_line": straight.women, "sites_exact": len(exact.selected)})
    return pd.DataFrame(rows)


def headline_mode(summary: dict, scenario: str) -> str:
    """The transport mode the brief plans for (see settings planning.brief_mode)."""
    plan_cfg = settings()["planning"]
    choice = plan_cfg.get("brief_mode", "auto")
    if choice != "auto":
        return choice
    runs = summary["runs"]
    modes = [m for m in plan_cfg["modes"] if f"{scenario}/{m}" in runs]
    return max(modes, key=lambda m: runs[f"{scenario}/{m}"]["beyond_target"]) if modes else plan_cfg["mode"]


def default_run(state: str, scenario: str, mode: str) -> dict | None:
    store = StateStore(state)
    if not store.default_runs.exists():
        return None
    rid = read_json(store.default_runs).get(mode, {}).get(scenario)
    return read_json(store.plan(rid)) if rid and store.plan(rid).exists() else None


def run(state: str, modes=None) -> dict:
    from .prepare import log

    store = StateStore(state)
    plan_cfg = settings()["planning"]
    out = read_json(store.default_runs) if store.default_runs.exists() else {}
    for mode in modes or plan_cfg["modes"]:
        inp = load_inputs(state, mode)
        out[mode] = {}
        for s in plan_cfg["scenarios"]:
            curve = sweep(inp, s)
            curve.to_parquet(store.curve(s, mode), index=False)
            res = run_plan(inp, s, float(plan_cfg["default_budget_naira"]))
            out[mode][s] = res["run_id"]
            log(f"  {mode:5} {s:6} {plan_cfg['default_budget_naira'] / 1e9:.1f}bn: {len(res['plan']['sites'])} sites, "
                f"{res['plan']['women_newly_within_target']:,.0f} more women within target "
                f"(greedy {res['greedy']['women_newly_within_target']:,.0f}, "
                f"straight-line {res['straight_line']['women_newly_within_target']:,.0f})")
        write_json(store.default_runs, out)
    return out
