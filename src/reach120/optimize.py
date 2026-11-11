"""Choosing which facilities to upgrade within a budget.

The model is a maximal covering problem with a budget: choose upgrades x_j (cost c_j) to
maximise the women w_i brought within two hours, where demand point i counts once any
chosen candidate reaches it.

    maximise    sum_i w_i y_i
    subject to  y_i <= sum_j a_ij x_j          a_ij = 1 if j reaches i within two hours
                sum_j c_j x_j <= budget
                x_j, y_i in {0, 1}

Three solvers, compared in every result:

* ``solve_exact``: OR-Tools CP-SAT. Demand points with the same set of covering candidates
  are merged first, which shrinks the model a great deal and changes nothing.
* ``solve_greedy``: best women-per-naira first, plus the classic check against the best
  single affordable site. It gives a stable ranked list.
* ``solve_ranked``: a naive comparison that ranks sites by uncovered women within 30 km in a
  straight line, ignoring roads and rivers.

Nothing here imports the geospatial stack, so the app can run it without GDAL.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Problem:
    cover: csr_matrix          # candidates x demand points, True where reached within the target
    weights: np.ndarray        # women at each demand point
    costs: np.ndarray          # naira per candidate
    allowed: np.ndarray        # candidates that may be chosen (not already capable)


@dataclass
class Solution:
    method: str
    selected: list[int]
    women: float
    cost: float
    status: str = "OPTIMAL"
    bound: float | None = None
    seconds: float = 0.0
    order: list = field(default_factory=list)   # greedy: (candidate, women added) in pick order


def covered_women(problem: Problem, selected) -> float:
    selected = list(selected)
    if not selected:
        return 0.0
    cols = np.unique(problem.cover[selected].indices)
    return float(problem.weights[cols].sum())


def _affordable(problem: Problem, budget: float) -> np.ndarray:
    return np.flatnonzero(problem.allowed & (problem.costs <= budget))


def solve_greedy(problem: Problem, budget: float) -> Solution:
    t0 = time.time()
    cover, w, costs = problem.cover, problem.weights.astype(np.float64), problem.costs.astype(np.float64)
    covered = np.zeros(w.size, dtype=bool)
    alive = [int(j) for j in _affordable(problem, budget)]
    remaining, chosen, order = float(budget), [], []
    while alive:
        best, best_ratio, best_gain = None, 0.0, 0.0
        for j in alive:
            if costs[j] > remaining:
                continue
            cols = cover.indices[cover.indptr[j]:cover.indptr[j + 1]]
            gain = float(w[cols][~covered[cols]].sum())
            if gain <= 0:
                continue
            ratio = gain / max(costs[j], 1.0)
            if ratio > best_ratio:
                best, best_ratio, best_gain = j, ratio, gain
        if best is None:
            break
        cols = cover.indices[cover.indptr[best]:cover.indptr[best + 1]]
        covered[cols] = True
        remaining -= costs[best]
        chosen.append(best)
        order.append((best, best_gain))
        alive.remove(best)
    total = float(w[covered].sum())
    # The best single affordable site can beat greedy when costs differ a lot.
    singles = _affordable(problem, budget)
    if singles.size:
        gains = np.array([w[cover.indices[cover.indptr[j]:cover.indptr[j + 1]]].sum() for j in singles])
        k = int(np.argmax(gains))
        if gains[k] > total:
            j = int(singles[k])
            chosen, order, total = [j], [(j, float(gains[k]))], float(gains[k])
    return Solution("greedy", chosen, total, float(costs[chosen].sum()) if chosen else 0.0,
                    seconds=time.time() - t0, order=order)


def solve_ranked(problem: Problem, budget: float, scores: np.ndarray) -> Solution:
    """Take sites in order of `scores` while the budget lasts."""
    t0 = time.time()
    remaining, chosen = float(budget), []
    for j in np.argsort(-np.asarray(scores, float), kind="stable"):
        if problem.allowed[j] and problem.costs[j] <= remaining and scores[j] > 0:
            chosen.append(int(j))
            remaining -= float(problem.costs[j])
    return Solution("straight_line", chosen, covered_women(problem, chosen),
                    float(problem.costs[chosen].sum()) if chosen else 0.0, seconds=time.time() - t0)


def solve_exact(problem: Problem, budget: float, time_limit: float = 20.0, workers: int = 8,
                hint: list[int] | None = None) -> Solution:
    from ortools.sat.python import cp_model

    t0 = time.time()
    cands = _affordable(problem, budget)
    if cands.size == 0:
        return Solution("exact", [], 0.0, 0.0, status="NO_AFFORDABLE_CANDIDATE", bound=0.0)
    sub = problem.cover[cands].tocsc()
    sub.sort_indices()
    nnz = np.diff(sub.indptr)
    dem = np.flatnonzero((nnz > 0) & (problem.weights > 0))
    if dem.size == 0:
        return Solution("exact", [], 0.0, 0.0, status="NOTHING_TO_REACH", bound=0.0)
    groups: dict[bytes, float] = {}
    for i in dem:
        key = sub.indices[sub.indptr[i]:sub.indptr[i + 1]].tobytes()
        groups[key] = groups.get(key, 0.0) + float(problem.weights[i])

    # Whole naira: costs rounded up and the budget down, so a plan can never exceed the budget.
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{k}") for k in range(cands.size)]
    cost_units = [int(math.ceil(problem.costs[j])) for j in cands]
    model.Add(sum(c * v for c, v in zip(cost_units, x)) <= int(math.floor(budget)))
    terms = []
    for key, weight in groups.items():
        members = np.frombuffer(key, dtype=sub.indices.dtype)
        y = model.NewBoolVar("")
        model.AddBoolOr([x[int(m)] for m in members]).OnlyEnforceIf(y)
        terms.append((int(round(weight * 10)), y))      # tenths of a woman
    model.Maximize(sum(c * y for c, y in terms))
    if hint:
        chosen = set(hint)
        for k, j in enumerate(cands):
            model.AddHint(x[k], 1 if int(j) in chosen else 0)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    status = solver.Solve(model)
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Solution("exact", [], 0.0, 0.0, status=name, seconds=time.time() - t0)
    selected = [int(cands[k]) for k in range(cands.size) if solver.Value(x[k])]
    return Solution("exact", selected, covered_women(problem, selected),
                    float(problem.costs[selected].sum()) if selected else 0.0, status=name,
                    bound=solver.BestObjectiveBound() / 10.0, seconds=time.time() - t0)


def attribute(problem: Problem, minutes: csr_matrix, selected) -> dict[int, float]:
    """Women newly within reach, credited to whichever chosen site is nearest to them."""
    selected = list(selected)
    if not selected:
        return {}
    n = problem.weights.size
    best = np.full(n, np.inf)
    owner = np.full(n, -1, dtype=np.int64)
    for j in selected:
        cols = minutes.indices[minutes.indptr[j]:minutes.indptr[j + 1]]
        vals = minutes.data[minutes.indptr[j]:minutes.indptr[j + 1]]
        better = vals < best[cols]
        best[cols[better]] = vals[better]
        owner[cols[better]] = j
    got = owner >= 0
    sums = np.bincount(owner[got], weights=problem.weights[got], minlength=max(selected) + 1)
    return {int(j): float(sums[j]) for j in selected}
