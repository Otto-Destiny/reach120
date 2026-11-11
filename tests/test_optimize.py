"""The upgrade optimiser: hand-solvable cases, brute force, and the comparisons."""
import itertools

import numpy as np
from scipy.sparse import csr_matrix

from reach120 import optimize


def problem(cover_rows, weights, costs, allowed=None):
    n_dem = len(weights)
    rows, cols = [], []
    for j, covered in enumerate(cover_rows):
        rows += [j] * len(covered)
        cols += list(covered)
    cover = csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(cover_rows), n_dem))
    cover.sort_indices()
    return optimize.Problem(cover, np.asarray(weights, float), np.asarray(costs, float),
                            np.ones(len(cover_rows), bool) if allowed is None else np.asarray(allowed, bool))


def brute_force(p, budget):
    best = 0.0
    n = p.cover.shape[0]
    for k in range(n + 1):
        for combo in itertools.combinations(range(n), k):
            if p.allowed[list(combo)].all() and p.costs[list(combo)].sum() <= budget:
                best = max(best, optimize.covered_women(p, combo))
    return best


def test_hand_solved_case():
    # Two cheap overlapping sites beat one expensive big one at this budget; greedy's first pick misleads it.
    p = problem([[0, 1, 2], [3, 4], [5, 6]], [10, 10, 10, 12, 12, 11, 11], [300, 150, 150])
    exact = optimize.solve_exact(p, 300)
    assert sorted(exact.selected) == [1, 2]
    assert exact.women == 46
    assert exact.status == "OPTIMAL"


def test_exact_matches_brute_force_and_beats_or_ties_greedy():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n_cand, n_dem = 7, 25
        cover_rows = [sorted(rng.choice(n_dem, size=rng.integers(1, 8), replace=False).tolist()) for _ in range(n_cand)]
        p = problem(cover_rows, rng.uniform(1, 50, n_dem).round(1), rng.choice([150, 300, 600], n_cand),
                    rng.random(n_cand) > 0.15)
        budget = float(rng.choice([300, 600, 900, 1200]))
        exact = optimize.solve_exact(p, budget)
        greedy = optimize.solve_greedy(p, budget)
        assert abs(exact.women - brute_force(p, budget)) < 0.11      # objective works in tenths of a woman
        assert exact.women >= greedy.women - 0.11
        assert exact.cost <= budget and greedy.cost <= budget
        assert p.allowed[exact.selected].all()


def test_more_budget_never_reaches_fewer_women():
    rng = np.random.default_rng(1)
    cover_rows = [sorted(rng.choice(40, size=6, replace=False).tolist()) for _ in range(10)]
    p = problem(cover_rows, rng.uniform(1, 20, 40), rng.choice([150, 300, 600], 10))
    values = [optimize.solve_exact(p, b).women for b in range(0, 3001, 150)]
    assert all(b >= a - 0.11 for a, b in zip(values, values[1:]))


def test_ranked_ignores_disallowed_and_unaffordable_sites():
    p = problem([[0], [1], [2]], [5, 5, 5], [100, 500, 100], allowed=[True, True, False])
    sol = optimize.solve_ranked(p, 200, scores=np.array([1.0, 3.0, 2.0]))
    assert sol.selected == [0]


def test_attribution_credits_the_nearest_site():
    p = problem([[0, 1], [1, 2]], [3, 4, 5], [1, 1])
    minutes = csr_matrix((np.array([10, 50, 20, 30], np.float32), np.array([0, 1, 1, 2]), np.array([0, 2, 4])), shape=(2, 3))
    credit = optimize.attribute(p, minutes, [0, 1])
    assert credit == {0: 3.0, 1: 9.0}
