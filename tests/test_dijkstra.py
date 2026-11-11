"""The least-cost search: units, barriers, caps, resets, and that pruning changes no answer."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from reach120 import dijkstra

CELL = (30.0, 30.0)


def reference(costs, sources, cell=CELL):
    """The same graph built explicitly and solved with SciPy."""
    h, w = costs.shape
    ch, cw = cell
    diag = np.hypot(ch, cw)
    steps = [(-1, 0, ch), (1, 0, ch), (0, -1, cw), (0, 1, cw), (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)]
    rows, cols, weights = [], [], []
    for r in range(h):
        for c in range(w):
            cu = costs[r, c]
            if not cu > 0:
                continue
            for dr, dc, length in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and costs[rr, cc] > 0:
                    rows.append(r * w + c)
                    cols.append(rr * w + cc)
                    weights.append(length * 0.5 * (float(cu) + float(costs[rr, cc])))
    graph = coo_matrix((weights, (rows, cols)), shape=(h * w, h * w)).tocsr()
    return scipy_dijkstra(graph, indices=np.asarray(sources), min_only=True).reshape(h, w)


def assert_same(ours, ref):
    assert np.array_equal(np.isinf(ours), np.isinf(ref))
    finite = np.isfinite(ref)
    np.testing.assert_allclose(ours[finite], ref[finite], rtol=2e-5, atol=1e-4)


def random_grid(rng, shape=(30, 40), barrier=0.15):
    costs = rng.uniform(0.001, 0.05, size=shape).astype(np.float32)
    costs[rng.random(shape) < barrier] = np.nan
    return costs


def test_matches_scipy_on_random_grids():
    rng = np.random.default_rng(0)
    for _ in range(6):
        costs = random_grid(rng)
        src = rng.choice(np.flatnonzero(costs.ravel() > 0), size=3, replace=False)
        assert_same(dijkstra.travel_time(costs, src, CELL), reference(costs, src))


def test_thirty_km_road_at_sixty_kmh_is_thirty_minutes():
    costs = np.full((3, 1001), 0.012, np.float32)     # walking, 5 km/h
    costs[1, :] = 0.001                               # road, 60 km/h = 0.001 min per metre
    dist = dijkstra.travel_time(costs, [1 * 1001 + 0], CELL)
    assert abs(dist[1, 1000] - 30.0) < 1e-3


def test_river_with_one_bridge():
    costs = np.full((21, 21), 0.01, np.float32)
    costs[:, 10] = np.nan
    costs[20, 10] = 0.01                              # the only crossing, at the bottom
    dist = dijkstra.travel_time(costs, [0], CELL)
    assert_same(dist, reference(costs, [0]))
    straight = 20 * 30 * 0.01
    assert dist[0, 20] > straight * 2                 # forced down to the bridge and back


def test_island_is_unreachable():
    costs = np.full((15, 15), 0.01, np.float32)
    costs[4:11, 4] = costs[4:11, 10] = np.nan
    costs[4, 4:11] = costs[10, 4:11] = np.nan
    dist = dijkstra.travel_time(costs, [0], CELL)
    assert np.isinf(dist[5:10, 5:10]).all()
    assert np.isfinite(dist[0:3, 0:3]).all()


def test_cap_keeps_values_below_it_and_drops_the_rest():
    rng = np.random.default_rng(1)
    costs = random_grid(rng, barrier=0.05)
    full = dijkstra.travel_time(costs, [0], CELL)
    capped = dijkstra.travel_time(costs, [0], CELL, max_minutes=5.0)
    near = full <= 5.0
    np.testing.assert_array_equal(capped[near], full[near])
    assert np.isinf(capped[~near]).all()


def test_reach_search_resets_between_runs():
    rng = np.random.default_rng(2)
    costs = random_grid(rng)
    starts = rng.choice(np.flatnonzero(costs.ravel() > 0), size=4, replace=False)
    search = dijkstra.ReachSearch(costs, CELL)
    for s in starts:
        got = search.run([s], 6.0).reshape(costs.shape).copy()
        np.testing.assert_array_equal(got, dijkstra.travel_time(costs, [s], CELL, max_minutes=6.0))


def test_pruning_changes_no_answer_at_demand_points():
    rng = np.random.default_rng(4)
    limit = 5.0
    for _ in range(5):
        costs = random_grid(rng, shape=(60, 70), barrier=0.1)
        passable = np.flatnonzero(costs.ravel() > 0)
        existing = rng.choice(passable, size=4, replace=False)
        base = dijkstra.travel_time(costs, existing, CELL).ravel()
        demand = np.flatnonzero(base > limit)          # beyond the limit today, incl. unreachable
        assert demand.size > 0
        floor = float(np.min(base[demand]))
        pruned = dijkstra.ReachSearch(costs, CELL, prune_field=base.reshape(costs.shape),
                                      prune_offset=floor - limit - 1e-3)
        plain = dijkstra.ReachSearch(costs, CELL)
        for cand in rng.choice(passable, size=8, replace=False):
            a = plain.run([cand], limit)[demand].copy()
            explored_plain = plain.last_expanded
            b = pruned.run([cand], limit)[demand].copy()
            np.testing.assert_array_equal(a <= limit, b <= limit)
            np.testing.assert_array_equal(a[a <= limit], b[a <= limit])
            assert pruned.last_expanded <= explored_plain
