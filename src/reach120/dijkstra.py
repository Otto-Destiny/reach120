"""Least-cost travel time on a friction grid, in compiled code.

The friction grid holds minutes per metre. Each cell joins its eight neighbours. Moving from
cell a to neighbour b costs distance(a, b) x (friction_a + friction_b) / 2, the rule used by
scikit-image's MCP_Geometric, so with distances in metres the result is in minutes. Cells
that are NaN, zero, negative or infinite are impassable.

Two things the planner needs, which off-the-shelf routines do not provide:

* **A cap.** A search stops at ``max_minutes``, so it explores only what it can reach.
* **Exact pruning.** Let T be travel time to the nearest existing capable hospital, and let L
  be the smallest T over the demand points. A cell k can lie on a path of at most ``limit``
  minutes from a candidate to a demand point only if T(k) >= d(candidate, k) + (L - limit).
  This follows from the triangle inequality T(i) <= T(k) + d(k, i). Cells failing the test
  are never expanded. No answer changes, but each search shrinks from the whole two-hour
  isochrone to the area where the new site would beat the existing ones.

Distances are stored as float32. Heap keys are the float32 values widened to float64, so the
"stale entry" test compares equal numbers and never drops a live entry.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_DR = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int64)
_DC = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int64)
_BIG = 1e30
_DUMMY_F = np.zeros(1, dtype=np.float32)
_DUMMY_I = np.zeros(1, dtype=np.int64)


@njit(cache=True)
def _push(keys, vals, size, key, val):
    i = size
    while i > 0:
        parent = (i - 1) >> 1
        if keys[parent] <= key:
            break
        keys[i] = keys[parent]
        vals[i] = vals[parent]
        i = parent
    keys[i] = key
    vals[i] = val


@njit(cache=True)
def _pop(keys, vals, size):
    """Remove and return the smallest item. `size` is the heap size before removal."""
    top_k = keys[0]
    top_v = vals[0]
    last = size - 1
    if last > 0:
        k = keys[last]
        v = vals[last]
        i = 0
        while True:
            child = 2 * i + 1
            if child >= last:
                break
            right = child + 1
            if right < last and keys[right] < keys[child]:
                child = right
            if keys[child] >= k:
                break
            keys[i] = keys[child]
            vals[i] = vals[child]
            i = child
        keys[i] = k
        vals[i] = v
    return top_k, top_v


@njit(cache=True)
def _grow_heap(keys, vals, size):
    cap = keys.shape[0] * 2
    new_keys = np.empty(cap, np.float64)
    new_vals = np.empty(cap, np.int64)
    new_keys[:size] = keys[:size]
    new_vals[:size] = vals[:size]
    return new_keys, new_vals


@njit(cache=True)
def _grow_int(arr, n):
    out = np.empty(arr.shape[0] * 2, np.int64)
    out[:n] = arr[:n]
    return out


@njit(cache=True)
def _search(cf, df, rows, cols, sources, cell_h, cell_w, max_cost,
            use_prune, pf, prune_offset, track, touched, n_touched):
    """Multi-source Dijkstra over a flattened grid.

    cf: friction, float32[rows*cols]. df: distances, float32, pre-filled with inf (or holding
    the result of earlier runs on the same grid). Returns (n_touched, touched, expanded).
    """
    diag = np.sqrt(cell_h * cell_h + cell_w * cell_w)
    lengths = np.empty(8, np.float64)
    lengths[0] = cell_h
    lengths[1] = cell_h
    lengths[2] = cell_w
    lengths[3] = cell_w
    for k in range(4, 8):
        lengths[k] = diag

    keys = np.empty(1 << 16, np.float64)
    vals = np.empty(1 << 16, np.int64)
    size = 0

    for s in sources:
        c = cf[s]
        if not (c > 0.0 and c < _BIG):
            continue
        if df[s] > 0.0:
            if track and df[s] == np.inf:
                if n_touched >= touched.shape[0]:
                    touched = _grow_int(touched, n_touched)
                touched[n_touched] = s
                n_touched += 1
            df[s] = 0.0
            if size >= keys.shape[0]:
                keys, vals = _grow_heap(keys, vals, size)
            _push(keys, vals, size, 0.0, s)
            size += 1

    expanded = 0
    while size > 0:
        d, u = _pop(keys, vals, size)
        size -= 1
        if d > df[u]:
            continue          # stale entry
        if d > max_cost:
            break             # everything left is further still
        expanded += 1
        ur = u // cols
        uc = u - ur * cols
        cu = cf[u]
        for k in range(8):
            vr = ur + _DR[k]
            vc = uc + _DC[k]
            if vr < 0 or vr >= rows or vc < 0 or vc >= cols:
                continue
            v = vr * cols + vc
            cv = cf[v]
            if not (cv > 0.0 and cv < _BIG):
                continue
            nd = np.float32(d + lengths[k] * 0.5 * (cu + cv))
            if nd > max_cost or nd >= df[v]:
                continue
            if use_prune and pf[v] < nd + prune_offset:
                continue
            if track and df[v] == np.inf:
                if n_touched >= touched.shape[0]:
                    touched = _grow_int(touched, n_touched)
                touched[n_touched] = v
                n_touched += 1
            df[v] = nd
            if size >= keys.shape[0]:
                keys, vals = _grow_heap(keys, vals, size)
            _push(keys, vals, size, np.float64(nd), v)
            size += 1
    return n_touched, touched, expanded


def _flat32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).reshape(-1)


def travel_time(costs: np.ndarray, sources, cell_size, max_minutes: float = np.inf) -> np.ndarray:
    """Minutes from the nearest source to every cell. Unreached cells are inf.

    costs: 2-D friction in minutes per metre. sources: flat indices into costs.
    cell_size: (cell height, cell width) in metres.
    """
    rows, cols = costs.shape
    dist = np.full(rows * cols, np.inf, dtype=np.float32)
    src = np.unique(np.asarray(sources, dtype=np.int64).reshape(-1))
    _search(_flat32(costs), dist, rows, cols, src, float(cell_size[0]), float(cell_size[1]),
            float(max_minutes), False, _DUMMY_F, 0.0, False, _DUMMY_I, 0)
    return dist.reshape(rows, cols)


class ReachSearch:
    """Many capped single-source searches on one grid.

    Each run resets only the cells the previous run touched, so the cost of a run is the
    area it explores, not the size of the grid. With ``prune_field`` it applies the exact
    pruning described at the top of this module.
    """

    def __init__(self, costs: np.ndarray, cell_size, prune_field=None, prune_offset: float = 0.0):
        self.shape = costs.shape
        self.rows, self.cols = costs.shape
        self.cf = _flat32(costs)
        self.dist = np.full(self.rows * self.cols, np.inf, dtype=np.float32)
        self.touched = np.empty(1 << 20, dtype=np.int64)
        self.n_touched = 0
        self.cell_h, self.cell_w = float(cell_size[0]), float(cell_size[1])
        if prune_field is None:
            self.pf, self.use_prune = _DUMMY_F, False
        else:
            if prune_field.shape != costs.shape:
                raise ValueError("prune_field must have the same shape as costs")
            self.pf, self.use_prune = _flat32(prune_field), True
        self.prune_offset = float(prune_offset)
        self.last_expanded = 0

    def reset(self) -> None:
        if self.n_touched:
            self.dist[self.touched[: self.n_touched]] = np.inf
            self.n_touched = 0

    def run(self, sources, max_minutes: float) -> np.ndarray:
        """Search from `sources` (flat indices). Returns the flat distance array, which is
        overwritten by the next run: copy what you need."""
        self.reset()
        src = np.unique(np.asarray(sources, dtype=np.int64).reshape(-1))
        n, touched, expanded = _search(
            self.cf, self.dist, self.rows, self.cols, src, self.cell_h, self.cell_w,
            float(max_minutes), self.use_prune, self.pf, self.prune_offset, True, self.touched, 0)
        self.n_touched, self.touched, self.last_expanded = int(n), touched, int(expanded)
        return self.dist
