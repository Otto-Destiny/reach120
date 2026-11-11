"""Where every output file lives, plus JSON helpers.

Layout, per state:

    outputs/<state>/inputs/     friction windows, population, wards, facilities
    outputs/<state>/access/     travel-time rasters
    outputs/<state>/coverage/   ward tables and the summary
    outputs/<state>/display/    map images and ward outlines for the app and the brief
    outputs/<state>/planning/   demand points, candidates, reach matrix, budget curves
    outputs/<state>/plans/      one JSON file per plan run
    outputs/<state>/validation/ checks and the validation report
    outputs/<state>/brief/      the one-page brief
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import outputs_root, slug


def _clean(o):
    """Make an object JSON-safe: numpy types to Python, NaN and infinity to null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return str(o)
    return o


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class StateStore:
    """Paths for one state's outputs. Directories are created on first use."""

    def __init__(self, state: str):
        self.state = state
        self.slug = slug(state)
        self.root = outputs_root() / self.slug

    def _p(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # inputs
    def friction(self, mode: str) -> Path:
        return self._p("inputs", f"friction_{mode}.tif")

    @property
    def window(self) -> Path:
        return self._p("inputs", "window.json")

    @property
    def state_geojson(self) -> Path:
        return self._p("inputs", "state.geojson")

    @property
    def facilities(self) -> Path:
        return self._p("inputs", "facilities.parquet")

    @property
    def women(self) -> Path:
        return self._p("inputs", "women_15_49.tif")

    @property
    def population(self) -> Path:
        return self._p("inputs", "population_total.tif")

    @property
    def ward_ids(self) -> Path:
        return self._p("inputs", "ward_ids.tif")

    @property
    def wards(self) -> Path:
        return self._p("inputs", "wards.parquet")

    @property
    def wards_geojson(self) -> Path:
        return self._p("inputs", "wards.geojson")

    # access
    def tt30(self, scenario: str, mode: str) -> Path:
        return self._p("access", f"tt30_{scenario}_{mode}.tif")

    def ttpop(self, scenario: str, mode: str) -> Path:
        return self._p("access", f"ttpop_{scenario}_{mode}.tif")

    def run_meta(self, scenario: str, mode: str) -> Path:
        return self._p("access", f"run_{scenario}_{mode}.json")

    # coverage and display
    def coverage(self, scenario: str, mode: str) -> Path:
        return self._p("coverage", f"wards_{scenario}_{mode}.parquet")

    @property
    def summary(self) -> Path:
        return self._p("coverage", "summary.json")

    def display_png(self, scenario: str, mode: str) -> Path:
        return self._p("display", f"tt_{scenario}_{mode}.png")

    @property
    def display_meta(self) -> Path:
        return self._p("display", "display.json")

    @property
    def display_wards(self) -> Path:
        return self._p("display", "wards.geojson")

    # planning (one set per transport mode)
    def demand(self, mode: str) -> Path:
        return self._p("planning", f"demand_{mode}.parquet")

    def candidates(self, mode: str) -> Path:
        return self._p("planning", f"candidates_{mode}.parquet")

    def reach(self, mode: str) -> Path:
        return self._p("planning", f"reach_{mode}.npz")

    def reach_meta(self, mode: str) -> Path:
        return self._p("planning", f"reach_{mode}.json")

    def curve(self, scenario: str, mode: str) -> Path:
        return self._p("planning", f"curve_{scenario}_{mode}.parquet")

    @property
    def default_runs(self) -> Path:
        return self.plans_dir / "default_runs.json"

    @property
    def plans_dir(self) -> Path:
        d = self.root / "plans"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def plan(self, run_id: str) -> Path:
        return self.plans_dir / f"{run_id}.json"

    # validation, brief, assistant
    def validation(self, name: str) -> Path:
        return self._p("validation", name)

    @property
    def brief_pdf(self) -> Path:
        return self._p("brief", f"reach120_{self.slug}_brief.pdf")

    @property
    def brief_png(self) -> Path:
        return self._p("brief", f"reach120_{self.slug}_brief.png")

    @property
    def assistant_log(self) -> Path:
        return self._p("assistant_log.jsonl")


def national(name: str) -> Path:
    p = outputs_root() / "_national" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache(name: str) -> Path:
    p = outputs_root() / "_cache" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def available_states() -> list[str]:
    """States with a finished coverage step, by their display name."""
    root = outputs_root()
    states = []
    if root.exists():
        for d in sorted(root.iterdir()):
            summary = d / "coverage" / "summary.json"
            if d.is_dir() and summary.exists():
                try:
                    states.append(read_json(summary)["state"])
                except (KeyError, ValueError, OSError):
                    continue
    return states
