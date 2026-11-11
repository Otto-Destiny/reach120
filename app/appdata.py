"""Cached loaders for the app. Every cache key includes the file's modification time, so a
re-run of the pipeline shows up without restarting the app."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from reach120 import display, plan, scenarios  # noqa: E402
from reach120.store import StateStore, available_states, national, read_json  # noqa: E402


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def states() -> list[str]:
    return available_states()


def current_state() -> str:
    return st.session_state.get("state") or states()[0]


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except AttributeError:
        return False


def series_colours() -> list[str]:
    return display.SERIES_DARK if is_dark() else display.SERIES_LIGHT


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float):
    return read_json(path)


def summary(state: str) -> dict:
    p = StateStore(state).summary
    return _json(str(p), _mtime(p))


def display_meta(state: str) -> dict:
    p = StateStore(state).display_meta
    return _json(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def _png_url(path: str, mtime: float) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def png_url(state: str, scenario: str, mode: str) -> str:
    p = StateStore(state).display_png(scenario, mode)
    return _png_url(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def _polygons(path: str, mtime: float) -> list[dict]:
    """Ward outer rings as plain records, which suits pydeck's PolygonLayer and its tooltip."""
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for feat in gj["features"]:
        geom, props = feat["geometry"], feat["properties"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"] if geom["type"] == "MultiPolygon" else []
        for poly in polys:
            out.append({"polygon": poly[0], **props})
    return out


def ward_polygons(state: str) -> list[dict]:
    p = StateStore(state).display_wards
    return _polygons(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def _parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def facilities(state: str) -> pd.DataFrame:
    p = StateStore(state).facilities
    return _parquet(str(p), _mtime(p))


def coverage(state: str, scenario: str, mode: str) -> pd.DataFrame:
    p = StateStore(state).coverage(scenario, mode)
    return _parquet(str(p), _mtime(p))


def curve(state: str, scenario: str, mode: str) -> pd.DataFrame | None:
    p = StateStore(state).curve(scenario, mode)
    return _parquet(str(p), _mtime(p)) if p.exists() else None


@st.cache_resource(show_spinner=False)
def _plan_inputs(state: str, mode: str, mtime: float):
    return plan.load_inputs(state, mode)


def plan_inputs(state: str, mode: str):
    store = StateStore(state)
    return _plan_inputs(state, mode, _mtime(store.reach(mode)) + _mtime(store.candidates(mode)))


def default_plan(state: str, scenario: str, mode: str) -> dict | None:
    return plan.default_run(state, scenario, mode)


def headline_plan(state: str) -> dict | None:
    """The plan the brief shows: default scenario, in the mode where upgrades matter."""
    s = summary(state)
    scenario = s["default_run"].split("/")[0]
    if scenario not in plan.settings()["planning"]["scenarios"]:
        scenario = "base"
    return default_plan(state, scenario, plan.headline_mode(s, scenario))


def scorecard() -> dict | None:
    p = national("capability_scorecard.json")
    return _json(str(p), _mtime(p)) if p.exists() else None


def validation_report(state: str) -> str | None:
    p = StateStore(state).validation("validation_report.md")
    return p.read_text(encoding="utf-8") if p.exists() else None


def scenario_options(planner_only: bool = False) -> dict[str, str]:
    return {s: scenarios.label(s) for s in scenarios.names(planner_only)}


def mode_label(mode: str) -> str:
    return "Vehicles on passable roads" if mode == "mixed" else "Walking only"


def pct(share, short: bool = False) -> str:
    """A share as a percentage; tiny non-zero shares read "under 0.1%" rather than "0.0%"."""
    if share is None:
        return "n/a"
    if 0 < share < 0.0005:
        return "<0.1%" if short else "under 0.1%"
    return f"{100 * share:.1f}%"
