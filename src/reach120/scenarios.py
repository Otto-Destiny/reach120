"""Capability scenarios: which facilities count as able to do a caesarean section."""
from __future__ import annotations

import pandas as pd

from .config import load
from .store import national, read_json


def config() -> dict:
    return load("scenarios")


def names(planner_only: bool = False) -> list[str]:
    sc = config()["scenarios"]
    return [k for k, v in sc.items() if v.get("planner", False) or not planner_only]


def label(name: str) -> str:
    return config()["scenarios"][name].get("label", name)


def description(name: str) -> str:
    return config()["scenarios"][name].get("description", "")


def default_scenario() -> str:
    choice = config().get("default", "base")
    if choice == "auto":
        path = national("capability_scorecard.json")
        if path.exists():
            rec = read_json(path).get("recommended")
            if rec in config()["scenarios"]:
                return rec
        return "base"
    return choice


def operating(df: pd.DataFrame) -> pd.Series:
    """Not marked Not-Functional or Closed. Unknown status counts as operating, and says so."""
    excluded = set(config().get("exclude_functional", []))
    return ~df["functional"].isin(excluded)


def capable(df: pd.DataFrame, name: str) -> pd.Series:
    spec = config()["scenarios"][name]
    mask = df["facility_type"].isin(spec["facility_types"]) & operating(df)
    if spec.get("ownership"):
        mask &= df["facility_ownership"].isin(spec["ownership"])
    return mask


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name in names():
        out[f"cap_{name}"] = capable(out, name).to_numpy()
    return out


def nested_check() -> list[str]:
    """The planner assumes its scenarios are nested (each a superset of the one before)."""
    sc = config()["scenarios"]
    planner = names(planner_only=True)
    problems = []
    for a, b in zip(planner, planner[1:]):
        if not set(sc[a]["facility_types"]) <= set(sc[b]["facility_types"]) or sc[a].get("ownership") != sc[b].get("ownership"):
            problems.append(f"{a} is not contained in {b}")
    return problems
