"""Settings and paths. Everything you might change lives in ../config/*.yaml."""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("REACH120_CONFIG", PROJECT_ROOT / "config"))


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def settings() -> dict:
    return load("settings")


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def shared_path(*parts: str) -> Path:
    """A path inside the shared data folder for the national layers ($REACH120_SHARED_DATA or settings.yaml)."""
    base = os.environ.get("REACH120_SHARED_DATA") or (PROJECT_ROOT / settings()["paths"]["shared_data"])
    return Path(base).resolve().joinpath(*parts)


def outputs_root() -> Path:
    return Path(os.environ.get("REACH120_OUTPUTS") or (PROJECT_ROOT / settings()["paths"]["outputs"]))


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


_FCT = {"fct", "fct abuja", "abuja", "federal capital territory"}


def norm_name(value) -> str:
    """Normalise a state, LGA or ward name so spellings from different datasets compare equal."""
    if value is None:
        return ""
    s = str(value).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\bstate\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return "fct" if s in _FCT else s
