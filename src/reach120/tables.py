"""Coverage summaries that need only pandas, so the app and the brief can use them anywhere."""
from __future__ import annotations

import numpy as np
import pandas as pd


def totals(table: pd.DataFrame, thresholds=(30, 60, 120)) -> dict:
    women = float(table["women"].sum())
    res = {"women": women}
    for th in thresholds:
        res[f"within_{th}"] = float(table[f"within_{th}"].sum())
        res[f"share_within_{th}"] = res[f"within_{th}"] / women if women else None
    res["beyond_target"] = float(table["beyond_target"].sum())
    res["share_beyond_target"] = res["beyond_target"] / women if women else None
    res["no_route"] = float(table["no_route"].sum())
    return res


def lga_table(table: pd.DataFrame, thresholds=(30, 60, 120)) -> pd.DataFrame:
    cols = ["women", *[f"within_{t}" for t in thresholds], "no_route", "beyond_target"]
    g = table.groupby("lga", as_index=False)[cols].sum()
    g["share_beyond_target"] = np.where(g["women"] > 0, g["beyond_target"] / g["women"], np.nan)
    return g.sort_values("share_beyond_target", ascending=False).reset_index(drop=True)
