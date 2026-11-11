"""How good is "facility type" as a stand-in for "can do a caesarean section"?

The OnTIME consortium verified 2,021 facilities with emergency caesarean capacity in 15 city
regions (by phone, registries and visits). Each one is matched to a GRID3 facility, by
shared national facility code where both have one, otherwise by name and distance. Then,
for every capability scenario:

* **catch rate**: the share of matched verified facilities the scenario counts as capable;
* **false inclusion**: the share of facilities the scenario counts as capable, inside the
  OnTIME study area, that are not on the verified list.

The verified list may itself be incomplete, so "not on the list" is not proof of "cannot
operate". The study area is urban, so rural capability may differ.

Two things about the data shape the matching. Coordinates for the same hospital often differ
between the two sources by 0.5-4 km, so a strong name match is accepted up to 1.5 km away.
And GRID3 holds duplicate records of some hospitals, so a pair counts in both directions: a
verified facility is caught if any acceptable GRID3 record of it is capable, and a GRID3
facility is on the list if any verified facility matches it.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from . import geo, inputs, scenarios
from .config import project_path, settings
from .store import cache, national, write_json

CONURBATIONS = {1: "Lagos", 2: "Aba", 3: "Abuja", 4: "Benin City", 5: "Ibadan", 6: "Ilorin", 7: "Jos", 8: "Kaduna",
                9: "Kano", 10: "Maiduguri", 11: "Onitsha", 12: "Owerri", 13: "Port Harcourt", 14: "Uyo", 15: "Warri"}
GENERIC = {"hospital", "hospitals", "hosp", "clinic", "clinics", "medical", "medicals", "centre", "center", "centres",
           "maternity", "home", "specialist", "specialists", "general", "the", "and", "of", "ltd", "limited", "nig",
           "nigeria", "plc", "health", "healthcare", "care", "services", "service", "consultants", "consultant",
           "diagnostic", "diagnostics", "and", "in", "at", "co", "company", "int", "international", "mat", "hosp."}
AREA_RADIUS_M = 600      # a facility this close to an OnTIME grid cell centre is inside the study area


def name_key(s) -> str:
    s = str(s or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s) -> set[str]:
    return {t for t in name_key(s).split() if t not in GENERIC and len(t) > 1}


def similarity(a, b) -> float:
    """Name similarity on the distinctive words, so shared words like "hospital" do not count."""
    ta, tb = tokens(a), tokens(b)
    if ta and tb:
        seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
        overlap = len(ta & tb) / min(len(ta), len(tb))
        return max(seq, overlap)
    ka, kb = name_key(a), name_key(b)       # a name made only of generic words: weaker evidence
    return 0.8 * SequenceMatcher(None, ka, kb).ratio() if ka and kb else 0.0


def acceptable(distance_m: float, score: float, same_code: bool) -> str | None:
    """Why a pair counts as the same facility, or None."""
    if same_code and distance_m <= 5000:
        return "code"
    if score >= 0.85 and distance_m <= 1500:
        return "strong name"
    if score >= 0.7 and distance_m <= 300:
        return "name and distance"
    if score >= 0.4 and distance_m <= 50:
        return "very close"
    return None


def load_ontime() -> pd.DataFrame:
    df = pd.read_csv(project_path(settings()["paths"]["ontime_facilities"]), encoding="latin1", low_memory=False)
    out = pd.DataFrame({
        "ontime_code": df["ontime_code"].astype(str),
        "facility_name": df["facility_name"].fillna("").astype(str).str.strip(),
        "alt_name": df.get("alt_name_fac", pd.Series([""] * len(df))).fillna("").astype(str),
        "nhfr_code": df["facility_code"].fillna("").astype(str).str.strip(),
        "conurbation": df["urban_conurb"].map(CONURBATIONS),
        "ownership": df["owner"].map({1: "Public", 2: "Private"}),
        "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
    })
    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    # The data dictionary labels latitude "(x)" and longitude "(y)". Maiduguri (lat ~11.8, lon ~13.1)
    # tells the two apart: check the columns are what their names say.
    mdg = out[out["conurbation"] == "Maiduguri"]
    if len(mdg) and not (mdg["latitude"].median() < 12.5 < mdg["longitude"].median()):
        raise ValueError("OnTIME latitude and longitude look swapped")
    return out


def ontime_cells() -> pd.DataFrame:
    """Unique grid-cell centres of the OnTIME travel-time database (cached)."""
    path = cache("ontime_cells.parquet")
    if path.exists():
        return pd.read_parquet(path)
    seen = set()
    for chunk in pd.read_csv(project_path(settings()["paths"]["ontime_travel_times"]), usecols=["center_lat_lng"],
                             chunksize=500_000):
        seen.update(chunk["center_lat_lng"].unique().tolist())
    lat, lon = zip(*(map(float, s.split()) for s in seen))
    cells = pd.DataFrame({"lat": lat, "lon": lon})
    cells.to_parquet(path, index=False)
    return cells


def pairs(g3: pd.DataFrame, ont: pd.DataFrame) -> pd.DataFrame:
    """Every acceptable (OnTIME facility, GRID3 facility) pair."""
    from scipy.spatial import cKDTree

    gx, gy = geo.to_utm(g3["longitude"].to_numpy(), g3["latitude"].to_numpy())
    ox, oy = geo.to_utm(ont["longitude"].to_numpy(), ont["latitude"].to_numpy())
    tree = cKDTree(np.column_stack([gx, gy]))
    codes = g3["nhfr_facility_code"].astype(str).to_numpy()
    names = g3["facility_name"].to_numpy()
    rows = []
    for i, near in enumerate(tree.query_ball_point(np.column_stack([ox, oy]), r=5000)):
        o = ont.iloc[i]
        for j in near:
            d = float(np.hypot(gx[j] - ox[i], gy[j] - oy[i]))
            same_code = bool(o["nhfr_code"]) and o["nhfr_code"] == codes[j]
            if d > 1500 and not same_code:
                continue
            score = similarity(o["facility_name"], names[j])
            if o["alt_name"]:
                score = max(score, similarity(o["alt_name"], names[j]))
            why = acceptable(d, score, same_code)
            if why:
                rows.append({"ontime_code": o["ontime_code"], "g3_index": int(j), "unique_id": g3["unique_id"].iloc[j],
                             "distance_m": round(d, 1), "name_score": round(score, 2), "method": why})
    return pd.DataFrame(rows, columns=["ontime_code", "g3_index", "unique_id", "distance_m", "name_score", "method"])


def best_matches(ont: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    """One row per verified facility: its best GRID3 match, if any (for the published match table)."""
    rank = {"code": 0, "strong name": 1, "name and distance": 2, "very close": 3}
    best = (pr.assign(r=pr["method"].map(rank)).sort_values(["ontime_code", "r", "distance_m"])
            .drop_duplicates("ontime_code").drop(columns=["r", "g3_index"]))
    out = ont.merge(best, on="ontime_code", how="left")
    out["method"] = out["method"].fillna("unmatched")
    return out


def run() -> dict:
    from scipy.spatial import cKDTree

    g3 = scenarios.add_flags(inputs.facilities_all())
    ont = load_ontime()
    pr = pairs(g3, ont)
    matched = best_matches(ont, pr)
    cells = ontime_cells()
    cx, cy = geo.to_utm(cells["lon"].to_numpy(), cells["lat"].to_numpy())
    gx, gy = geo.to_utm(g3["longitude"].to_numpy(), g3["latitude"].to_numpy())
    dist, _ = cKDTree(np.column_stack([cx, cy])).query(np.column_stack([gx, gy]), k=1)
    g3["in_study_area"] = dist <= AREA_RADIUS_M
    verified_g3 = set(pr["g3_index"])                       # GRID3 records some verified facility matches
    flags = g3[[f"cap_{s}" for s in scenarios.names()]].to_numpy()
    pr_caps = pd.DataFrame(flags[pr["g3_index"].to_numpy()], columns=scenarios.names()).assign(ontime_code=pr["ontime_code"].to_numpy())
    per_verified = pr_caps.groupby("ontime_code").any()     # caught if any acceptable record is capable
    n_matched = len(per_verified)
    hit = matched[matched["unique_id"].notna()].merge(g3[["unique_id", "facility_type"]], on="unique_id", how="left")

    rows = []
    for s in scenarios.names():
        caught = int(per_verified[s].sum())
        in_area = g3["in_study_area"] & g3[f"cap_{s}"]
        unverified = int((in_area & ~pd.Series(np.arange(len(g3))).isin(verified_g3).to_numpy()).sum())
        recall = caught / n_matched if n_matched else None
        precision = 1 - unverified / int(in_area.sum()) if in_area.sum() else None
        f1 = 2 * recall * precision / (recall + precision) if recall and precision else None
        rows.append({"scenario": s, "label": scenarios.label(s), "verified_matched": n_matched, "caught": caught,
                     "catch_rate": recall, "capable_in_study_area": int(in_area.sum()),
                     "not_on_verified_list": unverified, "false_inclusion_rate": 1 - precision if precision is not None else None,
                     "balance_f1": f1})
    table = pd.DataFrame(rows)
    planner = settings()["planning"]["scenarios"]
    scored = table[table["scenario"].isin(planner) & table["balance_f1"].notna()]
    recommended = scored.sort_values("balance_f1", ascending=False)["scenario"].iloc[0] if len(scored) else "base"
    by_type = hit["facility_type"].replace("", "(blank)").value_counts().head(12).to_dict()
    strong = pr[pr["method"] == "strong name"]
    result = {
        "verified_total": int(len(ont)), "verified_matched": int(n_matched),
        "match_methods": matched["method"].value_counts().to_dict(),
        "median_offset_m_strong_name_matches": float(strong["distance_m"].median()) if len(strong) else None,
        "grid3_records_matched": int(len(verified_g3)),
        "verified_private_share": float((ont["ownership"] == "Private").mean()),
        "matched_by_grid3_type": by_type, "scenarios": rows, "recommended": recommended,
        "study_area_radius_m": AREA_RADIUS_M,
        "note": "Not on the verified list is not proof a facility cannot operate. The study area is urban.",
    }
    write_json(national("capability_scorecard.json"), result)
    table.to_csv(national("capability_scorecard.csv"), index=False)
    matched.to_csv(national("ontime_matches.csv"), index=False)
    return result
