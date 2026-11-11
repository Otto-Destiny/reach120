"""Step 7: a one-page brief for a state, as PDF and PNG.

Every number comes from the result files and every sentence is built by rule; no language
model is involved. Colour follows meaning across the page: orange is the recommended plan,
on the map and in the chart.
"""
from __future__ import annotations

import json
import textwrap
import time

import numpy as np
import pandas as pd

from . import display, geo, scenarios
from .config import load, settings
from .store import StateStore, national, read_json

# Chart series in categorical slot order: the recommended plan is slot 2 (orange) so it
# matches the upgrade markers on the map.
CURVES = [("straight_line", "Straight-line ranking"), ("exact", "Recommended plan"), ("greedy", "Greedy ranking")]


def _pct(x) -> str:
    if x is None:
        return "n/a"
    if 0 < x < 0.0005:
        return "under 0.1%"
    return f"{100 * x:.1f}%"


def _plan_for(state: str, scenario: str, mode: str, run_id: str | None = None) -> dict | None:
    from .plan import default_run

    if run_id:
        path = StateStore(state).plan(run_id)
        return read_json(path) if path.exists() else None
    return default_run(state, scenario, mode)


def sentences(summary: dict, plan: dict | None, scenario: str) -> list[str]:
    """The brief's text, built by rule from the result files."""
    target_h = summary["target_min"] / 60
    hours = f"{target_h:g} hour" + ("" if target_h == 1 else "s")
    label = scenarios.label(scenario).lower()
    runs = summary["runs"]
    out = []
    vehicle, walk = runs.get(f"{scenario}/mixed"), runs.get(f"{scenario}/walk")
    if vehicle:
        out.append(f"With a vehicle on passable roads, {_pct(vehicle['share_beyond_target'])} of women aged 15-49 in "
                   f"{summary['state']} ({vehicle['beyond_target']:,.0f} women) live more than {hours} from a hospital "
                   f"that can likely perform an emergency caesarean section ({label} definition).")
    if walk:
        out.append(f"On foot, {_pct(walk['share_beyond_target'])} ({walk['beyond_target']:,.0f} women) do.")
    lo, hi = summary.get("range_share_beyond_target") or (None, None)
    if lo is not None:
        out.append(f"Across every capability definition and both transport assumptions the share ranges from "
                   f"{_pct(lo)} to {_pct(hi)}.")
    if plan:
        travel = "on foot" if plan["mode"] == "walk" else "by vehicle"
        if plan["mode"] == "walk" and vehicle and vehicle["share_beyond_target"] < 0.02:
            out.append(f"The plan below assumes travel on foot, because with a vehicle almost every woman is already "
                       f"within {hours}: in {summary['state']} getting transport matters more than where the hospitals are.")
        p = plan["plan"]
        if p["sites"]:
            cost_word = "a placeholder cost of " if plan["costs_are_placeholders"] else ""
            out.append(f"Upgrading the {len(p['sites'])} facilities below, at {cost_word}{display.format_naira(p['cost_naira'])}, "
                       f"would bring {p['women_newly_within_target']:,.0f} more women within {hours} {travel}, cutting the "
                       f"share to {_pct(plan['after']['share_beyond_target'])}.")
            gain = p["women_newly_within_target"] - plan["straight_line"]["women_newly_within_target"]
            if gain > 0:
                out.append(f"Picking sites by straight-line distance instead would reach {gain:,.0f} fewer women "
                           f"for the same budget.")
        else:
            out.append(f"No upgrade within {display.format_naira(plan['budget_naira'])} brings more women within {hours} {travel}.")
    return out


def _rings(geometry):
    if geometry["type"] == "Polygon":
        return geometry["coordinates"]
    if geometry["type"] == "MultiPolygon":
        return [ring for poly in geometry["coordinates"] for ring in poly]
    return []


def build(state: str, scenario: str | None = None, run_id: str | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch

    from .plan import headline_mode

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    store = StateStore(state)
    summary = read_json(store.summary)
    scenario = scenario or scenarios.default_scenario()
    if scenario not in settings()["planning"]["scenarios"]:
        scenario = "base"
    mode = headline_mode(summary, scenario)
    plan = _plan_for(state, scenario, mode, run_id)
    if plan:
        scenario, mode = plan["scenario"], plan["mode"]
    disp = read_json(store.display_meta)
    costs_cfg = load("costs")

    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
    fig.patch.set_facecolor(display.SURFACE)

    def text(x, y, s, size=9, color=display.INK, weight="normal", **kw):
        fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, va="top", **kw)

    text(0.06, 0.965, f"Reach120 · {summary['state']}", 18, weight="bold")
    text(0.06, 0.935, "Two-hour access to emergency obstetric care, women aged 15-49", 10.5, display.INK_SECONDARY)
    text(0.94, 0.965, time.strftime("%d %B %Y"), 8.5, display.MUTED, ha="right")
    if costs_cfg.get("placeholder", True):
        text(0.94, 0.947, "Draft: upgrade costs are placeholders", 8.5, display.INK_SECONDARY, ha="right", weight="bold")

    y = 0.905
    for s in sentences(summary, plan, scenario):
        wrapped = textwrap.fill(s, 105)
        text(0.06, y, wrapped, 9.5)
        y -= 0.0165 * (wrapped.count("\n") + 1) + 0.006

    # Map: travel-time bands, ward outlines, hospitals, upgrades.
    ax = fig.add_axes([0.04, 0.435, 0.52, y - 0.46])
    run_disp = disp["runs"][f"{scenario}/{mode}"]
    w, s, e, n = run_disp["bounds"]
    x0, y0 = geo.mercator(w, s)
    x1, y1 = geo.mercator(e, n)
    img = plt.imread(store.display_png(scenario, mode))
    ax.imshow(img, extent=(float(x0), float(x1), float(y0), float(y1)), interpolation="nearest", zorder=1)
    gj = json.loads(store.display_wards.read_text(encoding="utf-8"))
    segs = []
    for feat in gj["features"]:
        for ring in _rings(feat["geometry"]):
            arr = np.asarray(ring)
            mx, my = geo.mercator(arr[:, 0], arr[:, 1])
            segs.append(np.column_stack([mx, my]))
    ax.add_collection(LineCollection(segs, colors="white", linewidths=0.25, alpha=0.7, zorder=2))
    fac = pd.read_parquet(store.facilities)
    cap = fac[fac[f"cap_{scenario}"] & fac["in_state"]]
    fx, fy = geo.mercator(cap["longitude"].to_numpy(), cap["latitude"].to_numpy())
    ax.scatter(fx, fy, s=9, color=display.INK, edgecolors="white", linewidths=0.6, zorder=3)
    if plan and plan["plan"]["sites"]:
        sites = plan["plan"]["sites"]
        ux, uy = geo.mercator([p["longitude"] for p in sites], [p["latitude"] for p in sites])
        ax.scatter(ux, uy, s=34, color=display.UPGRADE, edgecolors="white", linewidths=1.0, zorder=4)
        for i, (a, b) in enumerate(zip(ux, uy), start=1):
            ax.annotate(str(i), (a, b), xytext=(4, 3), textcoords="offset points", fontsize=7, color=display.INK,
                        fontweight="bold", zorder=5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Travel time {'on foot' if mode == 'walk' else 'by vehicle'} to the nearest capable hospital",
                 fontsize=8.5, color=display.INK, loc="left")
    handles = [Patch(color=c, label=l) for l, c in disp["legend"]]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=display.INK, markeredgecolor="white", markersize=4,
                              label="Hospital counted as capable"))
    if plan and plan["plan"]["sites"]:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=display.UPGRADE, markeredgecolor="white", markersize=6,
                                  label="Recommended upgrade"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=2, fontsize=7, frameon=False,
              labelcolor=display.INK_SECONDARY)

    # Budget curve.
    curve_path = store.curve(scenario, mode)
    bx = fig.add_axes([0.63, 0.55, 0.33, y - 0.58])
    if curve_path.exists():
        curve = pd.read_parquet(curve_path)
        xs = curve["budget_naira"] / 1e9
        # Colours by series identity; the recommended plan is drawn last so it is never hidden.
        colours = dict(zip([c for c, _ in CURVES], display.SERIES_LIGHT))
        for col, label in sorted(CURVES, key=lambda cl: cl[0] == "exact"):
            bx.plot(xs, curve[col] / 1e3, color=colours[col], lw=1.5, drawstyle="steps-post", label=label,
                    zorder=3 if col == "exact" else 2)
        # Direct labels at the right end, spaced so they never collide.
        end = curve.iloc[-1]
        span = max(float(curve[[c for c, _ in CURVES]].to_numpy().max()) / 1e3, 1e-9)
        pos = display.spread_labels({label: float(end[col]) / 1e3 for col, label in CURVES}, 0.075 * span)
        for col, label in CURVES:
            bx.annotate(label, (float(end["budget_naira"]) / 1e9, pos[label]), xytext=(-2, 5), textcoords="offset points",
                        fontsize=6.5, color=display.INK_SECONDARY, ha="right", va="bottom")
        bx.set_ylim(0, max(pos.values()) * 1.12 + 1e-9)
        if plan:
            bx.axvline(plan["budget_naira"] / 1e9, color=display.MUTED, lw=0.8, ls=(0, (2, 2)))
        bx.set_xlabel("Budget (₦ billion" + (", placeholder costs)" if costs_cfg.get("placeholder", True) else ")"),
                      fontsize=7.5, color=display.INK_SECONDARY)
        bx.set_title(f"More women within two hours {'on foot' if mode == 'walk' else 'by vehicle'} (thousands)",
                     fontsize=8.5, color=display.INK, loc="left")
        handles, labels = bx.get_legend_handles_labels()
        order = [labels.index(lbl) for _, lbl in CURVES]
        bx.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=6.5, frameon=False,
                  loc="lower right", labelcolor=display.INK_SECONDARY)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        bx.spines[side].set_color(display.AXIS)
    bx.tick_params(colors=display.MUTED, labelsize=7)
    bx.grid(color=display.GRID, lw=0.5)
    bx.set_axisbelow(True)
    bx.set_facecolor(display.SURFACE)

    # Worst-served LGAs, for the same scenario and mode as the map.
    from .tables import lga_table

    lgas = lga_table(pd.read_parquet(store.coverage(scenario, mode))).head(6).to_dict(orient="records")
    ty = 0.505
    text(0.63, ty, f"Worst-served LGAs {'on foot' if mode == 'walk' else 'by vehicle'}", 8.5, weight="bold")
    ty -= 0.016
    text(0.63, ty, "LGA", 7, display.MUTED)
    text(0.85, ty, "Beyond 2 hours", 7, display.MUTED, ha="right")
    text(0.96, ty, "Women", 7, display.MUTED, ha="right")
    for row in lgas:
        ty -= 0.0145
        text(0.63, ty, textwrap.shorten(str(row["lga"]), 22), 7.8, display.INK_SECONDARY)
        text(0.85, ty, _pct(row["share_beyond_target"]), 7.8, display.INK, ha="right")
        text(0.96, ty, f"{row['beyond_target']:,.0f}", 7.8, display.INK, ha="right")

    # Upgrade table.
    ty = 0.355
    if plan and plan["plan"]["sites"]:
        text(0.06, ty, f"Recommended upgrades within {display.format_naira(plan['budget_naira'])} "
                       f"({plan['scenario_label'].lower()} definition, travel {'on foot' if mode == 'walk' else 'by vehicle'})",
             9.5, weight="bold")
        ty -= 0.022
        cols = [(0.06, "#"), (0.09, "Facility"), (0.47, "LGA"), (0.64, "Type"), (0.80, "Cost"), (0.94, "Women reached")]
        for xpos, head in cols:
            text(xpos, ty, head, 7.5, display.MUTED, ha="right" if head in ("Cost", "Women reached") else "left")
        for i, site in enumerate(plan["plan"]["sites"][:10], start=1):
            ty -= 0.0155
            text(0.06, ty, str(i), 7.5, display.INK_SECONDARY)
            text(0.09, ty, textwrap.shorten(site["facility_name"], 46), 7.5)
            text(0.47, ty, textwrap.shorten(site["lga"], 20), 7.5, display.INK_SECONDARY)
            text(0.64, ty, site["tier"], 7.5, display.INK_SECONDARY)
            text(0.80, ty, display.format_naira(site["cost_naira"]), 7.5, ha="right")
            text(0.94, ty, f"{site['women_newly_within_target']:,.0f}", 7.5, ha="right")
        if len(plan["plan"]["sites"]) > 10:
            ty -= 0.0155
            text(0.09, ty, f"and {len(plan['plan']['sites']) - 10} more in the plan file", 7.5, display.MUTED)

    # Caveats and sources.
    sc = read_json(national("capability_scorecard.json")) if national("capability_scorecard.json").exists() else None
    caveats = [
        "Capability is inferred from facility type: the data does not say which hospitals operate. "
        + (f"Against OnTIME's verified list in 15 cities, the {scenarios.label(scenario).lower()} definition catches "
           f"{_pct(next((r['catch_rate'] for r in sc['scenarios'] if r['scenario'] == scenario), None))} of verified facilities."
           if sc else ""),
        "Functional status is unknown for many facilities; only those marked Not-Functional or Closed are left out.",
        "Travel times assume dry-season roads and no traffic. Rivers are barriers. Costs, referral delays and staffing are not modelled.",
        "Upgrade costs are placeholders to be replaced from a state budget or costing study." if costs_cfg.get("placeholder", True)
        else "Upgrade costs come from config/costs.yaml.",
    ]
    ty = min(ty, 0.22) - 0.035
    text(0.06, ty, "What to keep in mind", 8.5, weight="bold")
    ty -= 0.02
    for c in caveats:
        wrapped = textwrap.fill("• " + c, 135)
        text(0.06, ty, wrapped, 7.2, display.INK_SECONDARY)
        ty -= 0.0125 * (wrapped.count("\n") + 1) + 0.004
    sources = ("Data: GRID3 Nigeria health facilities v4.0, wards v4.0 and travel-time friction surface v1.0 (CIESIN, "
               "Columbia University; CC BY 4.0 and CC BY-SA 4.0); WorldPop/GRID3 population v3.0 (CC BY 4.0); OnTIME "
               "verified facilities and travel times (CC BY 4.0). Maps derived from CC BY-SA data are shared under "
               "CC BY-SA 4.0. Built with Reach120.")
    text(0.06, 0.045, textwrap.fill(sources, 170), 6.3, display.MUTED)

    fig.savefig(store.brief_pdf, facecolor=fig.get_facecolor())
    fig.savefig(store.brief_png, facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    return {"pdf": str(store.brief_pdf), "png": str(store.brief_png), "plan": plan["run_id"] if plan else None}
