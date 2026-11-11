import json
import math

import altair as alt
import pandas as pd
import streamlit as st

import appdata
from reach120 import brief, display
from reach120 import plan as planner

state = appdata.current_state()
summary = appdata.summary(state)
opts = appdata.scenario_options(planner_only=True)
headline = appdata.headline_plan(state)
default_s = summary["default_run"].split("/")[0]
default_mode = headline["mode"] if headline else "mixed"
SERIES = [("straight_line", "Straight-line ranking"), ("exact", "Recommended plan"), ("greedy", "Greedy ranking")]

st.title("Planner")
st.caption("Pick a budget. The planner chooses the upgrades that bring the most women within two hours, and compares its "
           "choice with two simpler ways of picking sites.")

c0, c1, c2 = st.columns([1, 1, 3])
mode = c0.radio("Transport", ["mixed", "walk"], index=["mixed", "walk"].index(default_mode), format_func=appdata.mode_label,
                key="plan_mode")
scenario = c1.selectbox("Capability definition", list(opts), index=list(opts).index(default_s) if default_s in opts else 0,
                        format_func=opts.get, key="plan_scenario")
inp = appdata.plan_inputs(state, mode)
run = summary["runs"][f"{scenario}/{mode}"]
st.caption(f"Today {appdata.pct(run['share_beyond_target'])} of women ({run['beyond_target']:,.0f}) are more than two hours away "
           f"{'on foot' if mode == 'walk' else 'by vehicle'} under this definition.")
curve = appdata.curve(state, scenario, mode)
top = float(curve["budget_naira"].max()) if curve is not None and len(curve) else 10e9
top_m = max(100, int(math.ceil(top / 1e6 / 100) * 100))
default_m = int(min(planner.settings()["planning"]["default_budget_naira"] / 1e6, top_m))
budget_m = c2.slider("Budget (₦ million)", 0, top_m, default_m, step=50, key="plan_budget")

with st.expander("Upgrade costs: placeholders, replace them with real figures"):
    tiers = planner.default_tier_costs()
    labels = {k: v["label"] for k, v in planner.cost_settings()["tiers"].items()}
    table = pd.DataFrame([{"tier": k, "Upgrade type": labels[k], "Cost (₦ million)": tiers[k] / 1e6} for k in tiers])
    edited = st.data_editor(table, disabled=["tier", "Upgrade type"], hide_index=True, key="cost_editor",
                            column_config={"tier": None, "Cost (₦ million)": st.column_config.NumberColumn(min_value=0, step=10)})
    tier_costs = {r["tier"]: float(r["Cost (₦ million)"]) * 1e6 for _, r in edited.iterrows()}

if st.button("Find the best upgrades", type="primary"):
    with st.spinner("Optimising"):
        st.session_state[f"plan_{state}"] = planner.run_plan(inp, scenario, budget_m * 1e6, tier_costs=tier_costs)
res = st.session_state.get(f"plan_{state}")
if res is None or res["scenario"] != scenario or res["mode"] != mode:
    res = appdata.default_plan(state, scenario, mode)
    if res:
        st.info(f"Showing the precomputed plan for {display.format_naira(res['budget_naira'])}. Press the button to plan for the budget above.")

if res:
    p = res["plan"]
    m1, m2, m3 = st.columns(3)
    m1.metric("More women within two hours", f"{p['women_newly_within_target']:,.0f}")
    m2.metric("Cost of the chosen upgrades", display.format_naira(p["cost_naira"]),
              help="Placeholder costs" if res["costs_are_placeholders"] else None)
    before, after = res["baseline"]["share_beyond_target"], res["after"]["share_beyond_target"]
    if before is not None and after is not None:
        m3.metric("Women more than two hours away", f"{100 * after:.1f}%", delta=f"{100 * (after - before):.1f} points",
                  delta_color="inverse")
    st.dataframe(pd.DataFrame([
        {"Way of choosing": "Recommended plan (optimiser)", "Women reached": round(p["women_newly_within_target"]),
         "Cost": round(p["cost_naira"])},
        {"Way of choosing": "Greedy: best women per naira first", "Women reached": round(res["greedy"]["women_newly_within_target"]),
         "Cost": round(res["greedy"]["cost_naira"])},
        {"Way of choosing": "Straight-line: most women within 30 km",
         "Women reached": round(res["straight_line"]["women_newly_within_target"]), "Cost": round(res["straight_line"]["cost_naira"])}]),
        column_config={"Women reached": st.column_config.NumberColumn(format="localized"),
                       "Cost": st.column_config.NumberColumn("Cost (₦)", format="localized")}, hide_index=True, width="stretch")
    if p["sites"]:
        sites = pd.DataFrame(p["sites"])[["facility_name", "lga", "facility_type", "ownership", "tier", "cost_naira",
                                          "women_newly_within_target", "women_reachable_alone"]]
        num = ["cost_naira", "women_newly_within_target", "women_reachable_alone"]
        sites[num] = sites[num].round().astype(int)
        st.subheader(f"{len(sites)} recommended upgrades")
        st.dataframe(sites.rename(columns={"facility_name": "Facility", "lga": "LGA", "facility_type": "Type", "ownership": "Ownership",
                                           "tier": "Cost tier", "cost_naira": "Cost (₦)", "women_newly_within_target": "Women reached",
                                           "women_reachable_alone": "Reachable if alone"}),
                     column_config={"Cost (₦)": st.column_config.NumberColumn(format="localized"),
                                    "Women reached": st.column_config.NumberColumn(format="localized"),
                                    "Reachable if alone": st.column_config.NumberColumn(format="localized")},
                     hide_index=True, width="stretch")
        st.caption("Women reached are credited to the nearest chosen site. The Map page shows the sites in orange.")
    else:
        st.warning("No upgrade within this budget brings more women within two hours.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Plan (JSON)", json.dumps(res, indent=2), file_name=f"{res['run_id']}.json", mime="application/json")
    if p["sites"]:
        d2.download_button("Sites (CSV)", pd.DataFrame(p["sites"]).to_csv(index=False), file_name=f"{res['run_id']}_sites.csv")
    if d3.button("Build the brief for this plan"):
        with st.spinner("Drawing the brief"):
            out = brief.build(state, scenario, res["run_id"])
        st.session_state["brief_pdf"] = out["pdf"]
    if st.session_state.get("brief_pdf"):
        with open(st.session_state["brief_pdf"], "rb") as f:
            d3.download_button("Download the brief (PDF)", f.read(), file_name=f"reach120_{res['run_id']}.pdf", mime="application/pdf")

st.subheader("What each budget buys")
if curve is None or curve.empty:
    st.info("No budget curve yet: run `reach120 plan --state " + state + "`.")
else:
    long = curve.melt(id_vars=["budget_naira"], value_vars=[k for k, _ in SERIES], var_name="key", value_name="women")
    long["Series"] = long["key"].map(dict(SERIES))
    long["Budget (₦ billion)"] = long["budget_naira"] / 1e9
    domain = [label for _, label in SERIES]
    colour = alt.Color("Series:N", scale=alt.Scale(domain=domain, range=appdata.series_colours()),
                       legend=alt.Legend(orient="top", title=None))
    x = alt.X("Budget (₦ billion):Q", title="Budget (₦ billion" + (", placeholder costs)" if res and res["costs_are_placeholders"] else ")"))
    y = alt.Y("women:Q", title="More women within two hours")
    hover = alt.selection_point(fields=["Budget (₦ billion)"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long).encode(x=x, y=y, color=colour)
    # The recommended plan is drawn last, so it is never hidden where it matches the greedy ranking.
    lines = (base.transform_filter(alt.datum.Series != "Recommended plan").mark_line(interpolate="step-after", strokeWidth=2)
             + base.transform_filter(alt.datum.Series == "Recommended plan").mark_line(interpolate="step-after", strokeWidth=2))
    points = base.mark_point(size=70, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Series:N"), alt.Tooltip("Budget (₦ billion):Q", format=",.2f"), alt.Tooltip("women:Q", format=",.0f")]
    ).add_params(hover)
    rule = alt.Chart(long).mark_rule(color=display.MUTED).encode(x=x).transform_filter(hover)
    # Direct labels at the right end of the curves, spaced so they never collide.
    end = curve.iloc[-1]
    span = max(float(curve[[k for k, _ in SERIES]].to_numpy().max()), 1.0)
    pos = display.spread_labels({label: float(end[k]) for k, label in SERIES}, 0.075 * span)
    tag_df = pd.DataFrame({"Budget (₦ billion)": float(end["budget_naira"]) / 1e9, "women": list(pos.values()), "Series": list(pos)})
    ink = "#c3c2b7" if appdata.is_dark() else display.INK_SECONDARY
    tags = alt.Chart(tag_df).mark_text(align="right", dx=-4, dy=-8, fontSize=11, color=ink).encode(x=x, y=y, text="Series:N")
    marker = alt.Chart(pd.DataFrame({"Budget (₦ billion)": [budget_m / 1e3]})).mark_rule(
        strokeDash=[3, 3], color=display.MUTED).encode(x=x)
    st.altair_chart((lines + rule + points + tags + marker).properties(height=340), width="stretch")
    if st.toggle("Show as a table"):
        st.dataframe(curve.rename(columns={"budget_naira": "Budget (₦)", "exact": "Recommended plan", "greedy": "Greedy",
                                           "straight_line": "Straight-line", "sites_exact": "Sites", "exact_status": "Solver"}),
                     hide_index=True, width="stretch")
    st.caption("The recommended plan is solved exactly (OR-Tools CP-SAT). The best set at a larger budget is not always the "
               "smaller budget's set plus more sites, which is why the greedy list can fall behind.")
