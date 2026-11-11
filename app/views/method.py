import re

import pandas as pd
import streamlit as st

import appdata
from reach120 import scenarios
from reach120.store import StateStore

state = appdata.current_state()
summary = appdata.summary(state)
st.title("Method and limits")

st.header("What the numbers mean")
st.markdown(f"""
- **Women at risk:** women aged 15-49 from the WorldPop/GRID3 population grid (about 92 m), summed to wards. Nothing is
  reported below ward level, because single-cell values are fractional estimates.
- **Travel time:** the least-cost route across GRID3's 30 m friction surface, which holds minutes per metre. Vehicles
  move at road speeds on passable roads and people walk elsewhere ("vehicles on passable roads"), or everyone walks
  ("walking only"). Rivers are barriers. Each 30 m result is averaged into the population cell around it.
- **Two hours:** {summary['target_min']} minutes, the time within which most deaths from bleeding, eclampsia and
  obstructed labour can be prevented by a caesarean section.
- **Hospitals that can likely operate:** the data has no surgery field, so facility type stands in for capability, under
  four definitions:
""")
st.dataframe(pd.DataFrame([{"Definition": scenarios.label(s), "Counts as capable": scenarios.description(s),
                            "Facilities used here": summary["runs"].get(f"{s}/mixed", {}).get("sources")}
                           for s in scenarios.names()]), hide_index=True, width="stretch")
st.markdown("Facilities marked Not-Functional or Closed are left out; facilities with unknown status are kept.")

st.header("How upgrades are chosen")
st.markdown("""
Women more than two hours away are grouped into demand points of about 460 m. Each candidate facility (a hospital or
other secondary facility not yet counted as capable, or the best-placed public primary health centre in each LGA) gets
one search on the 30 m grid that records which demand points it would reach within two hours. The search skips ground
that an existing hospital already serves better. This is exact: it follows from the triangle inequality and is checked
against unpruned searches on every run.

The planner then solves a **maximal covering problem with a budget**: pick the upgrades that bring the most women within
two hours without exceeding the budget. It is solved exactly with OR-Tools CP-SAT, and every result is compared with a
greedy ranking (best women per naira first) and a straight-line ranking that ignores roads and rivers.

**Upgrade costs are placeholders** until real figures replace them in `config/costs.yaml` or on the Planner page.
""")

sc = appdata.scorecard()
st.header("Is facility type a fair stand-in?")
if sc:
    st.markdown(f"OnTIME verified {sc['verified_total']:,} facilities with emergency caesarean capacity in 15 city regions "
                f"({100 * sc['verified_private_share']:.0f}% private). {sc['verified_matched']:,} were matched to GRID3 facilities.")
    st.dataframe(pd.DataFrame(sc["scenarios"])[["label", "catch_rate", "capable_in_study_area", "false_inclusion_rate", "balance_f1"]]
                 .rename(columns={"label": "Definition", "catch_rate": "Catch rate", "capable_in_study_area": "Capable in study area",
                                  "false_inclusion_rate": "Not on the verified list", "balance_f1": "Balance (F1)"}),
                 column_config={"Catch rate": st.column_config.NumberColumn(format="percent"),
                                "Not on the verified list": st.column_config.NumberColumn(format="percent"),
                                "Balance (F1)": st.column_config.NumberColumn(format="%.2f")},
                 hide_index=True, width="stretch")
    st.caption(sc["note"])
else:
    st.info("Not scored yet: run `reach120 scorecard`.")

report = appdata.validation_report(state)
st.header("Validation")
if report:
    body = re.sub(r"^# .*\n", "", report)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", body)
    st.markdown(body)
    png = StateStore(state).validation("ontime_scatter.png")
    if png.exists():
        st.image(str(png), caption="Each dot is one city grid cell. Dots above the line: the model is slower than Google.")
else:
    st.info(f"Not run yet: `reach120 validate --state {state}`.")

st.header("Limits")
st.markdown("""
- Capability is inferred, not observed, outside the 15 OnTIME cities.
- Functional status is unknown for many facilities, and some facility records have no coordinates.
- The friction surface assumes dry-season roads and no traffic; it is not yet fully validated by government.
- Travel time is not the only barrier. Cost, referral delays, staffing and blood supply are outside the model.
- Upgrade costs are placeholders.

**Data and licences:** GRID3 Nigeria health facilities v4.0, operational wards v4.0 and travel-time friction surface v1.0
(CIESIN, Columbia University; CC BY 4.0 and CC BY-SA 4.0); WorldPop/GRID3 population v3.0 (CC BY 4.0); OnTIME verified
facilities and travel times (CC BY 4.0). Maps derived from CC BY-SA data are shared under CC BY-SA 4.0.
""")
