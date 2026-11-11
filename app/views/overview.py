import pandas as pd
import streamlit as st

import appdata
from reach120 import brief, scenarios, tables
from reach120.store import StateStore

state = appdata.current_state()
summary = appdata.summary(state)
scenario, mode = summary["default_run"].split("/")
run = summary["runs"][summary["default_run"]]
walk = summary["runs"].get(f"{scenario}/walk")
plan = appdata.headline_plan(state)

st.title(f"Reach120 · {state}")
st.caption(f"Women aged 15-49 and their travel time to a hospital that can likely perform an emergency caesarean section. "
           f"Default: {scenarios.label(scenario).lower()} definition, chosen by the capability scorecard.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Beyond 2 hours by vehicle", appdata.pct(run["share_beyond_target"], short=True),
          help=f"{run['beyond_target']:,.0f} women more than two hours away with a vehicle on passable roads")
if walk:
    c2.metric("Beyond 2 hours on foot", appdata.pct(walk["share_beyond_target"], short=True),
              help=f"{walk['beyond_target']:,.0f} women more than two hours away on foot")
lo, hi = summary["range_share_beyond_target"]
c3.metric("Range, all runs", f"{appdata.pct(lo, short=True)} to {appdata.pct(hi, short=True)}",
          help="Across every capability definition and both transport assumptions")
c4.metric("Capable hospitals used", f"{run['sources']}")

for sentence in brief.sentences(summary, plan, plan["scenario"] if plan else scenario):
    st.write(sentence)

left, right = st.columns([3, 2])
with left:
    table_mode = plan["mode"] if plan else mode
    table_scenario = plan["scenario"] if plan else scenario
    st.subheader(f"Worst-served LGAs {'on foot' if table_mode == 'walk' else 'by vehicle'}")
    lgas = tables.lga_table(appdata.coverage(state, table_scenario, table_mode)).head(10)
    lgas[["women", "beyond_target"]] = lgas[["women", "beyond_target"]].round().astype(int)
    if not lgas.empty:
        st.dataframe(
            lgas[["lga", "women", "beyond_target", "share_beyond_target"]].rename(columns={
                "lga": "LGA", "women": "Women 15-49", "beyond_target": "More than 2 hours away",
                "share_beyond_target": "Share"}),
            column_config={"Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                           "Women 15-49": st.column_config.NumberColumn(format="localized"),
                           "More than 2 hours away": st.column_config.NumberColumn(format="localized")},
            hide_index=True, width="stretch")
with right:
    st.subheader("Every run")
    rows = [{"Definition": r["label"], "Transport": "Vehicle" if r["mode"] == "mixed" else "On foot",
             "Beyond 2 h": r["share_beyond_target"]} for r in summary["runs"].values()]
    st.dataframe(pd.DataFrame(rows), column_config={"Beyond 2 h": st.column_config.ProgressColumn(
        format="percent", min_value=0, max_value=1)}, hide_index=True, width="stretch")

pdf = StateStore(state).brief_pdf
if pdf.exists():
    st.download_button("Download the one-page brief (PDF)", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf")
st.caption("Upgrade costs are placeholders. See Method and limits for how every number is made and what it leaves out.")
