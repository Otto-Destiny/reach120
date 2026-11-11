"""Reach120 app. Run with `streamlit run app/streamlit_app.py` from the project folder."""
import streamlit as st

st.set_page_config(page_title="Reach120", page_icon="🏥", layout="wide")

import appdata  # noqa: E402

states = appdata.states()
if not states:
    st.title("Reach120")
    st.error("No results yet. Run the pipeline first: `reach120 all --state Kano`.")
    st.stop()

with st.sidebar:
    if st.session_state.get("state") not in states:
        st.session_state["state"] = states[0]
    st.selectbox("State", states, key="state")
    st.caption("Two-hour access to emergency obstetric care, and hospital upgrades ranked within a budget.")

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/map.py", title="Map", icon=":material/map:"),
    st.Page("views/planner.py", title="Planner", icon=":material/tune:"),
    st.Page("views/ask.py", title="Ask", icon=":material/chat:"),
    st.Page("views/method.py", title="Method and limits", icon=":material/info:"),
]
st.navigation(pages).run()
