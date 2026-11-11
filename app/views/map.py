import pydeck as pdk
import streamlit as st
from pydeck.types import String

import appdata
from reach120 import display

state = appdata.current_state()
summary = appdata.summary(state)
default_s, default_m = summary["default_run"].split("/")
opts = appdata.scenario_options()

st.title("Map")
c1, c2, c3 = st.columns([2, 2, 3])
scenario = c1.selectbox("Capability definition", list(opts), index=list(opts).index(default_s), format_func=opts.get)
mode = c2.radio("Transport", ["mixed", "walk"], format_func=appdata.mode_label, horizontal=True)
layer = c3.radio("Show", ["Travel time", "Wards: share more than two hours away"], horizontal=True)
key = f"{scenario}/{mode}"
tag = key.replace("/", "_")
run = summary["runs"][key]
st.caption(f"{appdata.pct(run['share_beyond_target'])} of {run['women']:,.0f} women aged 15-49 are more than two hours away "
           f"under this definition. Hover over wards and facilities for details.")

meta = appdata.display_meta(state)["runs"][key]
w, s, e, n = meta["bounds"]
layers = []
if layer == "Travel time":
    # String(): pydeck would otherwise read the data URL as a JavaScript expression and fail on its ":".
    layers.append(pdk.Layer("BitmapLayer", image=String(appdata.png_url(state, scenario, mode)), bounds=[w, s, e, n],
                            opacity=0.9))

polys = []
for p in appdata.ward_polygons(state):
    share = p.get(f"share_{tag}")
    beyond = p.get(f"beyond_{tag}") or 0
    polys.append({"polygon": p["polygon"],
                  "fill": display.share_colour(share) if layer != "Travel time" else [0, 0, 0, 0],
                  "tip": f"<b>{p['ward']}</b>, {p['lga']}<br/>"
                         + (f"{100 * share:.1f}% of women more than two hours away ({beyond:,.0f})" if share is not None else "no women counted")})
layers.append(pdk.Layer("PolygonLayer", polys, get_polygon="polygon", get_fill_color="fill", get_line_color=[255, 255, 255, 140],
                        line_width_min_pixels=0.5, stroked=True, filled=True, pickable=True))

fac = appdata.facilities(state)
cap = fac[fac[f"cap_{scenario}"] & fac["in_state"]].copy()
cap["tip"] = ("<b>" + cap["facility_name"] + "</b><br/>" + cap["facility_type"] + " · " + cap["facility_ownership"]
              + "<br/>Counted as capable")
layers.append(pdk.Layer("ScatterplotLayer", cap[["longitude", "latitude", "tip"]], get_position=["longitude", "latitude"],
                        get_fill_color=list(display.hex_rgb(display.INK)), get_line_color=[255, 255, 255], stroked=True,
                        line_width_min_pixels=1.5, radius_min_pixels=4, radius_max_pixels=9, get_radius=500, pickable=True))

plan = st.session_state.get(f"plan_{state}")
if not plan or plan["mode"] != mode or plan["scenario"] != scenario:
    plan = appdata.default_plan(state, scenario, mode) if scenario != "base_public" else None
if plan and plan["plan"]["sites"]:
    sites = [{"longitude": p["longitude"], "latitude": p["latitude"],
              "tip": f"<b>{p['facility_name']}</b><br/>Recommended upgrade ({p['tier']})<br/>"
                     f"{p['women_newly_within_target']:,.0f} more women within two hours"} for p in plan["plan"]["sites"]]
    layers.append(pdk.Layer("ScatterplotLayer", sites, get_position=["longitude", "latitude"],
                            get_fill_color=list(display.hex_rgb(display.UPGRADE)), get_line_color=[255, 255, 255], stroked=True,
                            line_width_min_pixels=2, radius_min_pixels=7, radius_max_pixels=14, get_radius=900, pickable=True))

view = pdk.ViewState(longitude=(w + e) / 2, latitude=(s + n) / 2, zoom=7.4)
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="light",
                         tooltip={"html": "{tip}", "style": {"fontSize": "12px"}}), height=640)

if layer == "Travel time":
    items = appdata.display_meta(state)["legend"]
else:
    steps = display.SEQUENTIAL
    items = [[f"{int(100 * i / len(steps))}-{int(100 * (i + 1) / len(steps))}% more than two hours away", c] for i, c in enumerate(steps)]
legend = " ".join(f"<span style='display:inline-flex;align-items:center;margin-right:14px'><span style='width:14px;height:14px;"
                  f"background:{c};border-radius:3px;margin-right:6px;display:inline-block'></span>{label}</span>" for label, c in items)
legend += (f" <span style='display:inline-flex;align-items:center;margin-right:14px'><span style='width:10px;height:10px;"
           f"background:{display.INK};border:2px solid #fff;border-radius:50%;margin-right:6px'></span>Hospital counted as capable</span>")
if plan and plan["plan"]["sites"]:
    legend += (f" <span style='display:inline-flex;align-items:center'><span style='width:12px;height:12px;background:{display.UPGRADE};"
               f"border:2px solid #fff;border-radius:50%;margin-right:6px'></span>Recommended upgrade ({plan['run_id']})</span>")
st.markdown(f"<div style='font-size:13px;line-height:2'>{legend}</div>", unsafe_allow_html=True)
