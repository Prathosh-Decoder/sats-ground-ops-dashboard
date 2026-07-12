"""
pages/1_Overview.py
Overview page — KPIs, delay distribution, top carriers, aircraft body type analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.loader       import load_data, render_date_filters, merge_lw_features, load_lw_data
from utils.style        import inject_css, kpi_card, chart_template, chart_fc
from utils.insights     import insight_card, insight_strip, compute_overview_stats
from utils.crossfilter  import init_xf, apply_xf, render_xf_bar, handle_selection, get_xf, bar_colors, pie_pull

st.set_page_config(page_title="Overview | SATS", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()

COLOR_MAP = {"On-Time": "#00d977", "Acceptable": "#ffc107", "Delayed": "#ff4757"}
TEMPLATE  = chart_template()
FC        = chart_fc()

# ─── Data ────────────────────────────────────────────────────────────────────
df    = load_data()
df    = render_date_filters(df, page_key="overview")
df    = merge_lw_features(df)
init_xf()
df    = apply_xf(df)
valid = df.dropna(subset=["Target_Departure_Delay_Class"])
valid = valid[valid["Target_Departure_Delay_Class"] != "nan"]

_, lw_daily = load_lw_data()
lw_has_data = len(lw_daily) > 0

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px">
  <div style="font-size:1.6rem;font-weight:800;color:#dde8ff;letter-spacing:-0.5px">📊 Overview</div>
  <div style="font-size:0.78rem;color:#6b7fa3;margin-top:2px">
    High-level performance metrics across all departure flights.
  </div>
</div>
""", unsafe_allow_html=True)
render_xf_bar()

# ─── KPI Row ─────────────────────────────────────────────────────────────────
total     = len(valid)

if total == 0:
    st.warning("No flights match the current filters. Please adjust your filter selections.")
    st.stop()

n_ontime  = (valid["Target_Departure_Delay_Class"] == "On-Time").sum()
n_accept  = (valid["Target_Departure_Delay_Class"] == "Acceptable").sum()
n_delayed = (valid["Target_Departure_Delay_Class"] == "Delayed").sum()

on_time_pct = n_ontime / total * 100
delay_pct   = n_delayed / total * 100

if "Target_Departure_Delay_mins" in df.columns:
    delayed_rows  = valid[valid["Target_Departure_Delay_Class"] == "Delayed"]
    delay_clean   = delayed_rows["Target_Departure_Delay_mins"].clip(upper=120)
    avg_delay_min = delay_clean.mean()
    median_delay  = delay_clean.median()
else:
    avg_delay_min = median_delay = np.nan

k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi_card("Total Flights", f"{total:,}",
    f"{n_ontime:,} on-time · {n_delayed:,} delayed",
    "#4d9fff", "rgba(77,159,255,0.4)"), unsafe_allow_html=True)
k2.markdown(kpi_card("On-Time Rate", f"{on_time_pct:.1f}%",
    f"{n_ontime:,} flights within schedule",
    "#00d977", "rgba(0,217,119,0.4)"), unsafe_allow_html=True)
k3.markdown(kpi_card("Delayed Rate", f"{delay_pct:.1f}%",
    f"{n_delayed:,} flights > 4 min late",
    "#ff4757", "rgba(255,71,87,0.4)"), unsafe_allow_html=True)
k4.markdown(kpi_card("Avg Delay", f"{avg_delay_min:.1f} min" if not np.isnan(avg_delay_min) else "N/A",
    f"median {median_delay:.0f} min · capped at 120 min",
    "#ffc107", "rgba(255,193,7,0.4)"), unsafe_allow_html=True)

# ─── Lightning Warning KPI strip ─────────────────────────────────────────────
if lw_has_data and "LW_Day_Had_Warning" in valid.columns:
    lw_flights   = valid[valid["LW_Day_Had_Warning"] == 1]
    lw_gw        = valid[valid["LW_In_Ground_Window"] == 1] if "LW_In_Ground_Window" in valid.columns else pd.DataFrame()
    lw_day_delay = (lw_flights["Target_Departure_Delay_Class"] == "Delayed").mean() * 100 if len(lw_flights) else 0
    all_delay    = (valid["Target_Departure_Delay_Class"] == "Delayed").mean() * 100
    delta_pp     = lw_day_delay - all_delay

    lw1, lw2, lw3, lw4 = st.columns(4)
    lw1.markdown(kpi_card("⚡ LW Days (Flights)",
        f"{len(lw_flights):,}",
        f"{valid['LW_Day_Had_Warning'].mean()*100:.0f}% of filtered flights",
        "#f1c40f", "rgba(241,196,15,0.35)"), unsafe_allow_html=True)
    lw2.markdown(kpi_card("⚡ LW in Ground Window",
        f"{len(lw_gw):,}",
        "LW directly overlapped ground-handling",
        "#e67e22", "rgba(230,126,34,0.35)"), unsafe_allow_html=True)
    lw3.markdown(kpi_card("⚡ Delay Rate on LW Days",
        f"{lw_day_delay:.1f}%",
        f"{delta_pp:+.1f} pp vs overall {all_delay:.1f}%",
        "#e74c3c" if delta_pp > 5 else "#f1c40f",
        "rgba(231,76,60,0.35)" if delta_pp > 5 else "rgba(241,196,15,0.35)"), unsafe_allow_html=True)
    lw4.markdown(kpi_card("⚡ Total LW Mins (avg/day)",
        f"{valid[valid['LW_Day_Had_Warning']==1]['Total_LW_Mins_On_Date'].mean():.0f} min" if len(lw_flights) else "—",
        "Average LW coverage on days with warnings",
        "#9b59b6", "rgba(155,89,182,0.35)"), unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── Page Insight ─────────────────────────────────────────────────────────────
ov = compute_overview_stats(valid)
if ov:
    sev = "red" if ov["delay_pct"] > 40 else ("amber" if ov["delay_pct"] > 25 else "green")
    carrier_line = (f" The worst carrier is **{ov['worst_c']}** at {ov['worst_r']:.0f}% delayed"
                    f" — best is **{ov['best_c']}** at {ov['best_r']:.0f}%."
                    if ov["worst_c"] != "N/A" else "")
    insight_card(
        problem=f"{ov['delay_pct']:.1f}% of all departures are late ({n_delayed:,} flights).{carrier_line}",
        impact=(f"Each delayed flight means missed connections, crew schedule disruptions, "
                f"and passenger complaints — with an average delay of {avg_delay_min:.0f} min when it happens."),
        action=(f"Start with carrier **{ov['worst_c']}**: they are "
                f"{ov['worst_r'] - ov['delay_pct']:.0f} pp above the fleet average. "
                f"Use the **Activity Analysis** page to find their slowest ground operation." if ov["worst_c"] != "N/A"
                else "Use the Activity Analysis page to identify the slowest ground operation."),
        icon="📊", severity=sev,
    )

st.divider()

# ─── Row 2: Donut + Top Carriers ─────────────────────────────────────────────
c_left, c_right = st.columns([1, 2])

with c_left:
    st.markdown("#### Delay Class Breakdown")
    st.caption("Click a slice to cross-filter all charts.")
    counts  = valid["Target_Departure_Delay_Class"].value_counts()
    ordered = ["On-Time", "Acceptable", "Delayed"]
    vals    = [counts.get(c, 0) for c in ordered]
    colors  = [COLOR_MAP[c] for c in ordered]

    _active_cls = get_xf("delay_class")
    _pull = pie_pull(ordered, _active_cls) or [0] * len(ordered)

    donut = go.Figure(go.Pie(
        labels=ordered, values=vals,
        hole=0.55,
        pull=_pull,
        marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
        textinfo="percent+label",
        textfont=dict(size=13),
    ))
    donut.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        height=320,
        annotations=[dict(
            text=f"<b>{total:,}</b><br>flights",
            x=0.5, y=0.5,
            font=dict(size=14, color=FC),
            showarrow=False,
        )],
    )
    _donut_evt = st.plotly_chart(donut, use_container_width=True,
                                 on_select="rerun", key="xf_overview_donut")
    handle_selection(_donut_evt, "delay_class", axis="label")

with c_right:
    st.markdown("#### Top 15 Carriers by Delay Rate")
    st.caption("Click a bar to cross-filter all charts.")
    if ov and ov["worst_c"] != "N/A":
        gap = ov["worst_r"] - ov["best_r"]
        insight_strip(
            f"⚡ Gap between worst ({ov['worst_c']} {ov['worst_r']:.0f}%) and best ({ov['best_c']} {ov['best_r']:.0f}%) carrier is <b>{gap:.0f} percentage points</b>. "
            f"Closing half that gap on {ov['worst_c']} alone would save ~{int(n_delayed * (ov['worst_r'] - ov['delay_pct']) / ov['worst_r'] * 0.5):,} delays.",
            severity="amber",
        )
    if "identification_carrierCode" in valid.columns:
        carrier_stats = (
            valid.groupby("identification_carrierCode")
            .agg(
                total=("Target_Departure_Delay_Class", "count"),
                delayed=("Target_Departure_Delay_Class", lambda x: (x == "Delayed").sum()),
            )
            .reset_index()
        )
        carrier_stats = carrier_stats[carrier_stats["total"] >= 20]
        carrier_stats["delay_rate"] = carrier_stats["delayed"] / carrier_stats["total"] * 100
        top15 = carrier_stats.nlargest(15, "delay_rate").sort_values("delay_rate")

        _active_carrier = get_xf("carrier")
        _bar_c = bar_colors(top15["identification_carrierCode"], _active_carrier)
        _marker = dict(color=_bar_c) if _bar_c else dict(
            color=top15["delay_rate"],
            colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
            showscale=False,
        )

        bar_carrier = go.Figure()
        bar_carrier.add_trace(go.Bar(
            y=top15["identification_carrierCode"],
            x=top15["delay_rate"],
            orientation="h",
            marker=_marker,
            text=[f"{v:.0f}%  ({n:,} flights)" for v, n in zip(top15["delay_rate"], top15["total"])],
            textposition="outside",
            textfont=dict(size=11),
        ))
        bar_carrier.update_layout(
            template=TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=80),
            xaxis=dict(title="Delay Rate (%)", range=[0, top15["delay_rate"].max() * 1.25]),
            yaxis=dict(title=""),
            height=380,
        )
        _carrier_evt = st.plotly_chart(bar_carrier, use_container_width=True,
                                       on_select="rerun", key="xf_overview_carrier")
        handle_selection(_carrier_evt, "carrier", axis="y")
    else:
        st.info("Carrier code column not found in data.")

st.divider()

# ─── Row 3: Delay Histogram ───────────────────────────────────────────────────
st.markdown("#### Departure Delay Distribution")
st.caption("How many minutes late do flights typically depart? Negative = early.")

if "Target_Departure_Delay_mins" in valid.columns:
    delay_vals = valid["Target_Departure_Delay_mins"].dropna()
    delay_clipped = delay_vals.clip(-30, 90)

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=delay_clipped,
        nbinsx=60,
        name="All Flights",
        marker_color="#1a73e8",
        opacity=0.85,
    ))
    hist_fig.add_vline(x=0, line_color="#2ecc71", line_dash="dash", line_width=2,
                       annotation_text="On-Time (0 min)",
                       annotation_position="top left",
                       annotation_font_color="#2ecc71",
                       annotation_font_size=11)
    hist_fig.add_vline(x=4, line_color="#f39c12", line_dash="dash", line_width=2,
                       annotation_text="Delayed (+4 min)",
                       annotation_position="top right",
                       annotation_font_color="#f39c12",
                       annotation_font_size=11)
    hist_fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Departure Delay (minutes)",
        yaxis_title="Number of Flights",
        height=320,
        margin=dict(t=20, b=40, l=40, r=20),
        bargap=0.02,
        showlegend=False,
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Stats row
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Minimum",  f"{delay_vals.min():.0f} min")
    s2.metric("Median",   f"{delay_vals.median():.1f} min")
    s3.metric("Mean",     f"{delay_vals.mean():.1f} min")
    s4.metric("75th pct", f"{delay_vals.quantile(0.75):.1f} min")
    s5.metric("Maximum",  f"{delay_vals.max():.0f} min")

st.divider()

# ─── Row 4: Box Plot by Aircraft Body Type ────────────────────────────────────
st.markdown("#### Delay by Aircraft Body Type")
if ov and ov["wb_rate"] > 0 and ov["nb_rate"] > 0:
    diff = ov["wb_rate"] - ov["nb_rate"]
    worse = "Widebody" if diff > 0 else "Narrowbody"
    insight_strip(
        f"{'⚠️' if abs(diff) > 5 else 'ℹ️'} <b>{worse}</b> flights have a {abs(diff):.1f} pp higher delay rate "
        f"({ov['wb_rate']:.0f}% vs {ov['nb_rate']:.0f}%). "
        f"{'Complex turnarounds on widebodies need more buffer time in the schedule.' if diff > 5 else 'Delay rates are similar across both aircraft sizes.'}",
        severity="amber" if abs(diff) > 5 else "blue",
    )
else:
    st.caption("Widebody aircraft tend to have more complex turnarounds — does it show in delays?")

if "aircraft_bodyType" in valid.columns and "Target_Departure_Delay_mins" in valid.columns:
    body_df = valid.dropna(subset=["aircraft_bodyType", "Target_Departure_Delay_mins"])
    body_df = body_df[body_df["aircraft_bodyType"].isin(["Widebody", "Narrowbody", "W", "N", "NB", "WB"])]

    # Normalise body type labels
    body_map = {"W": "Widebody", "WB": "Widebody", "N": "Narrowbody", "NB": "Narrowbody"}
    body_df["Body Type"] = body_df["aircraft_bodyType"].replace(body_map)
    body_df = body_df[body_df["Body Type"].isin(["Widebody", "Narrowbody"])]

    if not body_df.empty:
        box_fig = go.Figure()
        body_colors = {"Widebody": "#1a73e8", "Narrowbody": "#9c27b0"}

        for btype in ["Widebody", "Narrowbody"]:
            subset = body_df[body_df["Body Type"] == btype]["Target_Departure_Delay_mins"].clip(-30, 90)
            if subset.empty:
                continue
            box_fig.add_trace(go.Box(
                y=subset,
                name=btype,
                marker_color=body_colors[btype],
                boxpoints="outliers",
                line=dict(width=2),
                notched=True,
            ))
        box_fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Departure Delay (minutes)",
            xaxis_title="Aircraft Body Type",
            height=340,
            margin=dict(t=20, b=40, l=40, r=20),
            showlegend=False,
        )
        st.plotly_chart(box_fig, use_container_width=True)
    else:
        st.info("Not enough data to plot by aircraft body type.")
else:
    st.info("Aircraft body type column not found in data.")
