"""
pages/3_Activity_Analysis.py
Activity Analysis — BU overview → BU drill-down → Activity drill-down.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.loader       import load_data, get_activity_stats, render_date_filters
from utils.style        import inject_css, chart_template, chart_fc, card_bg, card_text, card_sub
from utils.insights     import (insight_card, insight_strip, compute_activity_stats,
                                no_data_card, no_data_info, no_data_metric, has_enough_data)
from utils.cascade      import NODES as _CASCADE_NODES, DEPS as _CASCADE_DEPS
from utils.crossfilter  import init_xf, apply_xf, render_xf_bar

st.set_page_config(page_title="Activity Analysis | SATS", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")
inject_css()

TEMPLATE = chart_template()
FC       = chart_fc()

BU_INFO = {
    "techramp":    {"label": "Tech Ramp",     "color": "#34495e", "icon": "🛠️"},
    "ramp":        {"label": "Ramp",          "color": "#1a73e8", "icon": "🔧"},
    "pax":         {"label": "Passenger Svc", "color": "#9c27b0", "icon": "🧑‍✈️"},
    "aic":         {"label": "AIC Cleaning",  "color": "#00bcd4", "icon": "🧹"},
    "cargo":       {"label": "Cargo",         "color": "#795548", "icon": "📦"},
    "security":    {"label": "Security",      "color": "#f44336", "icon": "🔒"},
    "cabinsvc":    {"label": "Cabin Service", "color": "#4caf50", "icon": "🪑"},
    "loadcontrol": {"label": "Load Control",  "color": "#607d8b", "icon": "⚖️"},
    "baggage":     {"label": "Baggage",       "color": "#ff9800", "icon": "🧳"},
    "tpo":         {"label": "TPO / Towing",  "color": "#e91e63", "icon": "🚛"},
}

# ─── Session state ────────────────────────────────────────────────────────────
for key in ("selected_bu", "selected_activity"):
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Data ────────────────────────────────────────────────────────────────────
df    = load_data()
df    = render_date_filters(df, page_key="activity")
init_xf()
df    = apply_xf(df)
valid = df.dropna(subset=["Target_Departure_Delay_Class"])
valid = valid[valid["Target_Departure_Delay_Class"] != "nan"]
stats = get_activity_stats(df)
stats_filtered = stats[stats["count"] >= 50].copy()
stats_filtered["team_clean"] = stats_filtered["team"].str.lower().str.strip()
# Tech Ramp activities are stored under ramp_* in the data but belong to the
# Tech Ramp team (see utils/cascade.py). Remap so grouping matches the flowchart.
_TECHRAMP_RAW = {"milestone_ramp_thumbsUp"}
stats_filtered.loc[stats_filtered["raw_name"].isin(_TECHRAMP_RAW), "team_clean"] = "techramp"

# ─── Breadcrumb helpers ───────────────────────────────────────────────────────
def breadcrumb():
    parts = ["🔧 Activity Analysis"]
    if st.session_state["selected_bu"]:
        info = BU_INFO.get(st.session_state["selected_bu"], {})
        parts.append(f"{info.get('icon','')} {info.get('label', st.session_state['selected_bu'])}")
    if st.session_state["selected_activity"]:
        parts.append(st.session_state["selected_activity"])
    st.markdown(" › ".join(f"**{p}**" for p in parts))
    render_xf_bar()

def back_to_overview():
    st.session_state["selected_bu"] = None
    st.session_state["selected_activity"] = None

def back_to_bu():
    st.session_state["selected_activity"] = None

# ─── BU stats aggregation ────────────────────────────────────────────────────
# A BU-level number is only reliable once the activities backing it have
# recorded enough flights combined — number of distinct activities doesn't
# matter (a BU with 2 well-recorded activities is fine; one with 5 barely-
# recorded ones isn't).
BU_MIN_TOTAL_FLIGHTS = 500

def get_bu_stats(stats_df):
    rows = []
    for bu, info in BU_INFO.items():
        bu_activities = stats_df[stats_df["team_clean"] == bu]
        if bu_activities.empty:
            continue
        counts = bu_activities["count"]
        total_count = int(counts.sum())
        # Flight-count-weighted average — an activity with more recorded
        # flights should count for more than one with barely 50, unlike a
        # flat .mean() across activities.
        avg_delay     = float((bu_activities["avg_delay_mins"] * counts).sum() / total_count)
        avg_late_rate = float((bu_activities["late_rate"]      * counts).sum() / total_count)
        rows.append({
            "bu":           bu,
            "label":        info["label"],
            "icon":         info["icon"],
            "color":        info["color"],
            "n_activities": len(bu_activities),
            "total_count":  total_count,
            "avg_delay":    avg_delay,
            "avg_late_rate":avg_late_rate,
            "worst_activity": bu_activities.sort_values("avg_delay_mins", ascending=False).iloc[0]["activity"],
        })
    return pd.DataFrame(rows)

bu_stats = get_bu_stats(stats_filtered)


# ─── Delay type decomposition helper ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_delay_types(_df, raw_name):
    """
    Notebook-faithful decomposition (from prepare_data.py compute_activity_analysis):

      For INTERVAL activities (those with _start / _end milestones):
        finish_delay     = _analysis_Delay_mins  = actual_end - planned_end
        duration_overrun = _analysis_ActualDuration_mins - _analysis_PlannedDuration_mins
        start_delay      = finish_delay - duration_overrun  (algebraic identity)

      For POINT milestones:
        _analysis_Delay_mins = delay_mins if late else 0  (always >= 0)
        No duration columns exist → has_duration = False

    Uses SEPARATE sample sets so sparse duration data doesn't hide finish-delay stats:
      n_finish : rows where delay_col is non-null (used for percentiles, % finished late)
      n_decomp : rows where all 3 cols are non-null (used for start/duration decomposition)

    Threshold = 0 throughout (per notebook's strict comparison: actual > planned).
    """
    delay_col    = raw_name + "_analysis_Delay_mins"
    dur_col      = raw_name + "_analysis_ActualDuration_mins"
    plan_dur_col = raw_name + "_analysis_PlannedDuration_mins"

    if delay_col not in _df.columns:
        return None

    # ── Finish-delay sample (most data available) ─────────────────────────────
    sub_finish = _df[delay_col].dropna().astype(float)
    if len(sub_finish) < 10:
        return None

    pcts = {k: float(sub_finish.quantile(v))
            for k, v in [("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9)]}
    pct_finished_late = float((sub_finish > 0).mean() * 100)

    result = {
        "has_duration":      False,
        "n_finish":          len(sub_finish),
        "n_decomp":          0,
        "finish_delay":      sub_finish,
        "percentiles":       pcts,
        "pct_finished_late": pct_finished_late,
    }

    # ── Duration decomposition sample (only when both duration cols exist) ────
    has_dur = dur_col in _df.columns and plan_dur_col in _df.columns
    if has_dur:
        sub_dur = _df[[delay_col, dur_col, plan_dur_col]].dropna()
        if len(sub_dur) >= 10:
            sub_dur = sub_dur.astype(float)
            finish_d  = sub_dur[delay_col]
            dur_over  = sub_dur[dur_col] - sub_dur[plan_dur_col]
            start_del = finish_d - dur_over          # algebraic identity

            # Threshold = 0: any positive value = delayed (notebook uses actual > planned)
            is_sl = start_del > 0                    # started_late flag from notebook
            is_do = dur_over  > 0                    # higher_duration flag from notebook
            # 4 mutually-exclusive categories (sum = n_decomp)
            cats = {
                "On Time":             int((~is_sl & ~is_do).sum()),
                "Late Start Only":     int((is_sl  & ~is_do).sum()),
                "Slow Execution Only": int((~is_sl & is_do ).sum()),
                "Late Start + Slow":   int((is_sl  & is_do ).sum()),
            }
            result.update({
                "has_duration":      True,
                "n_decomp":          len(sub_dur),
                "dur_overrun":       dur_over,
                "start_delay_s":     start_del,
                "avg_start_delay":   float(start_del.mean()),
                "avg_dur_overrun":   float(dur_over.mean()),
                "pct_started_late":  float(is_sl.mean() * 100),
                "pct_dur_exceeded":  float(is_do.mean() * 100),
                "cats":              cats,
            })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — Activity detail
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["selected_activity"]:
    act_name = st.session_state["selected_activity"]
    breadcrumb()

    bc1, bc2 = st.columns([1, 10])
    with bc1:
        if st.button("← Back", key="back_to_bu"):
            back_to_bu()
            st.rerun()
    st.divider()

    # Find this activity's data
    act_row = stats_filtered[stats_filtered["activity"] == act_name]
    if act_row.empty:
        st.warning("Activity data not found.")
        st.stop()
    act_row = act_row.iloc[0]
    delay_col = act_row["raw_name"] + "_analysis_Delay_mins"
    bu_key = act_row["team_clean"]
    bu_info = BU_INFO.get(bu_key, {"color": "#9e9e9e", "label": bu_key, "icon": ""})

    st.markdown(
        f"<h2 style='color:{bu_info['color']}'>"
        f"{bu_info['icon']} {act_name}</h2>",
        unsafe_allow_html=True,
    )

    # Headline PTS-overrun sentence — ground-staff-facing plain language
    if act_row["count"] >= 10 and not np.isnan(act_row["avg_delay_mins"]):
        st.markdown(
            f"<p style='font-size:1.05rem;color:{card_text()};margin-bottom:4px'>"
            f"📐 On average, this activity exceeded <b>PTS</b> "
            f"(Planned Time Standard) by <b style='color:{bu_info['color']}'>"
            f"{act_row['avg_delay_mins']:.1f} minutes</b>.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No data")

    # Extended KPI row — compute live stats from raw delay data
    _raw_s = df[delay_col].dropna() if delay_col in df.columns else pd.Series(dtype=float)
    _med_d = float(_raw_s.median())       if len(_raw_s) else float("nan")
    _std_d = float(_raw_s.std())          if len(_raw_s) else float("nan")
    _p90_d = float(_raw_s.quantile(0.90)) if len(_raw_s) else float("nan")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Delay",     f"{act_row['avg_delay_mins']:.1f} min")
    k2.metric("Median",        f"{_med_d:.1f} min")
    k3.metric("Std Deviation", f"±{_std_d:.1f} min")
    k4.metric("P90 Delay",     f"{_p90_d:.1f} min")
    k5.metric("Late Rate",     f"{act_row['late_rate']*100:.1f}%")
    k6.metric("Flights",       f"{int(act_row['count']):,}")

    st.divider()

    # Activity-level insight
    _late_pct = act_row["late_rate"] * 100
    _avg_d    = act_row["avg_delay_mins"]
    _sev      = "red" if _late_pct > 50 else ("amber" if _late_pct > 30 else "green")
    _eff = ""
    if not np.isnan(act_row.get("avg_planned_dur", float("nan"))) and act_row.get("avg_planned_dur", 0) > 0:
        _eff = (f" Planned duration is {act_row['avg_planned_dur']:.1f} min — "
                f"{'actual is running over plan.' if _avg_d > 0 else 'team is meeting the plan on time.'}")
    insight_card(
        problem=(f"**{act_name.split(': ')[-1]}** runs late on **{_late_pct:.0f}%** of flights "
                 f"with an average overrun of **{_avg_d:.1f} min**.{_eff}"),
        impact=(f"This activity runs late on 1 in every {max(1, int(100/_late_pct)) if _late_pct > 0 else '—'} flights. "
                f"Use the 'Impact on Departure Delay' chart below to see how strongly this translates "
                f"into actual departure delays." if _late_pct > 0 else
                "This activity is performing well — late rate is low."),
        action=("Check the 'Delay by Hour of Day' chart to find the peak overrun window, "
                "then investigate: Is it a staffing level issue? Equipment availability? "
                "Or a preceding activity causing a late handover?"),
        icon=bu_info.get("icon", "🔧"), severity=_sev,
    )

    # ── Pinpoint Table — exact flights where this activity ran late ──────────
    st.divider()
    st.markdown("#### 🎯 Pinpoint Table — Which Flights Were Affected")
    st.caption(
        f"Every flight where {act_name.split(': ')[-1]} ran late, worst first — "
        "use this to trace specific cases for follow-up."
    )
    if delay_col in df.columns:
        pin_df = df[df[delay_col] > 0].copy()
        if not pin_df.empty:
            if "identification_iata" in pin_df.columns:
                pin_df["Flight"] = pin_df["identification_iata"]
            if "departure_offBlock.scheduled" in pin_df.columns:
                _psched = pd.to_datetime(pin_df["departure_offBlock.scheduled"], errors="coerce", utc=True) \
                    .dt.tz_convert("Asia/Singapore")
                pin_df["Date"] = _psched.dt.strftime("%Y-%m-%d")
                pin_df["Dep Time"] = _psched.dt.strftime("%H:%M")
            if "origin_terminal" in pin_df.columns:
                pin_df["Terminal"] = pin_df["origin_terminal"].astype(str).replace(
                    {"1": "T1", "2": "T2", "3": "T3", "4": "T4"})
            if "destination_iata" in pin_df.columns:
                pin_df["Destination"] = pin_df["destination_iata"]
            pin_df["This Activity (min)"] = pin_df[delay_col].round(1)
            if "Target_Departure_Delay_mins" in pin_df.columns:
                pin_df["Dep Delay (min)"] = pin_df["Target_Departure_Delay_mins"].round(1)

            _show_cols = [c for c in ["Flight", "Date", "Dep Time", "Terminal", "Destination",
                                      "This Activity (min)", "Dep Delay (min)"] if c in pin_df.columns]
            pin_table = pin_df[_show_cols].sort_values("This Activity (min)", ascending=False)
            st.dataframe(
                pin_table.reset_index(drop=True),
                use_container_width=True, hide_index=True, height=380,
                column_config={
                    "This Activity (min)": st.column_config.NumberColumn(format="%.1f"),
                    "Dep Delay (min)":     st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption(f"{len(pin_table):,} of {int(act_row['count']):,} flights had this activity running late.")
        else:
            st.info("No data")
    else:
        st.info("No data")

    if delay_col in df.columns:
        delay_series = df[delay_col].dropna()
        delay_clipped = delay_series.clip(-60, 120)

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("#### Delay Distribution")
            hist = go.Figure()
            hist.add_trace(go.Histogram(
                x=delay_clipped, nbinsx=50,
                marker_color=bu_info["color"], opacity=0.85,
            ))
            hist.add_vline(x=0, line_dash="dash", line_color="#2ecc71", line_width=2)
            hist.update_layout(
                template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Delay (min)", yaxis_title="Flights",
                height=320, margin=dict(t=10, b=40, l=40, r=10), showlegend=False,
            )
            st.plotly_chart(hist, use_container_width=True)

        with right_col:
            st.markdown("#### Delay by Hour of Day")
            if "Hour_of_Day" in df.columns:
                hour_df  = df[["Hour_of_Day", delay_col]].dropna()
                hour_agg = hour_df.groupby("Hour_of_Day")[delay_col].agg(
                    mean="mean", std="std",
                    p90=lambda x: x.quantile(0.90), n="count"
                ).reset_index()
                # Only show error bars / P90 for hours with enough data
                MIN_N = 15
                hour_agg["std_display"] = hour_agg.apply(
                    lambda r: r["std"] if r["n"] >= MIN_N else 0.0, axis=1
                ).fillna(0)
                hour_agg["p90_display"] = hour_agg.apply(
                    lambda r: r["p90"] if r["n"] >= MIN_N else float("nan"), axis=1
                )
                _hc = bu_info["color"]
                hour_fig = go.Figure()
                hour_fig.add_trace(go.Bar(
                    x=hour_agg["Hour_of_Day"], y=hour_agg["mean"],
                    name="Avg Delay",
                    marker_color=_hc,
                    error_y=dict(type="data", array=hour_agg["std_display"],
                                 visible=True, color="rgba(180,180,180,0.40)",
                                 thickness=1.2, width=3),
                    customdata=np.stack([hour_agg["std_display"], hour_agg["n"]], axis=1),
                    hovertemplate="Hour %{x}:00<br>Avg: %{y:.1f} min<br>±1σ: %{customdata[0]:.1f} min<br>n=%{customdata[1]}<extra></extra>",
                ))
                hour_fig.add_trace(go.Scatter(
                    x=hour_agg["Hour_of_Day"], y=hour_agg["p90_display"],
                    mode="lines+markers", name="P90",
                    line=dict(color="#e74c3c", dash="dot", width=1.5),
                    marker=dict(size=4),
                    connectgaps=False,
                    hovertemplate="Hour %{x}:00<br>P90: %{y:.1f} min<extra></extra>",
                ))
                hour_fig.add_hline(y=0, line_dash="dash", line_color="#2ecc71", line_width=1)
                # Clamp y-axis so sparse-hour spikes don't dominate
                _y_max = hour_agg.loc[hour_agg["n"] >= MIN_N, "p90"].max()
                _y_max = float(_y_max) if not np.isnan(_y_max) else 30
                hour_fig.update_layout(
                    template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Hour (UTC)", dtick=3, range=[-0.5, 23.5]),
                    yaxis=dict(title="Delay (min)", range=[None, _y_max * 1.25]),
                    height=320, margin=dict(t=10, b=40, l=50, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(hour_fig, use_container_width=True)

        # Correlation with departure delay
        if "Target_Departure_Delay_mins" in df.columns:
            st.markdown("#### Impact on Departure Delay")
            corr_df = df[[delay_col, "Target_Departure_Delay_mins"]].dropna()
            corr_df = corr_df[corr_df[delay_col].between(-30, 120) &
                              corr_df["Target_Departure_Delay_mins"].between(-30, 120)]
            if len(corr_df) > 100:
                corr_val = corr_df[delay_col].corr(corr_df["Target_Departure_Delay_mins"])
                scatter = px.scatter(
                    corr_df.sample(min(2000, len(corr_df))),
                    x=delay_col, y="Target_Departure_Delay_mins",
                    trendline="ols", opacity=0.35,
                    labels={delay_col: f"{act_name} Delay (min)",
                            "Target_Departure_Delay_mins": "Departure Delay (min)"},
                    template=TEMPLATE,
                    color_discrete_sequence=[bu_info["color"]],
                )
                scatter.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=340, margin=dict(t=30, b=40, l=50, r=10),
                    title=f"Correlation with departure delay: r = {corr_val:.2f}",
                )
                st.plotly_chart(scatter, use_container_width=True)
            else:
                st.info("No data")

    # ── Percentile stats + delay type breakdown ──────────────────────────────
    st.divider()
    dt = compute_delay_types(df, act_row["raw_name"])
    if dt:
        st.markdown("#### 📐 Delay Percentiles")
        p = dt["percentiles"]
        kp1, kp2, kp3, kp4 = st.columns(4)
        kp1.metric("P50 (Median)", f"{p['p50']:.1f} min")
        kp2.metric("P75", f"{p['p75']:.1f} min")
        kp3.metric("P90 (Worst 10%)", f"{p['p90']:.1f} min")
        if dt["has_duration"]:
            kp4.metric("% Started Late", f"{dt['pct_started_late']:.0f}%",
                       help="% of flights where this activity started after its planned time (threshold = 0 min).")
        else:
            kp4.metric("P25", f"{p['p25']:.1f} min")
        _cover = f"Finish-delay percentiles based on **{dt['n_finish']:,}** flights."
        if dt["has_duration"]:
            _cover += (f" Start/duration decomposition based on **{dt['n_decomp']:,}** flights"
                       " (requires both actual start & end recorded).")
        st.caption(_cover)

        st.divider()
        st.markdown("#### 📊 Delay Breakdown by Day & Type")

        dow_col_chart, type_col_chart = st.columns(2)

        # Day of week chart
        with dow_col_chart:
            st.markdown("**Avg Delay by Day of Week**")
            DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            if "Day_of_Week" in df.columns and delay_col in df.columns:
                dow_df = df[["Day_of_Week", delay_col]].dropna()
                dow_agg = (dow_df.groupby("Day_of_Week")[delay_col]
                           .agg(mean="mean", n="count").reset_index())
                dow_agg = dow_agg[dow_agg["n"] >= 10]
                dow_agg["Day_of_Week"] = pd.Categorical(
                    dow_agg["Day_of_Week"], categories=DAY_ORDER, ordered=True)
                dow_agg = dow_agg.sort_values("Day_of_Week")
                worst_dow = dow_agg.loc[dow_agg["mean"].idxmax(), "Day_of_Week"] if not dow_agg.empty else "—"
                dow_fig = go.Figure(go.Bar(
                    x=dow_agg["Day_of_Week"],
                    y=dow_agg["mean"],
                    marker_color=[
                        bu_info["color"] if d != worst_dow else "#e74c3c"
                        for d in dow_agg["Day_of_Week"]
                    ],
                    text=[f"{v:.1f}" for v in dow_agg["mean"]],
                    textposition="outside",
                    customdata=dow_agg["n"],
                    hovertemplate="%{x}<br>Avg delay: %{y:.1f} min<br>Flights: %{customdata}<extra></extra>",
                ))
                dow_fig.add_hline(y=0, line_dash="dash", line_color="#2ecc71", line_width=1)
                dow_fig.update_layout(
                    template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="", yaxis_title="Avg Delay (min)",
                    height=300, margin=dict(t=10, b=50, l=50, r=10), showlegend=False,
                )
                st.plotly_chart(dow_fig, use_container_width=True)
            else:
                st.info("Day_of_Week column not available.")

        # Delay type breakdown (pie) or box plot fallback
        with type_col_chart:
            if dt["has_duration"]:
                st.markdown("**Delay Type Breakdown**")
                cats  = dt["cats"]
                lbls  = list(cats.keys())
                vals  = list(cats.values())
                pie_fig = go.Figure(go.Pie(
                    labels=lbls, values=vals, hole=0.48,
                    marker=dict(colors=["#2ecc71","#f39c12","#e67e22","#e74c3c"]),
                    textinfo="percent+label",
                    textfont=dict(size=9.5, color=FC),
                    hovertemplate="%{label}<br>%{value:,} flights (%{percent})<extra></extra>",
                ))
                pie_fig.add_annotation(
                    text=f"<b>{dt['n_decomp']:,}</b><br><span style='font-size:10px'>flights</span>",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=13, color=FC),
                )
                pie_fig.update_layout(
                    template=TEMPLATE, height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=50, l=10, r=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.08,
                                xanchor="center", x=0.5, font=dict(size=9)),
                )
                st.plotly_chart(pie_fig, use_container_width=True)
            else:
                st.markdown("**Delay Distribution (Box Plot)**")
                box_fig = go.Figure(go.Box(
                    y=dt["finish_delay"].clip(-30, 120),
                    name=act_name.split(": ")[-1],
                    marker_color=bu_info["color"], boxpoints="outliers",
                    line=dict(color=bu_info["color"]),
                ))
                box_fig.add_hline(y=0, line_dash="dash", line_color="#2ecc71", line_width=1)
                box_fig.update_layout(
                    template=TEMPLATE, height=300,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Delay (min)",
                    margin=dict(t=10, b=40, l=50, r=10), showlegend=False,
                )
                st.plotly_chart(box_fig, use_container_width=True)

        # Start delay vs duration overrun component bar
        if dt["has_duration"]:
            st.markdown("#### ⏱ Start Delay vs Duration Overrun (averages)")
            insight_strip(
                "Start Delay = team arrived / began after the planned time. "
                "Duration Overrun = the activity itself took longer than planned. "
                "These are independent root causes requiring different fixes.",
                severity="blue",
            )
            sd_val  = max(0, dt["avg_start_delay"])
            do_val  = max(0, dt["avg_dur_overrun"])
            comp_fig = go.Figure()
            comp_fig.add_trace(go.Bar(
                name="Avg Start Delay (min)",
                x=["Delay Composition"],
                y=[sd_val],
                marker_color="#f39c12",
                text=[f"{sd_val:.1f} min"],
                textposition="outside",
            ))
            comp_fig.add_trace(go.Bar(
                name="Avg Duration Overrun (min)",
                x=["Delay Composition"],
                y=[do_val],
                marker_color="#e74c3c",
                text=[f"{do_val:.1f} min"],
                textposition="outside",
            ))
            comp_fig.update_layout(
                template=TEMPLATE, barmode="group", height=220,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=40, l=10, r=90),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                yaxis_title="Minutes",
                yaxis=dict(range=[0, max(sd_val, do_val) * 1.4 + 0.5]),
            )
            st.plotly_chart(comp_fig, use_container_width=True)
    else:
        st.info("No data")

    # ── Predecessor analysis ─────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔗 Why Did This Activity Start Late? — Upstream Correlations")
    st.caption(
        "If an upstream activity finished late, this one likely started late. "
        "Correlations below reveal which predecessors are most responsible."
    )

    _col_to_node = {n[2]: n[0] for n in _CASCADE_NODES if n[2]}
    _node_to_col = {n[0]: n[2] for n in _CASCADE_NODES if n[2]}
    _this_node   = _col_to_node.get(delay_col)

    if _this_node and _this_node in _CASCADE_DEPS:
        _predecessors = _CASCADE_DEPS[_this_node]
        _pred_results = []
        for _pred_name in _predecessors:
            _pred_col = _node_to_col.get(_pred_name)
            if not _pred_col or _pred_col not in df.columns or _pred_col == delay_col:
                continue
            _pair = df[[_pred_col, delay_col]].dropna()
            if len(_pair) < 30:
                _pred_results.append({
                    "Predecessor": _pred_name,
                    "Column": _pred_col,
                    "Correlation": None,
                    "n": len(_pair),
                    "note": "Insufficient data",
                })
                continue
            _corr = float(_pair[_pred_col].corr(_pair[delay_col]))
            _avg_pred_delay = float(_pair[_pred_col].mean())
            _pred_results.append({
                "Predecessor": _pred_name,
                "Column": _pred_col,
                "Correlation": _corr,
                "n": len(_pair),
                "Avg Pred Delay (min)": round(_avg_pred_delay, 1),
                "note": "",
            })

        if _pred_results:
            _pred_with_data = [r for r in _pred_results if r["Correlation"] is not None]
            _pred_no_data   = [r for r in _pred_results if r["Correlation"] is None]

            if _pred_with_data:
                for _pr in sorted(_pred_with_data, key=lambda x: -abs(x["Correlation"])):
                    _c   = _pr["Correlation"]
                    _clr = "#e74c3c" if _c > 0.5 else "#f39c12" if _c > 0.25 else "#2ecc71"
                    _str = "strong" if abs(_c) > 0.5 else "moderate" if abs(_c) > 0.25 else "weak"
                    st.markdown(
                        f"<div style='background:{card_bg()};border-left:3px solid {_clr};"
                        f"border-radius:0 10px 10px 0;padding:10px 18px;margin-bottom:8px'>"
                        f"<b style='color:{card_text()}'>{_pr['Predecessor']}</b> &nbsp;—&nbsp; "
                        f"<span style='color:{_clr}'>r = {_c:.2f} ({_str} correlation)</span>"
                        f"  ·  avg pred delay: <b>{_pr['Avg Pred Delay (min)']:.1f} min</b>"
                        f"  ·  <span style='color:{card_sub()}'>{_pr['n']:,} paired observations</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Scatter for strongest predecessor
                _top_pred = max(_pred_with_data, key=lambda x: abs(x["Correlation"]))
                _tp_col   = _top_pred["Column"]
                if _tp_col in df.columns:
                    _sp_df = df[[_tp_col, delay_col]].dropna().sample(
                        min(1500, len(df[[_tp_col, delay_col]].dropna())), random_state=42
                    )
                    _sp_fig = px.scatter(
                        _sp_df, x=_tp_col, y=delay_col,
                        opacity=0.3, trendline="ols",
                        labels={_tp_col: f"{_top_pred['Predecessor']} Delay (min)",
                                delay_col: f"{act_name.split(': ')[-1]} Delay (min)"},
                        template=TEMPLATE,
                        color_discrete_sequence=[bu_info["color"]],
                        title=f"Predecessor vs This Activity: r = {_top_pred['Correlation']:.2f}",
                    )
                    _sp_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=320, margin=dict(t=40, b=40, l=50, r=10), showlegend=False,
                    )
                    _sp_fig.update_traces(marker_size=5, selector=dict(mode="markers"))
                    st.plotly_chart(_sp_fig, use_container_width=True)

            for _pr in _pred_no_data:
                st.caption(f"↳ {_pr['Predecessor']}: No data available ({_pr['n']} paired rows)")
        else:
            st.info("No direct predecessors defined in the cascade graph for this activity.")
    elif not _this_node:
        st.info("This activity is not mapped to a cascade node — predecessor analysis not available.")
    else:
        st.info("This activity has no upstream predecessors in the cascade (it is triggered directly by Aircraft Arrives).")

    # ── Monthly trend ────────────────────────────────────────────────────────
    if "departure_offBlock.scheduled" in df.columns and delay_col in df.columns:
        st.divider()
        st.markdown("#### 📅 Monthly Trend")
        st.caption("Is this activity getting better or worse over time?")

        _trend_df = df[["departure_offBlock.scheduled", delay_col]].copy()
        _trend_df["_dt"] = pd.to_datetime(_trend_df["departure_offBlock.scheduled"], errors="coerce", utc=True)
        _trend_df = _trend_df.dropna(subset=["_dt", delay_col])
        _trend_df["_month"] = _trend_df["_dt"].dt.tz_localize(None).dt.to_period("M").astype(str)
        _month_agg = (
            _trend_df.groupby("_month")[delay_col]
            .agg(avg_delay="mean", std_delay="std",
                 p90_delay=lambda x: x.quantile(0.90),
                 late_pct=lambda x: (x > 0).mean() * 100, n="count")
            .reset_index()
        )
        _month_agg = _month_agg[_month_agg["n"] >= 20].sort_values("_month")

        if len(_month_agg) >= 2:
            _hx = bu_info["color"].lstrip("#")
            _rr, _gg, _bb = int(_hx[0:2], 16), int(_hx[2:4], 16), int(_hx[4:6], 16)
            _band_fill = f"rgba({_rr},{_gg},{_bb},0.13)"
            _hi = (_month_agg["avg_delay"] + _month_agg["std_delay"]).fillna(_month_agg["avg_delay"])
            _lo = (_month_agg["avg_delay"] - _month_agg["std_delay"]).fillna(_month_agg["avg_delay"])

            _tm_fig = go.Figure()
            _tm_fig.add_trace(go.Scatter(
                x=list(_month_agg["_month"]) + list(_month_agg["_month"])[::-1],
                y=list(_hi) + list(_lo)[::-1],
                fill="toself", fillcolor=_band_fill,
                line=dict(color="rgba(0,0,0,0)"),
                name="±1σ band", showlegend=True, hoverinfo="skip",
            ))
            _tm_fig.add_trace(go.Scatter(
                x=_month_agg["_month"], y=_month_agg["avg_delay"],
                mode="lines+markers", name="Avg Delay (min)",
                line=dict(color=bu_info["color"], width=2),
                marker=dict(size=6),
                hovertemplate="%{x}<br>Avg: %{y:.1f} min<extra></extra>",
            ))
            _tm_fig.add_trace(go.Scatter(
                x=_month_agg["_month"], y=_month_agg["p90_delay"],
                mode="lines", name="P90",
                line=dict(color="#e74c3c", dash="dot", width=1.5),
                hovertemplate="%{x}<br>P90: %{y:.1f} min<extra></extra>",
            ))
            _tm_fig.add_trace(go.Scatter(
                x=_month_agg["_month"], y=_month_agg["late_pct"],
                mode="lines+markers", name="% Late",
                line=dict(color="#f39c12", width=1.5, dash="dot"),
                marker=dict(size=4),
                yaxis="y2",
                hovertemplate="%{x}<br>%% late: %{y:.1f}%<extra></extra>",
            ))
            _tm_fig.add_hline(y=0, line_dash="dash", line_color="#2ecc71", line_width=1)
            _tm_fig.update_layout(
                template=TEMPLATE, height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Month", gridcolor="rgba(128,128,128,0.10)"),
                yaxis=dict(title="Delay (min)", gridcolor="rgba(128,128,128,0.08)"),
                yaxis2=dict(title="% Late", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)", range=[0, 100]),
                margin=dict(t=10, b=50, l=50, r=60),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(_tm_fig, use_container_width=True)
        else:
            st.info("Not enough monthly data to show a trend (need at least 2 months with 20+ flights each).")

    # ── Day × Hour heatmap ───────────────────────────────────────────────────
    if "Day_of_Week" in df.columns and "Hour_of_Day" in df.columns and delay_col in df.columns:
        st.divider()
        st.markdown("#### 🗓️ When Does It Run Late? — Day × Hour Heatmap")
        st.caption("Cell = % of flights where this activity ran late. Darker red = more frequent delays.")

        _DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        _hm_df = df[["Day_of_Week", "Hour_of_Day", delay_col]].dropna()
        _hm_piv = (
            _hm_df.groupby(["Day_of_Week", "Hour_of_Day"])[delay_col]
            .apply(lambda x: (x > 0).mean() * 100)
            .reset_index()
            .rename(columns={delay_col: "pct_late"})
        )
        # Filter to cells with enough flights
        _hm_n = (
            _hm_df.groupby(["Day_of_Week", "Hour_of_Day"])[delay_col]
            .count().reset_index().rename(columns={delay_col: "n"})
        )
        _hm_piv = _hm_piv.merge(_hm_n, on=["Day_of_Week", "Hour_of_Day"])
        _hm_piv = _hm_piv[_hm_piv["n"] >= 5]

        if not _hm_piv.empty:
            _hm_piv["Day_of_Week"] = pd.Categorical(
                _hm_piv["Day_of_Week"], categories=_DAY_ORDER, ordered=True
            )
            _hm_piv = _hm_piv.sort_values(["Day_of_Week", "Hour_of_Day"])
            _pivot_tbl = _hm_piv.pivot(index="Day_of_Week", columns="Hour_of_Day", values="pct_late")
            _pivot_tbl = _pivot_tbl.reindex(_DAY_ORDER)

            _hm_fig = go.Figure(go.Heatmap(
                z=_pivot_tbl.values,
                x=[f"{h:02d}:00" for h in _pivot_tbl.columns],
                y=_pivot_tbl.index.tolist(),
                colorscale=[[0, "#0d1117"], [0.5, "#7b1818"], [1, "#e74c3c"]],
                zmin=0, zmax=100,
                colorbar=dict(title="% Late", ticksuffix="%"),
                hovertemplate="Day: %{y}<br>Hour: %{x}<br>% Late: %{z:.1f}%<extra></extra>",
            ))
            _hm_fig.update_layout(
                template=TEMPLATE, height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Hour (UTC)", side="bottom"),
                yaxis=dict(title=""),
                margin=dict(t=10, b=50, l=90, r=10),
            )
            st.plotly_chart(_hm_fig, use_container_width=True)
        else:
            st.info("Not enough data to build a day × hour heatmap (need 5+ flights per cell).")

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — BU detail
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["selected_bu"]:
    bu_key  = st.session_state["selected_bu"]
    bu_info = BU_INFO.get(bu_key, {"color": "#9e9e9e", "label": bu_key, "icon": ""})
    breadcrumb()

    bc1, bc2 = st.columns([1, 10])
    with bc1:
        if st.button("← Back", key="back_to_overview"):
            back_to_overview()
            st.rerun()
    st.divider()

    bu_acts = stats_filtered[stats_filtered["team_clean"] == bu_key].sort_values("avg_delay_mins", ascending=False)

    st.markdown(
        f"<h2 style='color:{bu_info['color']}'>"
        f"{bu_info['icon']} {bu_info['label']}</h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"{len(bu_acts)} activities tracked across {bu_acts['count'].sum():,.0f} flight records.")

    # BU-level insight
    if not bu_acts.empty:
        _worst_act = bu_acts.iloc[0]
        _best_act  = bu_acts.iloc[-1]
        # Flight-count-weighted, not a flat mean across activities
        _avg_late  = (bu_acts["late_rate"] * bu_acts["count"]).sum() / bu_acts["count"].sum() * 100
        insight_card(
            problem=(f"Within **{bu_info['label']}**, the slowest activity is "
                     f"**{_worst_act['activity'].split(': ')[-1]}** — on average it exceeded "
                     f"**PTS** by **{_worst_act['avg_delay_mins']:.1f} min** "
                     f"and runs late on {_worst_act['late_rate']*100:.0f}% of flights."),
            impact=(f"The team average is {_avg_late:.0f}% late across all activities. "
                    f"The best activity ({_best_act['activity'].split(': ')[-1]}) runs at only "
                    f"{_best_act['avg_delay_mins']:.1f} min avg — the gap shows where "
                    f"improvement is possible."),
            action=(f"Click **{_worst_act['activity'].split(': ')[-1]}** below to see its "
                    "delay distribution by hour of day and its correlation with departure delay. "
                    "Identify whether it is a resources, process, or timing issue."),
            icon=bu_info["icon"], severity="amber",
        )

    # KPI row
    k1, k2, k3 = st.columns(3)
    _bu_total = int(bu_acts["count"].sum()) if not bu_acts.empty else 0
    if bu_acts.empty:
        with k1: no_data_metric("Avg Delay Across Activities", 0, min_n=1)
        with k2: no_data_metric("Avg Late Rate", 0, min_n=1)
        k3.metric("Worst Activity", "—")
    elif _bu_total < BU_MIN_TOTAL_FLIGHTS:
        with k1: no_data_metric("Avg Delay Across Activities", _bu_total, min_n=BU_MIN_TOTAL_FLIGHTS)
        with k2: no_data_metric("Avg Late Rate", _bu_total, min_n=BU_MIN_TOTAL_FLIGHTS)
        k3.metric("Worst Activity", bu_acts.iloc[0]["activity"].split(": ")[-1])
    else:
        _w = bu_acts["count"]
        k1.metric("Avg Delay Across Activities",
                 f"{(bu_acts['avg_delay_mins'] * _w).sum() / _w.sum():.1f} min")
        k2.metric("Avg Late Rate",
                 f"{(bu_acts['late_rate'] * _w).sum() / _w.sum() * 100:.1f}%")
        k3.metric("Worst Activity", bu_acts.iloc[0]["activity"].split(": ")[-1])

    st.divider()

    if bu_acts.empty:
        st.info("No activity data available for this team.")
        st.stop()

    # Bar chart with std dev error bars and P90 markers
    st.markdown("#### Activities by Average Delay")
    bar_data = bu_acts.dropna(subset=["avg_delay_mins"]).sort_values("avg_delay_mins")
    _std_vals, _p90_vals = [], []
    for _, _br in bar_data.iterrows():
        _dc = _br["raw_name"] + "_analysis_Delay_mins"
        if _dc in df.columns:
            _s = df[_dc].dropna()
            _std_vals.append(float(_s.std())          if len(_s) > 1 else 0.0)
            _p90_vals.append(float(_s.quantile(0.90)) if len(_s) > 1 else 0.0)
        else:
            _std_vals.append(0.0); _p90_vals.append(0.0)

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        y=bar_data["activity"].str.split(": ").str[-1],
        x=bar_data["avg_delay_mins"],
        orientation="h",
        marker_color=bu_info["color"],
        error_x=dict(type="data", array=_std_vals, visible=True,
                     color="rgba(180,180,180,0.55)", thickness=1.5, width=4),
        text=[f"{v:.1f}  σ±{s:.1f}" for v, s in zip(bar_data["avg_delay_mins"], _std_vals)],
        textposition="outside",
        name="Avg ± 1σ",
        hovertemplate="<b>%{y}</b><br>Avg: %{x:.1f} min<extra></extra>",
    ))
    bar_fig.add_trace(go.Scatter(
        y=bar_data["activity"].str.split(": ").str[-1],
        x=_p90_vals,
        mode="markers", name="P90",
        marker=dict(symbol="diamond", size=9, color="#e74c3c",
                    line=dict(width=1, color=FC)),
        hovertemplate="<b>%{y}</b><br>P90: %{x:.1f} min<extra></extra>",
    ))
    _x_max = max(
        bar_data["avg_delay_mins"].max() + (max(_std_vals) if _std_vals else 0),
        max(_p90_vals) if _p90_vals else 0,
    ) * 1.35
    bar_fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Delay (min)", yaxis_title="",
        height=max(280, len(bar_data) * 35 + 60),
        margin=dict(t=10, b=40, l=20, r=120),
        xaxis=dict(range=[0, _x_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Delay Type Summary — All Activities")
    st.caption(
        "For each activity: % that started late, % that finished late, "
        "and % where duration exceeded plan. These three flags are independent root causes."
    )

    _summary_rows = []
    for _, _ar in bu_acts.iterrows():
        _dc   = _ar["raw_name"] + "_analysis_Delay_mins"
        _durc = _ar["raw_name"] + "_analysis_ActualDuration_mins"
        _plnc = _ar["raw_name"] + "_analysis_PlannedDuration_mins"
        _n_flights = int(_ar["count"])

        _pct_finish_late = _ar["late_rate"] * 100

        if _dc in df.columns and _durc in df.columns and _plnc in df.columns:
            _s = df[[_dc, _durc, _plnc]].dropna()
            if len(_s) >= 10:
                _fd   = _s[_dc].astype(float)
                _do   = (_s[_durc] - _s[_plnc]).astype(float)
                _sl   = _fd - _do
                _pct_start_late    = float((_sl > 0).mean() * 100)
                _pct_dur_exceeded  = float((_do > 0).mean() * 100)
                _avg_start_d       = float(_sl.mean())
                _avg_dur_over      = float(_do.mean())
            else:
                _pct_start_late = _pct_dur_exceeded = _avg_start_d = _avg_dur_over = None
        else:
            _pct_start_late = _pct_dur_exceeded = _avg_start_d = _avg_dur_over = None

        _summary_rows.append({
            "Activity":          _ar["activity"].split(": ")[-1],
            "Flights":           _n_flights,
            "Times Recorded":    _n_flights,
            "% Finish Late":     round(_pct_finish_late, 1),
            "% Start Late":      round(_pct_start_late, 1) if _pct_start_late is not None else "No data",
            "% Duration Over":   round(_pct_dur_exceeded, 1) if _pct_dur_exceeded is not None else "No data",
            "Avg Finish Delay":  round(_ar["avg_delay_mins"], 1),
            "Avg Start Delay":   round(_avg_start_d, 1) if _avg_start_d is not None else "—",
            "Avg Dur Overrun":   round(_avg_dur_over, 1) if _avg_dur_over is not None else "—",
        })

    if _summary_rows:
        _sum_df = pd.DataFrame(_summary_rows).sort_values("% Finish Late", ascending=False)

        def _color_pct(val):
            try:
                v = float(val)
                if v > 60: return "color:#e74c3c;font-weight:700"
                if v > 35: return "color:#f39c12;font-weight:600"
                return "color:#2ecc71"
            except Exception:
                return "color:#6b7fa3"

        _styled = (
            _sum_df.style
            .map(_color_pct, subset=["% Finish Late", "% Start Late", "% Duration Over"])
            .format({"Flights": "{:,}", "Times Recorded": "{:,}", "Avg Finish Delay": "{:.1f}", "Avg Start Delay": "{}", "Avg Dur Overrun": "{}"})
        )
        st.dataframe(_styled, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Click an Activity to Explore It")

    # Clickable activity cards
    cols_per_row = 3
    act_list = bu_acts.reset_index(drop=True)
    for i in range(0, len(act_list), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(row_cols):
            idx = i + j
            if idx >= len(act_list):
                break
            row = act_list.iloc[idx]
            act_label = row["activity"].split(": ")[-1]
            late_pct = row["late_rate"] * 100
            avg_d = row["avg_delay_mins"]
            color = bu_info["color"]
            with col:
                st.markdown(
                    f"""<div style="background:{card_bg()};border-left:4px solid {color};
                        border-radius:10px;padding:14px 16px;margin-bottom:8px">
                      <div style="font-size:.8rem;color:{card_sub()};text-transform:uppercase;letter-spacing:1px">
                        {act_label}
                      </div>
                      <div style="font-size:1.4rem;font-weight:700;color:{color};margin:6px 0">
                        {avg_d:.1f} min over PTS
                      </div>
                      <div style="font-size:.78rem;color:{card_sub()}">{late_pct:.0f}% late · {int(row['count']):,} flights</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button(f"Explore →", key=f"act_{bu_key}_{idx}"):
                    st.session_state["selected_activity"] = row["activity"]
                    st.rerun()

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — BU Overview
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔧 Activity Analysis")
st.caption("Select a Business Unit below to drill into its activities, then click any activity for a full breakdown.")
st.divider()

if stats_filtered.empty:
    st.warning("No activity data found. Please ensure prepare_data.py has run successfully.")
    st.stop()

# ─── Page Insight ─────────────────────────────────────────────────────────────
_as = compute_activity_stats(bu_stats)
if _as:
    insight_card(
        problem=(f"**{_as['worst_label']}** is the worst-performing team — "
                 f"average **{_as['worst_delay']:.1f} min** delay per activity, "
                 f"{_as['worst_rate']:.0f}% of their activities run late. "
                 f"Their single biggest problem is **{_as['worst_act']}**."),
        impact=("Every minute a ground activity overruns compresses the available turnaround "
                "window and directly increases the probability of a late departure. "
                "Activities that chain (e.g. loading before door close) multiply the effect."),
        action=(f"Click **{_as['worst_icon']} {_as['worst_label']}** below → then select "
                f"**{_as['worst_act']}** to see its delay distribution, peak hours, "
                "and correlation with departure delay. That is where to start the fix."),
        icon="🔧", severity="amber",
    )

# ── BU summary cards ───────────────────────────────────────────────────────────
st.markdown("### Business Units — Performance at a Glance")

if not bu_stats.empty:
    cols_per_row = 4
    for i in range(0, len(bu_stats), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(row_cols):
            idx = i + j
            if idx >= len(bu_stats):
                break
            row = bu_stats.iloc[idx]
            color = row["color"]
            late_pct = row["avg_late_rate"] * 100
            n_acts = int(row["n_activities"])
            bu_total = int(row["total_count"])
            with col:
                if bu_total < BU_MIN_TOTAL_FLIGHTS:
                    no_data_card(
                        row["label"], bu_total, icon=row["icon"], min_n=BU_MIN_TOTAL_FLIGHTS,
                    )
                else:
                    st.markdown(
                        f"""<div style="background:{card_bg()};border:1px solid {color}33;
                            border-top:4px solid {color};border-radius:12px;
                            padding:16px 18px;margin-bottom:8px">
                          <div style="font-size:1.5rem">{row['icon']}</div>
                          <div style="font-size:1rem;font-weight:700;color:{card_text()};margin:6px 0">
                            {row['label']}
                          </div>
                          <div style="font-size:1.6rem;font-weight:900;color:{color}">
                            {row['avg_delay']:.1f} min
                          </div>
                          <div style="font-size:.75rem;color:{card_sub()};margin:4px 0">
                            avg delay · {late_pct:.0f}% late rate
                          </div>
                          <div style="font-size:.72rem;color:{card_sub()};margin-top:6px">
                            {n_acts} activities tracked
                          </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                if st.button(f"View {row['label']} →", key=f"bu_btn_{row['bu']}"):
                    st.session_state["selected_bu"] = row["bu"]
                    st.session_state["selected_activity"] = None
                    st.rerun()

st.divider()

# ── Cross-BU comparison chart ─────────────────────────────────────────────────
st.markdown("### Average Delay by Business Unit")

_bu_reliable = bu_stats[bu_stats["total_count"] >= BU_MIN_TOTAL_FLIGHTS]
_bu_excluded = bu_stats[bu_stats["total_count"] < BU_MIN_TOTAL_FLIGHTS]

if not _bu_reliable.empty:
    bu_sorted = _bu_reliable.sort_values("avg_delay", ascending=True)
    bu_bar = go.Figure(go.Bar(
        y=bu_sorted["label"],
        x=bu_sorted["avg_delay"],
        orientation="h",
        marker=dict(color=bu_sorted["color"].tolist()),
        text=[f"{v:.1f} min" for v in bu_sorted["avg_delay"]],
        textposition="outside",
        textfont=dict(size=12),
    ))
    bu_bar.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Average Activity Delay (minutes)",
        height=320, margin=dict(t=10, b=40, l=20, r=90),
        xaxis=dict(range=[0, bu_sorted["avg_delay"].max() * 1.25]),
    )
    st.plotly_chart(bu_bar, use_container_width=True)
else:
    no_data_info("Average Delay by Business Unit", n=0, min_n=BU_MIN_TOTAL_FLIGHTS)

if not _bu_excluded.empty:
    st.caption(
        f"ℹ️ Not shown (fewer than {BU_MIN_TOTAL_FLIGHTS:,} recorded flights): "
        + ", ".join(_bu_excluded["label"].tolist())
    )

st.divider()

# ── On-time performance stacked bar ───────────────────────────────────────────
st.markdown("### On-Time Performance by Business Unit")
st.caption(
    "% of milestone completions that were on-time vs delayed. "
    "Any positive delay = Delayed (milestones use binary threshold — 0 tolerance)."
)

bu_perf_rows = []
for bu_key, info in BU_INFO.items():
    delay_cols = [
        c for c in df.columns
        if f"milestone_{bu_key}" in c.lower() and "_analysis_Delay_mins" in c
        and "ActualDuration" not in c and "PlannedDuration" not in c
    ]
    vals = []
    for col in delay_cols:
        col_data = df[col].dropna()
        if len(col_data) < 50:
            continue
        vals.append(col_data.values)
    if not vals:
        continue
    combined = np.concatenate(vals)
    total = len(combined)
    bu_perf_rows.append({
        "BU":      info["label"],
        "color":   info["color"],
        "On-Time": float((combined <= 0).sum() / total * 100),
        "Delayed": float((combined > 0).sum()  / total * 100),
    })

if bu_perf_rows:
    perf_df = pd.DataFrame(bu_perf_rows).sort_values("Delayed", ascending=False)
    stacked = go.Figure()
    for cls, clr in [("On-Time", "#2ecc71"), ("Delayed", "#e74c3c")]:
        stacked.add_trace(go.Bar(
            x=perf_df["BU"], y=perf_df[cls], name=cls,
            marker_color=clr,
            text=[f"{v:.0f}%" for v in perf_df[cls]],
            textposition="inside", textfont=dict(size=11, color=FC),
        ))
    stacked.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        barmode="stack", height=340, margin=dict(t=10, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="% of milestone completions",
    )
    st.plotly_chart(stacked, use_container_width=True)
else:
    no_data_info("On-Time Performance by Business Unit", n=0, min_n=50)

st.divider()

# ── Bubble chart: frequency vs impact ────────────────────────────────────────
st.markdown("### Activity Risk Matrix — Frequency vs Impact")
st.caption("Top-right = most problematic. Bubble size = number of recorded flights.")

bubble_data = stats_filtered.dropna(subset=["late_rate","avg_delay_mins"])
bubble_data = bubble_data[bubble_data["late_rate"].between(0, 1)]

if not bubble_data.empty:
    # Identify top risk activities (frequency * impact) to label on the chart to prevent text overlap
    bubble_data = bubble_data.copy()
    bubble_data["risk_score"] = bubble_data["late_rate"] * bubble_data["avg_delay_mins"]
    top_risk_activities = set(bubble_data.nlargest(6, "risk_score")["activity"])

    bubble_fig = go.Figure()
    for bu_key, info in BU_INFO.items():
        tdf = bubble_data[bubble_data["team_clean"] == bu_key]
        if tdf.empty:
            continue
        
        # Only show text labels for top-risk outliers to keep the chart clean
        display_text = [
            act.split(": ")[-1] if act in top_risk_activities else ""
            for act in tdf["activity"]
        ]

        bubble_fig.add_trace(go.Scatter(
            x=tdf["late_rate"] * 100,
            y=tdf["avg_delay_mins"],
            mode="markers+text",
            name=info["label"],
            marker=dict(
                size=np.sqrt(tdf["count"].clip(100, 10000)) * 1.2,
                color=info["color"], opacity=0.8,
                line=dict(width=1, color=FC),
            ),
            text=display_text,
            textposition="top center", textfont=dict(size=9),
            customdata=tdf["activity"],
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Late: %{x:.1f}%<br>"
                "Avg delay: %{y:.1f} min<br>"
                "<extra></extra>"
            ),
        ))
    bubble_fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="How Often Late (%)", yaxis_title="Avg Delay When Late (min)",
        height=480, margin=dict(t=20, b=40, l=60, r=20),
        legend=dict(title="Business Unit"),
    )
    st.plotly_chart(bubble_fig, use_container_width=True)
else:
    no_data_info("Activity Risk Matrix", n=0, min_n=50)

st.divider()

# ── Delay Type Composition per BU ─────────────────────────────────────────────
st.markdown("### ⏱ Delay Type Composition — Start Delay vs Duration Overrun")
st.caption(
    "For **interval activities** (those with both a start and end timestamp): "
    "how much of average delay comes from a **late start** (team arrived after scheduled time) "
    "vs **slow execution** (the work itself took longer than planned)? "
    "Point milestones (single-event) and activities with no start/end data recorded "
    "are listed in the expander below."
)

type_rows = []
no_data_acts_point  = []   # point milestones — duration concept not applicable
no_data_acts_empty  = []   # interval activities — columns exist but all-null in dataset

for _, arow in stats_filtered.iterrows():
    dt_bu = compute_delay_types(df, arow["raw_name"])
    bu = arow["team_clean"]
    info = BU_INFO.get(bu, {"label": bu.title(), "color": "#888"})
    act_short = arow["activity"].split(": ")[-1]
    if dt_bu is None or not dt_bu["has_duration"]:
        dur_col_check = arow["raw_name"] + "_analysis_ActualDuration_mins"
        if dur_col_check in df.columns:
            no_data_acts_empty.append(f"{info['label']} · {act_short}")
        else:
            no_data_acts_point.append(f"{info['label']} · {act_short}")
        continue
    type_rows.append({
        "bu":           bu,
        "label":        info["label"],
        "color":        info["color"],
        "activity":     act_short,
        "avg_start":    dt_bu["avg_start_delay"],
        "avg_dur_over": dt_bu["avg_dur_overrun"],
        "avg_finish":   float(dt_bu["finish_delay"].mean()),
    })

if type_rows:
    type_df = pd.DataFrame(type_rows)

    # ── Per-activity stacked bar ───────────────────────────────────────────────
    type_df_sorted = type_df.sort_values("avg_finish", ascending=True)
    t_fig = go.Figure()
    t_fig.add_trace(go.Bar(
        name="Start Delay (min)",
        y=type_df_sorted["label"] + " · " + type_df_sorted["activity"],
        x=type_df_sorted["avg_start"].clip(lower=0),
        orientation="h",
        marker_color="#f39c12",
        hovertemplate="<b>%{y}</b><br>Avg start delay: %{x:.1f} min<extra></extra>",
    ))
    t_fig.add_trace(go.Bar(
        name="Duration Overrun (min)",
        y=type_df_sorted["label"] + " · " + type_df_sorted["activity"],
        x=type_df_sorted["avg_dur_over"].clip(lower=0),
        orientation="h",
        marker_color="#e74c3c",
        hovertemplate="<b>%{y}</b><br>Avg duration overrun: %{x:.1f} min<extra></extra>",
    ))
    t_fig.update_layout(
        template=TEMPLATE, barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Average Delay (min)", yaxis_title="",
        height=max(340, len(type_df_sorted) * 28 + 80),
        margin=dict(t=20, b=40, l=20, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(t_fig, use_container_width=True)
    insight_strip(
        "🔑 Orange = late start (coordination / handover issue). "
        "Red = slow execution (process speed / resource constraint). "
        "Click into any BU → activity to see the full delay type breakdown for that specific activity.",
        severity="blue",
    )

_total_no_data = len(no_data_acts_point) + len(no_data_acts_empty)
if _total_no_data:
    with st.expander(f"ℹ️ {_total_no_data} activities not shown in chart above"):
        if no_data_acts_empty:
            st.markdown(
                f"**{len(no_data_acts_empty)} interval activit{'ies' if len(no_data_acts_empty)>1 else 'y'} — "
                "no start/end timestamps available:**"
            )
            st.caption(
                "These activities have start/end milestone columns but no actual timestamps in this "
                "dataset, so duration decomposition cannot be computed. No data here can mean the "
                "service isn’t provided for those airlines/flights, or it simply wasn’t captured."
            )
            for _a in sorted(no_data_acts_empty):
                st.markdown(f"- {_a}")
        if no_data_acts_point:
            st.markdown(
                f"**{len(no_data_acts_point)} point milestone{'s' if len(no_data_acts_point)>1 else ''} — "
                "single-event, duration concept not applicable:**"
            )
            st.caption(
                "Point milestones record a single moment in time (e.g. 'Man at Bay'). "
                "There is no start→end interval to decompose — their delay is entirely "
                "a late-arrival / late-completion figure. Click into any activity for its full stats."
            )
            for _a in sorted(no_data_acts_point):
                st.markdown(f"- {_a}")

if not type_rows and not _total_no_data:
    st.info("Delay type decomposition requires planned duration data — not available for point milestone activities.")

if type_rows:
    type_df = pd.DataFrame(type_rows)

    st.divider()

    # ── BU-level aggregated comparison ────────────────────────────────────────
    st.markdown("### 🏢 Delay Type Summary by Business Unit")
    st.caption(
        "Average start delay vs duration overrun per BU — two independent root causes. "
        "Start delay = team arrived late; Duration overrun = work took longer than planned."
    )
    bu_type = (type_df.groupby(["bu", "label", "color"])
               [["avg_start", "avg_dur_over"]].mean().reset_index())
    bu_type = bu_type.sort_values("avg_start", ascending=False)

    bu_comp = go.Figure()
    bu_comp.add_trace(go.Bar(
        name="Avg Start Delay (min)",
        y=bu_type["label"],
        x=bu_type["avg_start"].clip(lower=0),
        orientation="h",
        marker_color="#f39c12",
        text=[f"{v:.1f}" for v in bu_type["avg_start"].clip(lower=0)],
        textposition="outside",
    ))
    bu_comp.add_trace(go.Bar(
        name="Avg Duration Overrun (min)",
        y=bu_type["label"],
        x=bu_type["avg_dur_over"].clip(lower=0),
        orientation="h",
        marker_color="#e74c3c",
        text=[f"{v:.1f}" for v in bu_type["avg_dur_over"].clip(lower=0)],
        textposition="outside",
    ))
    _x_max = max(
        bu_type["avg_start"].clip(lower=0).max(),
        bu_type["avg_dur_over"].clip(lower=0).max(),
    ) * 1.35
    bu_comp.update_layout(
        template=TEMPLATE, barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Average Delay Component (min)", height=320,
        margin=dict(t=10, b=40, l=20, r=90),
        xaxis=dict(range=[0, _x_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(bu_comp, use_container_width=True)
