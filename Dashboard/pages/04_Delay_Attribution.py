"""
pages/04_Delay_Attribution.py
Delay Attribution — classifies each delayed flight as SATS-attributable,
propagated from incoming flight, tight schedule (planning), weather / lightning
warning, or unknown.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.loader       import load_data, render_date_filters, load_lw_data, merge_lw_features
from utils.style        import inject_css, chart_template, chart_fc, card_text, card_sub, header_bg, header_border
from utils.insights     import insight_card, insight_strip
from utils.crossfilter  import init_xf, apply_xf, render_xf_bar, handle_selection, get_xf, bar_colors, pie_pull

st.set_page_config(
    page_title="Delay Attribution | SATS",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
TEMPLATE = chart_template()
FC       = chart_fc()

# ── Data ──────────────────────────────────────────────────────────────────────
df    = load_data()
df    = render_date_filters(df, page_key="attribution")
df    = merge_lw_features(df)
init_xf()
df    = apply_xf(df)

valid = df.dropna(subset=["Target_Departure_Delay_Class"])
valid = valid[valid["Target_Departure_Delay_Class"] != "nan"]

# ── LW data availability ──────────────────────────────────────────────────────
_, lw_daily = load_lw_data()
lw_has_data = len(lw_daily) > 0
lw_covered  = set(lw_daily["date"].tolist()) if lw_has_data else set()
lw_date_min = lw_daily["date"].min() if lw_has_data else None
lw_date_max = lw_daily["date"].max() if lw_has_data else None

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{header_bg()};
            border:1px solid {header_border()};border-radius:16px;
            padding:22px 32px;margin-bottom:20px">
  <h2 style="margin:0;color:{card_text()};font-size:1.5rem">🔍 Delay Attribution</h2>
  <p style="margin:8px 0 0;color:{card_sub()};font-size:.9rem">
    Not every late departure is a SATS problem. This page classifies the
    <b style="color:#c5d3f0">root cause</b> of each delayed flight — including
    <b style="color:#f1c40f">lightning warning (weather) events</b> — so you can
    separate what SATS controls from what it does not.
  </p>
</div>
""", unsafe_allow_html=True)
render_xf_bar()

# ── Attribution framework explanation ─────────────────────────────────────────
with st.expander("📖  How attribution works — click to read", expanded=False):
    st.markdown("""
    ### Attribution Categories

    | Category | Definition | SATS Responsible? |
    |---|---|---|
    | ⚡ **Weather (Lightning Warning)** | An active lightning warning (LW) was in effect during the ground-handling window or at scheduled departure. Ground operations are suspended or slowed during LW periods — SATS cannot safely continue work. | **No** |
    | 🔴 **SATS Operation** | Adequate ground time available, inbound on time, no active LW — but flight still departed late. Ground activities ran slow. | **Yes** |
    | 🟡 **Propagated Delay** | The inbound aircraft arrived significantly late (> 20 min). Turnaround window was compressed before SATS even began work. | Partial / No |
    | 🟠 **Tight Schedule** | Even with on-time arrival, the airline scheduled less ground time than the minimum required. The delay was baked in at planning stage. | No (planning issue) |
    | 🟣 **Compound** | Multiple causes combined (LW + propagated, propagated + tight schedule, etc.). | Shared |
    | ⚪ **Unknown** | Insufficient data to classify clearly. Requires airline activity codes to resolve. | Unknown |

    ### Lightning Warning priority
    LW attribution takes highest priority — if an active LW covered the ground-handling window,
    the delay is classified as weather-driven regardless of other factors. This prevents LW-driven
    delays from inflating SATS's operational accountability metrics.

    > **Data note:** LW data covers **{lw_min}** to **{lw_max}**. Flights outside this range have
    > LW features set to zero (no LW data available — those flights fall into other categories as normal).
    """.format(
        lw_min=str(lw_date_min) if lw_date_min else "N/A",
        lw_max=str(lw_date_max) if lw_date_max else "N/A",
    ))

st.divider()

# ── Attribution Logic ─────────────────────────────────────────────────────────
PROPAGATED_THRESHOLD = 20
TIGHT_SCHED_RATIO    = 1.0

ATTR_COLORS = {
    "Weather (Lightning Warning)":                "#f1c40f",
    "SATS Operation":                             "#e74c3c",
    "Propagated (inbound late)":                  "#f39c12",
    "Tight Schedule (planning)":                  "#e67e22",
    "Compound (LW + Propagated)":                 "#16a085",
    "Compound (LW + Tight Schedule)":             "#1abc9c",
    "Compound (propagated + tight schedule)":     "#9b59b6",
    "Unknown / Insufficient Data":                "#607d8b",
    "Not Delayed":                                "#2ecc71",
}


def attribute_flight(row) -> str:
    is_delayed = row.get("Target_Departure_Delay_Class") == "Delayed"
    if not is_delayed:
        return "Not Delayed"

    incoming  = row.get("Incoming_Delay_mins",      np.nan)
    deficient = row.get("Is_Ground_Time_Deficient",  np.nan)
    gt_ratio  = row.get("Ground_Time_Ratio",         np.nan)
    lw_ground = int(row.get("LW_Active_During_Ground_Time",   0))
    lw_dep    = int(row.get("LW_Active_At_Sched_Departure", 0))

    lw_active     = (lw_ground == 1 or lw_dep == 1)
    late_incoming = pd.notna(incoming) and incoming > PROPAGATED_THRESHOLD
    tight_sched   = (str(deficient) == "1" or deficient == 1.0 or
                     (pd.notna(gt_ratio) and gt_ratio < TIGHT_SCHED_RATIO))

    # LW takes highest priority — it suspends ground ops entirely
    if lw_active:
        if late_incoming:
            return "Compound (LW + Propagated)"
        if tight_sched:
            return "Compound (LW + Tight Schedule)"
        return "Weather (Lightning Warning)"

    if late_incoming and tight_sched:
        return "Compound (propagated + tight schedule)"
    if late_incoming:
        return "Propagated (inbound late)"
    if tight_sched:
        return "Tight Schedule (planning)"
    if pd.notna(incoming) and incoming <= PROPAGATED_THRESHOLD and not tight_sched:
        return "SATS Operation"
    return "Unknown / Insufficient Data"


valid = valid.copy()
valid["attribution"] = valid.apply(attribute_flight, axis=1)
delayed_only = valid[valid["Target_Departure_Delay_Class"] == "Delayed"].copy()

# ── Attribution counts ─────────────────────────────────────────────────────────
attr_counts = (delayed_only["attribution"]
               .value_counts()
               .reset_index()
               .rename(columns={"attribution": "Category", "count": "Delayed Flights"}))

if attr_counts.empty:
    st.warning("No delayed flights in the current filter selection.")
    st.stop()

total_delayed = len(delayed_only)
sats_n   = (delayed_only["attribution"] == "SATS Operation").sum()
prop_n   = (delayed_only["attribution"] == "Propagated (inbound late)").sum()
tight_n  = (delayed_only["attribution"] == "Tight Schedule (planning)").sum()
comp_n   = delayed_only["attribution"].str.startswith("Compound").sum()
lw_n     = (delayed_only["attribution"] == "Weather (Lightning Warning)").sum()
lw_all_n = delayed_only["attribution"].str.contains("LW").sum()  # all LW-involved
unk_n    = (delayed_only["attribution"] == "Unknown / Insufficient Data").sum()

sats_pct  = sats_n  / total_delayed * 100
prop_pct  = prop_n  / total_delayed * 100
tight_pct = tight_n / total_delayed * 100
lw_pct    = lw_all_n / total_delayed * 100

# ── Page insight ───────────────────────────────────────────────────────────────
_non_sats_pct = 100 - sats_pct
_lw_msg = (f" ⚡ **{lw_pct:.0f}% ({lw_all_n:,} flights)** are weather-driven (active lightning warning)."
           if lw_all_n > 0 else "")
insight_card(
    problem=(f"Of **{total_delayed:,} delayed flights**, "
             f"**{sats_pct:.0f}% ({sats_n:,})** are attributable to SATS ground operations.{_lw_msg} "
             f"**{_non_sats_pct:.0f}%** are caused by factors outside SATS control."),
    impact=(f"Roughly **{total_delayed - sats_n:,} delays cannot be fixed** by improving ground ops alone. "
            + (f"Lightning warnings account for {lw_pct:.0f}% of all delayed flights — "
               f"these should be excluded from SATS SLA targets. " if lw_all_n > 0 else "")
            + f"Blaming SATS for all {total_delayed:,} delays is inaccurate."),
    action=("Focus SATS improvement effort on the 🔴 SATS Operation category. "
            + ("For ⚡ LW delays, review whether pre-positioning plans during warning windows can reduce recovery time. " if lw_all_n > 0 else "")
            + "For 🟡 Propagated, work with airlines to review minimum connect times."),
    icon="🔍",
    severity="red" if sats_pct > 30 else "amber",
)

# ── KPI chips ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Delayed",          f"{total_delayed:,}")
k2.metric("🔴 SATS Operation",      f"{sats_pct:.0f}%",  f"{sats_n:,} flights")
k3.metric("⚡ Weather / LW",        f"{lw_pct:.0f}%",    f"{lw_all_n:,} flights")
k4.metric("🟡 Propagated",          f"{prop_pct:.0f}%",  f"{prop_n:,} flights")
k5.metric("🟠 Tight Schedule",      f"{tight_pct:.0f}%", f"{tight_n:,} flights")
k6.metric("⚪ Unknown",             f"{unk_n/total_delayed*100:.0f}%", f"{unk_n:,} flights")

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 1.6])

with col_l:
    st.markdown("#### Attribution Breakdown — All Delayed Flights")
    st.caption("Click a slice to cross-filter by carrier.")
    insight_strip(
        f"⚡ SATS directly controls <b>{sats_pct:.0f}%</b> of delays. "
        + (f"Lightning warnings explain <b>{lw_pct:.0f}%</b> — these are weather events. " if lw_all_n > 0 else "")
        + f"The remaining <b>{max(0, 100 - sats_pct - lw_pct):.0f}%</b> require airline or airport-level action.",
        severity="amber" if sats_pct > 30 else "green",
    )
    cats   = attr_counts["Category"].tolist()
    counts = attr_counts["Delayed Flights"].tolist()
    colors = [ATTR_COLORS.get(c, "#9e9e9e") for c in cats]

    _active_cls = get_xf("delay_class")
    _pull = pie_pull(cats, _active_cls) or [0] * len(cats)

    donut  = go.Figure(go.Pie(
        labels=cats, values=counts, hole=0.52,
        pull=_pull,
        marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    donut.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=360, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        annotations=[dict(
            text=f"<b>{total_delayed:,}</b><br>delayed",
            x=0.5, y=0.5, font=dict(size=13, color=FC), showarrow=False,
        )],
    )
    _attr_donut_evt = st.plotly_chart(donut, use_container_width=True,
                                      on_select="rerun", key="xf_attr_donut")
    handle_selection(_attr_donut_evt, "delay_class", axis="label")

with col_r:
    st.markdown("#### Average Delay by Attribution Category")
    avg_by_cat = (delayed_only.groupby("attribution")["Target_Departure_Delay_mins"]
                  .mean().clip(upper=120).reset_index()
                  .rename(columns={"Target_Departure_Delay_mins": "Avg Delay (min)",
                                   "attribution": "Category"})
                  .sort_values("Avg Delay (min)", ascending=True))

    if not avg_by_cat.empty:
        bar_colors = [ATTR_COLORS.get(c, "#9e9e9e") for c in avg_by_cat["Category"]]
        bar_fig = go.Figure(go.Bar(
            y=avg_by_cat["Category"],
            x=avg_by_cat["Avg Delay (min)"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f} min" for v in avg_by_cat["Avg Delay (min)"]],
            textposition="outside",
            textfont=dict(size=11),
        ))
        bar_fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Average Departure Delay (min, capped at 120)",
                       range=[0, avg_by_cat["Avg Delay (min)"].max() * 1.3]),
            yaxis=dict(title=""),
            height=360, margin=dict(t=10, b=40, l=20, r=90),
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

st.divider()

# ── Inbound Arrival Impact Table ──────────────────────────────────────────────
st.markdown("### Inbound Arrival Condition vs Departure Performance")
insight_strip(
    "How does the inbound aircraft's punctuality affect SATS's ability to depart on time? "
    "The table below segments all flights by how early or late the inbound arrived.",
    severity="blue",
)

if "Incoming_Delay_mins" in valid.columns and "Target_Departure_Delay_Class" in valid.columns:
    _v = valid.dropna(subset=["Incoming_Delay_mins"])
    _baseline_rate = (_v["Target_Departure_Delay_Class"] == "Delayed").mean() * 100

    _bands = [
        ("On-time (–5 to +10 min)",  (_v["Incoming_Delay_mins"] >= -5)  & (_v["Incoming_Delay_mins"] <= 10)),
        ("Late (> 10 min)",          _v["Incoming_Delay_mins"] > 10),
        ("Very late (> 30 min)",     _v["Incoming_Delay_mins"] > 30),
        ("Early (< –5 min)",         _v["Incoming_Delay_mins"] < -5),
    ]

    _rows = []
    _baseline_dr = None
    for label, mask in _bands:
        grp = _v[mask]
        if grp.empty:
            continue
        dr = (grp["Target_Departure_Delay_Class"] == "Delayed").mean() * 100
        if label.startswith("On-time"):
            _baseline_dr = dr
        delta = (f"{dr - _baseline_dr:+.1f}pp" if _baseline_dr is not None and not label.startswith("On-time") else "—")
        _rows.append({"Inbound Arrival Condition": label,
                      "Flights": f"{len(grp):,}",
                      "Dep. Delay Rate": f"{dr:.1f}%",
                      "vs Baseline": delta})

    if _rows:
        _tbl_df = pd.DataFrame(_rows)

        def _color_delta(val):
            if val == "—":
                return ""
            pp = float(val.replace("pp", "").replace("+", ""))
            if pp > 5:
                return "color: #e74c3c; font-weight: bold"
            if pp < -3:
                return "color: #2ecc71; font-weight: bold"
            return "color: #f39c12; font-weight: bold"

        def _color_dr(val):
            try:
                pct = float(val.replace("%", ""))
                if pct > 50:
                    return "color: #e74c3c; font-weight: bold"
                if pct < 38:
                    return "color: #2ecc71"
            except Exception:
                pass
            return ""

        styled = (
            _tbl_df.style
            .map(_color_delta, subset=["vs Baseline"])
            .map(_color_dr,    subset=["Dep. Delay Rate"])
            .set_properties(**{"text-align": "center"}, subset=["Flights", "Dep. Delay Rate", "vs Baseline"])
            .set_table_styles([
                {"selector": "thead th",
                 "props": [("background-color", "#0d2b4d"),
                           ("color", "white"),
                           ("font-weight", "bold"),
                           ("text-align", "center"),
                           ("padding", "8px 12px")]},
                {"selector": "tbody td",
                 "props": [("padding", "7px 12px"), ("border-bottom", "1px solid rgba(255,255,255,0.06)")]},
                {"selector": "tbody tr:nth-child(even)",
                 "props": [("background-color", "rgba(255,255,255,0.03)")]},
                {"selector": "tbody tr:hover",
                 "props": [("background-color", "rgba(77,159,255,0.08)")]},
            ])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ── ⚡ LIGHTNING WARNING IMPACT SECTION ───────────────────────────────────────
st.markdown("## ⚡ Lightning Warning Impact Analysis")
insight_strip(
    "Lightning warnings suspend ramp and field operations at Changi. "
    "The section below shows when LW events occurred, their overlap with flight ground windows, "
    "and their measurable impact on departure delays.",
    severity="amber",
)

# Check if we have LW data in the current filter scope
lw_flights = valid[valid["LW_Day_Had_Warning"] == 1].copy()
no_lw_flights = valid[valid["LW_Day_Had_Warning"] == 0].copy()
has_lw_coverage = len(lw_flights) > 0

if not has_lw_coverage:
    st.info(
        "⚡ No lightning warning data found for the current date filter. "
        f"LW data is available from **{lw_date_min}** to **{lw_date_max}**. "
        "Adjust the date filters to include that period."
    )
else:
    # ── KPI comparison: LW day vs non-LW day ─────────────────────────────────
    lw_delay_rate    = (lw_flights["Target_Departure_Delay_Class"] == "Delayed").mean() * 100
    no_lw_delay_rate = (no_lw_flights["Target_Departure_Delay_Class"] == "Delayed").mean() * 100 if len(no_lw_flights) > 0 else 0

    lw_avg_delay    = lw_flights[lw_flights["Target_Departure_Delay_Class"] == "Delayed"]["Target_Departure_Delay_mins"].clip(upper=120).mean()
    no_lw_avg_delay = no_lw_flights[no_lw_flights["Target_Departure_Delay_Class"] == "Delayed"]["Target_Departure_Delay_mins"].clip(upper=120).mean() if len(no_lw_flights) > 0 else 0

    lw_gw_flights = valid[valid["LW_Active_During_Ground_Time"] == 1]
    lw_gw_delay_rate = (lw_gw_flights["Target_Departure_Delay_Class"] == "Delayed").mean() * 100 if len(lw_gw_flights) > 0 else 0

    lk1, lk2, lk3, lk4 = st.columns(4)
    lk1.metric("Flights on LW Days",          f"{len(lw_flights):,}", help="Flights departing on a date with at least one active LW")
    lk2.metric("Delay Rate — LW Days",        f"{lw_delay_rate:.1f}%",
               delta=f"{lw_delay_rate - no_lw_delay_rate:+.1f}pp vs non-LW days",
               delta_color="inverse")
    lk3.metric("Avg Delay — LW Days (min)",   f"{lw_avg_delay:.1f}" if pd.notna(lw_avg_delay) else "—",
               delta=f"{lw_avg_delay - no_lw_avg_delay:+.1f} min vs non-LW" if pd.notna(lw_avg_delay) and pd.notna(no_lw_avg_delay) else None,
               delta_color="inverse")
    lk4.metric("LW Active During Ground Time", f"{len(lw_gw_flights):,} flights",
               help="LW window directly overlapped the aircraft's ground-handling window")

    st.markdown("")

    # ── LW vs non-LW delay class comparison chart ─────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Delay Class — LW Day vs Non-LW Day")
        lw_cls   = ["On-Time", "Acceptable", "Delayed"]
        lw_pcts  = [(lw_flights["Target_Departure_Delay_Class"] == c).mean() * 100 for c in lw_cls]
        nlw_pcts = [(no_lw_flights["Target_Departure_Delay_Class"] == c).mean() * 100 for c in lw_cls] if len(no_lw_flights) else [0] * 3

        grp_fig = go.Figure()
        grp_fig.add_trace(go.Bar(name="LW Day", x=lw_cls, y=lw_pcts,
                                  marker_color="#f1c40f", text=[f"{v:.1f}%" for v in lw_pcts],
                                  textposition="outside", textfont=dict(color=FC)))
        grp_fig.add_trace(go.Bar(name="Non-LW Day", x=lw_cls, y=nlw_pcts,
                                  marker_color="#2980b9", text=[f"{v:.1f}%" for v in nlw_pcts],
                                  textposition="outside", textfont=dict(color=FC)))
        grp_fig.update_layout(
            template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="group", yaxis=dict(title="% of Flights", range=[0, 100]),
            height=320, margin=dict(t=20, b=40, l=50, r=20),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(grp_fig, use_container_width=True)

    with c2:
        st.markdown("#### Avg Departure Delay by LW Overlap")
        valid2 = valid.dropna(subset=["Target_Departure_Delay_mins"]).copy()
        valid2["LW_Overlap_Bin"] = pd.cut(
            valid2["LW_Overlap_With_Ground_Window_Mins"],
            bins=[-0.01, 0, 10, 20, 9999],
            labels=["0 min (no overlap)", "1–10 min", "11–20 min", ">20 min"],
        )
        ov_stats = (valid2.groupby("LW_Overlap_Bin", observed=True)["Target_Departure_Delay_mins"]
                    .agg(avg="mean", cnt="count").reset_index())

        ov_fig = go.Figure(go.Bar(
            x=ov_stats["LW_Overlap_Bin"].astype(str),
            y=ov_stats["avg"].clip(upper=60),
            marker_color=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
            text=[f"{v:.1f} min<br>({n:,} flights)" for v, n in zip(ov_stats["avg"], ov_stats["cnt"])],
            textposition="outside",
            textfont=dict(color=FC, size=10),
        ))
        ov_fig.update_layout(
            template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="LW overlap with ground-handling window"),
            yaxis=dict(title="Avg Departure Delay (min)"),
            height=320, margin=dict(t=20, b=60, l=60, r=20), showlegend=False,
        )
        st.plotly_chart(ov_fig, use_container_width=True)

    # ── Monthly LW frequency & delay rate trend ────────────────────────────────
    st.markdown("#### Monthly: LW Frequency vs Delay Rate")
    insight_strip(
        "Months with more lightning warning days typically show higher departure delay rates. "
        "The left axis shows delay rate (%), the bars show active LW minutes that month.",
        severity="blue",
    )

    if "_dep_month" in valid.columns and "_dep_year" in valid.columns:
        monthly = (valid.groupby(["_dep_year", "_dep_month"])
                   .agg(
                       delay_rate=("Target_Departure_Delay_Class", lambda x: (x == "Delayed").mean() * 100),
                       total_lw_mins=("Total_LW_Mins_On_Date", "mean"),
                       lw_days=("LW_Day_Had_Warning", "sum"),
                       n=("Target_Departure_Delay_Class", "count"),
                   ).reset_index())
        monthly["label"] = monthly.apply(
            lambda r: f"{int(r['_dep_year'])}-{int(r['_dep_month']):02d}", axis=1
        )

        if len(monthly) >= 2:
            mon_fig = go.Figure()
            mon_fig.add_trace(go.Bar(
                x=monthly["label"], y=monthly["total_lw_mins"],
                name="Avg LW Mins / Day", marker_color="rgba(241,196,15,0.55)",
                yaxis="y2",
            ))
            mon_fig.add_trace(go.Scatter(
                x=monthly["label"], y=monthly["delay_rate"],
                name="Delay Rate (%)", mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=7),
            ))
            mon_fig.update_layout(
                template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Delay Rate (%)", range=[0, 100]),
                yaxis2=dict(title="Avg LW Mins on LW Days", overlaying="y", side="right",
                            showgrid=False),
                height=350, margin=dict(t=20, b=60, l=60, r=80),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(mon_fig, use_container_width=True)

    # ── Per-hour LW impact ─────────────────────────────────────────────────────
    st.markdown("#### Departure Hour: Delay Rate with LW Active vs Not")
    insight_strip(
        "During hours when a lightning warning was active at the scheduled departure time, "
        "the delay rate is consistently higher — showing direct operational impact.",
        severity="amber",
    )

    if "Hour_of_Day" in valid.columns:
        hour_lw   = valid[valid["LW_Active_At_Sched_Departure"] == 1].groupby("Hour_of_Day").agg(
            delay_rate=("Target_Departure_Delay_Class", lambda x: (x == "Delayed").mean() * 100),
            n=("Target_Departure_Delay_Class", "count"),
        ).reset_index()
        hour_nlw  = valid[valid["LW_Active_At_Sched_Departure"] == 0].groupby("Hour_of_Day").agg(
            delay_rate=("Target_Departure_Delay_Class", lambda x: (x == "Delayed").mean() * 100),
            n=("Target_Departure_Delay_Class", "count"),
        ).reset_index()

        hour_fig = go.Figure()
        if not hour_nlw.empty:
            hour_fig.add_trace(go.Bar(
                x=hour_nlw["Hour_of_Day"], y=hour_nlw["delay_rate"],
                name="No LW at Departure", marker_color="rgba(41,128,185,0.7)",
            ))
        if not hour_lw.empty:
            hour_fig.add_trace(go.Bar(
                x=hour_lw["Hour_of_Day"], y=hour_lw["delay_rate"],
                name="⚡ LW Active at Departure", marker_color="rgba(241,196,15,0.9)",
            ))
        hour_fig.update_layout(
            template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="overlay",
            xaxis=dict(title="UTC Departure Hour", dtick=1),
            yaxis=dict(title="Delay Rate (%)", range=[0, 100]),
            height=320, margin=dict(t=20, b=60, l=60, r=20),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(hour_fig, use_container_width=True)

    # ── LW-attributed flight table ─────────────────────────────────────────────
    lw_attributed = delayed_only[delayed_only["attribution"].str.contains("LW", na=False)].copy()
    if len(lw_attributed) > 0:
        st.markdown(f"#### ⚡ {len(lw_attributed)} Delayed Flights with Active LW — Flight List")
        insight_strip(
            f"These <b>{len(lw_attributed)}</b> flights had an active lightning warning during their "
            "ground-handling window or at scheduled departure. They are classified as weather-driven "
            "and should be excluded from SATS operational SLA calculations.",
            severity="amber",
        )
        show_cols = [c for c in [
            "identification_carrierCode", "identification_iata",
            "departure_offBlock.scheduled", "origin_terminal", "destination_iata",
            "Target_Departure_Delay_mins", "attribution",
            "LW_Active_During_Ground_Time", "LW_Overlap_With_Ground_Window_Mins", "LW_Active_At_Sched_Departure",
        ] if c in lw_attributed.columns]
        disp = lw_attributed[show_cols].copy()
        disp.rename(columns={
            "identification_carrierCode": "Carrier",
            "identification_iata":        "Flight",
            "departure_offBlock.scheduled": "Sched Dep (UTC)",
            "origin_terminal":            "Terminal",
            "destination_iata":           "Dest",
            "Target_Departure_Delay_mins": "Delay (min)",
            "attribution":                "Attribution",
            "LW_Active_During_Ground_Time":        "LW in GW",
            "LW_Overlap_With_Ground_Window_Mins":     "LW Overlap (min)",
            "LW_Active_At_Sched_Departure":     "LW at Dep",
        }, inplace=True)
        if "Delay (min)" in disp.columns:
            disp["Delay (min)"] = disp["Delay (min)"].round(1)
        st.dataframe(disp.sort_values("Delay (min)", ascending=False).reset_index(drop=True),
                     use_container_width=True, height=300)

st.divider()

# ── SATS-attributable drill-down ───────────────────────────────────────────────
st.markdown("### 🔴 SATS-Attributable Delays — Root Cause Breakdown")
st.caption("Among flights where SATS is responsible, which specific activities are contributing most?")

sats_flights = delayed_only[delayed_only["attribution"] == "SATS Operation"]

if sats_flights.empty:
    st.info("No SATS-attributable delays in the current filter selection.")
else:
    sats_milestone_cols = [
        c for c in df.columns
        if "_analysis_Delay_mins" in c
        and not any(t in c for t in ["ActualDuration", "PlannedDuration"])
        and any(team in c for team in ["ramp_", "pax_", "aic_", "cabinSvc_",
                                       "security_", "baggage_", "loadControl_"])
    ]

    milestone_avgs = []
    for col in sats_milestone_cols:
        if col not in sats_flights.columns:
            continue
        vals = sats_flights[col].dropna()
        if len(vals) < 10:
            continue
        late_rate = (vals > 0).mean() * 100
        avg_d     = vals[vals > 0].mean() if (vals > 0).any() else 0.0
        name = (col.replace("milestone_", "")
                   .replace("_analysis_Delay_mins", "")
                   .replace("_", " ").title())
        milestone_avgs.append({
            "Activity":        name,
            "% Late":          late_rate,
            "Avg Delay (min)": avg_d,
            "Flights":         len(vals),
        })

    if milestone_avgs:
        m_df = pd.DataFrame(milestone_avgs).sort_values("% Late", ascending=False).head(15)
        insight_strip(
            f"⚡ Top SATS contributor: <b>{m_df.iloc[0]['Activity']}</b> — late on "
            f"{m_df.iloc[0]['% Late']:.0f}% of SATS-attributable flights, "
            f"{m_df.iloc[0]['Avg Delay (min)']:.1f} min avg overrun.",
            severity="red",
        )
        m_fig = go.Figure(go.Bar(
            y=m_df["Activity"], x=m_df["% Late"], orientation="h",
            marker=dict(color=m_df["% Late"],
                        colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
                        showscale=False),
            text=[f"{v:.0f}% late · {d:.1f} min avg" for v, d in
                  zip(m_df["% Late"], m_df["Avg Delay (min)"])],
            textposition="outside", textfont=dict(size=10),
        ))
        m_fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="% of SATS-attributable delayed flights where this activity ran late",
                       range=[0, m_df["% Late"].max() * 1.35]),
            yaxis=dict(title=""),
            height=max(300, len(m_df) * 30 + 60),
            margin=dict(t=10, b=40, l=20, r=200), showlegend=False,
        )
        st.plotly_chart(m_fig, use_container_width=True)
    else:
        st.info("Milestone delay data not available for SATS-attributable flights.")

st.divider()

# ── Attribution by Carrier ─────────────────────────────────────────────────────
st.markdown("### Attribution by Carrier")
insight_strip(
    "Each carrier's delay mix may differ. A carrier with many 'Propagated' delays may have poor "
    "inbound connections. One with many '⚡ LW' delays may operate on routes more exposed to "
    "Singapore's afternoon thunderstorm season.",
    severity="blue",
)

if "identification_carrierCode" in delayed_only.columns:
    carrier_attr  = (delayed_only.groupby(["identification_carrierCode", "attribution"])
                     .size().reset_index(name="count"))
    carrier_total = delayed_only.groupby("identification_carrierCode").size().reset_index(name="total")
    carrier_attr  = carrier_attr.merge(carrier_total, on="identification_carrierCode")
    carrier_attr["pct"] = carrier_attr["count"] / carrier_attr["total"] * 100

    big_carriers = carrier_total[carrier_total["total"] >= 20]["identification_carrierCode"]
    carrier_attr = carrier_attr[carrier_attr["identification_carrierCode"].isin(big_carriers)]

    if not carrier_attr.empty:
        st.caption("Click a bar segment to cross-filter all charts by carrier.")
        attr_order = [
            "Weather (Lightning Warning)", "SATS Operation",
            "Propagated (inbound late)", "Tight Schedule (planning)",
            "Compound (LW + Propagated)", "Compound (LW + Tight Schedule)",
            "Compound (propagated + tight schedule)", "Unknown / Insufficient Data",
        ]

        _active_carrier = get_xf("carrier")
        _all_carriers = carrier_attr["identification_carrierCode"].unique().tolist()
        _carrier_opacities = {
            c: 1.0 if (_active_carrier is None or str(c) == str(_active_carrier)) else 0.25
            for c in _all_carriers
        }

        c_fig = go.Figure()
        for attr_cat in attr_order:
            subset = carrier_attr[carrier_attr["attribution"] == attr_cat]
            if subset.empty:
                continue
            c_fig.add_trace(go.Bar(
                x=subset["identification_carrierCode"],
                y=subset["pct"],
                name=attr_cat,
                marker=dict(
                    color=ATTR_COLORS.get(attr_cat, "#9e9e9e"),
                    opacity=[_carrier_opacities.get(c, 1.0) for c in subset["identification_carrierCode"]],
                ),
                hovertemplate="%{x}: %{y:.1f}%<extra>" + attr_cat + "</extra>",
            ))
        c_fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            barmode="stack", height=420,
            xaxis_title="Carrier", yaxis_title="% of Carrier's Delayed Flights",
            margin=dict(t=20, b=60, l=50, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        _carrier_attr_evt = st.plotly_chart(c_fig, use_container_width=True,
                                            on_select="rerun", key="xf_attr_carrier")
        handle_selection(_carrier_attr_evt, "carrier", axis="x")

st.divider()

# ── Future Framework & LW SLA Exclusion ──────────────────────────────────────
st.markdown("### 🔮 Future Attribution & LW SLA Exclusion Policy")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"""
    <div style="background:rgba(241,196,15,0.08);border:1px solid rgba(241,196,15,0.30);
                border-radius:12px;padding:18px 20px">
      <div style="font-size:0.8rem;font-weight:700;color:#f1c40f;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:10px">⚡ Lightning Warning SLA Exclusion</div>
      <div style="font-size:0.85rem;color:{card_sub()};line-height:1.8">
        • LW warnings are issued by Meteorological Service Singapore (MSS)<br>
        • During active LW, all ground staff must stop outdoor activities<br>
        • Ground time effectively freezes — delays are beyond SATS control<br>
        • <b>{lw_all_n:,} flights</b> ({lw_pct:.1f}%) in this period had LW-active ground windows<br>
        • Excluding LW delays reduces SATS-attributed delay rate from
          <b>{sats_pct:.0f}%</b> to <b>{sats_n/max(total_delayed-lw_all_n,1)*100:.0f}%</b> of remaining delayed flights
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div style="background:rgba(26,115,232,0.08);border:1px solid rgba(26,115,232,0.25);
                border-radius:12px;padding:18px 20px">
      <div style="font-size:0.8rem;font-weight:700;color:#4d9fff;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:10px">📡 Additional data needed</div>
      <div style="font-size:0.85rem;color:{card_sub()};line-height:1.8">
        • <b>ATC/ATFM slot</b> — flag delays caused by airspace flow management<br>
        • <b>Fuel start/end actuals</b> — fuelling overrun vs ground ops overrun<br>
        • <b>Crew sign-on time</b> — late crew boarding (airline responsibility)<br>
        • <b>MEL / technical release</b> — aircraft defect holds<br>
        • <b>Gate change events</b> — disruption by airport infrastructure<br>
        • <b>IATA delay codes</b> — standard per-flight cause codes
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(243,156,18,0.07);border:1px solid rgba(243,156,18,0.2);
            border-radius:10px;padding:14px 20px;margin-top:16px;font-size:0.85rem;line-height:1.7">
  <b style="color:#f39c12">⚠️ Current limitation:</b>
  Attribution uses available signals (incoming delay, ground time ratio, lightning warnings).
  Approximately <b>{unk_pct:.0f}%</b> of delayed flights fall into "Unknown" because the data
  needed to classify them is not yet collected. Adding even one new data source
  (e.g. IATA delay codes) would significantly reduce this unknown bucket.
</div>
""".format(unk_pct=unk_n / total_delayed * 100 if total_delayed > 0 else 0),
unsafe_allow_html=True)
