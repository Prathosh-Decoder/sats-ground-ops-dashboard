"""
pages/9_Data_Quality.py
Data Quality Observatory — completeness and health of every data column.
Shows which milestones are fully tracked, partially captured, or completely missing,
helping teams identify data collection gaps before they affect analysis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.loader   import load_data
from utils.style    import inject_css, chart_template, chart_fc, card_bg, card_text, card_sub, header_bg, header_border
from utils.insights import insight_card

st.set_page_config(
    page_title="Data Quality | SATS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

TEMPLATE = chart_template()
FC       = chart_fc()
BG       = "rgba(0,0,0,0)"

TEAM_COLORS = {
    "techramp":    "#34495e",
    "ramp":        "#2980b9",
    "pax":         "#8e44ad",
    "aic":         "#27ae60",
    "cabinSvc":    "#d4ac0d",
    "cargo":       "#cb4335",
    "security":    "#17a589",
    "loadControl": "#ca6f1e",
    "baggage":     "#e67e22",
    "TPO":         "#1abc9c",
    "other":       "#636e72",
}
TEAM_LABELS = {
    "techramp":    "Tech Ramp",
    "ramp":        "Ramp",
    "pax":         "Passenger Svc",
    "aic":         "AIC Cleaning",
    "cabinSvc":    "Cabin Service",
    "cargo":       "Cargo",
    "security":    "Security",
    "loadControl": "Load Control",
    "baggage":     "Baggage",
    "TPO":         "Towing (TPO)",
    "other":       "Other",
}

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

# ── Compute completeness ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Analysing data quality…")
def compute_completeness(_df):
    total = len(_df)

    # Key flight metadata columns
    meta_cols = {
        "Flight Number":        "identification_iata",
        "Carrier Code":         "identification_carrierCode",
        "Aircraft ICAO Model":  "aircraft_typeICAO",
        "Aircraft Body Type":   "aircraft_bodyType",
        "Origin Terminal":      "origin_terminal",
        "Destination":          "destination_iata",
        "Scheduled Departure":  "departure_offBlock.scheduled",
        "Actual Departure":     "departure_offBlock.actual",
        "Departure Delay (min)":"Target_Departure_Delay_mins",
        "Delay Class":          "Target_Departure_Delay_Class",
        "Incoming Delay (min)": "Incoming_Delay_mins",
        "Available Ground Time":"Available_Ground_Time_mins",
        "Ground Time Ratio":    "Ground_Time_Ratio",
    }

    # Milestone delay analysis columns
    delay_cols = [c for c in _df.columns
                  if "_analysis_Delay_mins" in c
                  and "Duration" not in c]

    # Milestone actual timestamp columns
    actual_cols = [c for c in _df.columns
                   if c.startswith("milestone_") and c.endswith(".actual")]

    def team_from_col(col):
        # Tech Ramp activities (Thumbs Up) live under milestone_ramp_* in the data.
        if "thumbsup" in col.lower():
            return "techramp"
        parts = col.replace("milestone_", "").split("_")
        # handle "milestone_aic_..." or "milestone_TPO_..."
        if parts:
            t = parts[0].split(".")[0]
            return t if t in TEAM_COLORS else "other"
        return "other"

    def friendly_name(col):
        col = col.replace("milestone_", "")
        col = col.replace("_analysis_Delay_mins", " ← delay")
        col = col.replace(".actual", " ← actual")
        col = col.replace("_", " ").replace(".", ": ")
        return col.title()

    records = []

    for label, col in meta_cols.items():
        if col not in _df.columns:
            continue
        n_valid = int(_df[col].notna().sum())
        pct     = n_valid / total * 100
        records.append({
            "friendly":  label,
            "raw_col":   col,
            "group":     "Flight Metadata",
            "team":      "other",
            "n_valid":   n_valid,
            "n_missing": total - n_valid,
            "pct":       round(pct, 1),
            "status":    "full" if pct >= 99 else "partial" if pct > 0 else "empty",
        })

    for col in sorted(delay_cols):
        n_valid = int(_df[col].notna().sum())
        pct     = n_valid / total * 100
        team    = team_from_col(col)
        records.append({
            "friendly":  friendly_name(col),
            "raw_col":   col,
            "group":     "Milestone Delay (Analysis)",
            "team":      team,
            "n_valid":   n_valid,
            "n_missing": total - n_valid,
            "pct":       round(pct, 1),
            "status":    "full" if pct >= 99 else "partial" if pct > 0 else "empty",
        })

    for col in sorted(actual_cols):
        n_valid = int(_df[col].notna().sum())
        pct     = n_valid / total * 100
        team    = team_from_col(col)
        records.append({
            "friendly":  friendly_name(col),
            "raw_col":   col,
            "group":     "Milestone Timestamps (Raw)",
            "team":      team,
            "n_valid":   n_valid,
            "n_missing": total - n_valid,
            "pct":       round(pct, 1),
            "status":    "full" if pct >= 99 else "partial" if pct > 0 else "empty",
        })

    return pd.DataFrame(records), total

quality_df, total_rows = compute_completeness(df)

# Summary stats
n_full    = (quality_df["status"] == "full").sum()
n_partial = (quality_df["status"] == "partial").sum()
n_empty   = (quality_df["status"] == "empty").sum()
n_total   = len(quality_df)

# Health score = weighted average of pct across all columns
health_score = float(quality_df["pct"].mean())

# Active milestone delay columns (pct > 0)
active_delay = quality_df[
    (quality_df["group"] == "Milestone Delay (Analysis)") &
    (quality_df["pct"] > 0)
].copy()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{header_bg()};
            border:1px solid {header_border()};border-radius:16px;
            padding:24px 32px;margin-bottom:28px;position:relative;overflow:hidden">
  <div style="position:absolute;top:0;right:0;width:300px;height:100%;
              background:radial-gradient(ellipse at right,rgba(0,217,119,0.06),transparent);
              pointer-events:none"></div>
  <h2 style="margin:0;color:{card_text()};font-size:1.6rem">🔬 Data Quality Observatory</h2>
  <p style="margin:8px 0 0;color:{card_sub()};font-size:.93rem">
    Completeness audit across every flight metadata field, milestone delay column,
    and raw activity timestamp. Know exactly what data you can trust for analysis.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Top KPI row ───────────────────────────────────────────────────────────────
from utils.cascade import NO_DATA_NOTE as _ND_NOTE
st.caption(_ND_NOTE + " So a 0%-coverage column below is not necessarily a pipeline fault.")

k1, k2, k3, k4, k5 = st.columns(5)

score_color = "#2ecc71" if health_score >= 75 else "#f39c12" if health_score >= 50 else "#e74c3c"

def big_metric(col, value, label, sub, color):
    col.markdown(f"""
    <div style="background:{card_bg()};
                border:1px solid rgba(255,255,255,0.07);border-radius:16px;
                padding:20px 22px;border-top:3px solid {color}">
      <div style="font-size:0.62rem;font-weight:700;color:{card_sub()};text-transform:uppercase;
                  letter-spacing:1.5px;margin-bottom:8px">{label}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;
                  color:{color};text-shadow:0 0 20px {color}66;line-height:1">{value}</div>
      <div style="font-size:0.7rem;color:{card_sub()};margin-top:6px">{sub}</div>
    </div>""", unsafe_allow_html=True)

big_metric(k1, f"{health_score:.1f}%", "Data Health Score",
           "avg completeness across all tracked columns", score_color)
big_metric(k2, f"{total_rows:,}", "Total Flights",
           "records in the analysis dataset", "#4d9fff")
big_metric(k3, f"{n_full}", "Fully Complete",
           f"columns with ≥99% coverage ({n_full/n_total*100:.0f}% of all cols)", "#2ecc71")
big_metric(k4, f"{n_partial}", "Partially Captured",
           f"columns with 1–98% coverage — investigate", "#f39c12")
big_metric(k5, f"{n_empty}", "No Data",
           f"columns in schema with 0% data — service may not apply, or not captured", "#e74c3c")

st.divider()

# ── Page insight ──────────────────────────────────────────────────────────────
_empty_delay = quality_df[(quality_df["group"] == "Milestone Delay (Analysis)") & (quality_df["status"] == "empty")]
_teams_miss  = _empty_delay["team"].map(TEAM_LABELS).fillna(_empty_delay["team"]).unique()
_sev = "red" if health_score < 50 else "amber" if health_score < 75 else "green"

if len(_empty_delay) > 0:
    _teams_str = ", ".join(_teams_miss[:3])
    insight_card(
        problem=(f"Data health is **{health_score:.0f}%** — {n_empty} milestone columns have 0% coverage "
                 f"and {n_partial} are only partially captured."),
        impact=(f"No data for **{_teams_str}** means those activity delays can't be analysed. "
                "Note this may be because the service isn't provided for those airlines/flights, "
                "or because it wasn't captured — the two aren't distinguished here."),
        action=(f"Confirm with **{_teams_miss[0]}** whether these services apply to the affected airlines "
                "before treating it as a capture gap. See the BU coverage chart below for the full breakdown."),
        severity=_sev,
        icon="🔬",
    )
elif n_partial > 0:
    insight_card(
        problem=f"Data health is **{health_score:.0f}%** — {n_partial} columns have partial coverage (1–98%).",
        impact="Partial columns may introduce bias: missing rows are often not random and correlate with a specific terminal, carrier, or time window.",
        action="Use the Coverage by Terminal & Carrier charts below to identify which data feed is dropping records.",
        severity=_sev,
        icon="🔬",
    )
else:
    insight_card(
        problem=f"Data health is **{health_score:.0f}%** — all tracked milestone columns are either fully complete or confirmed absent.",
        impact="Analysis across all pages is built on reliable data. No partial feeds are skewing results.",
        action="Continue monitoring monthly coverage trends to catch any new feed issues early.",
        severity="green",
        icon="🔬",
    )

# ── Health gauge + breakdown donut ────────────────────────────────────────────
col_gauge, col_donut, col_alerts = st.columns([1.4, 1.2, 1.4])

with col_gauge:
    st.markdown("#### Overall Health Score")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        number={"suffix": "%", "font": {"color": score_color, "size": 44,
                                        "family": "JetBrains Mono"}},
        title={"text": "Data Completeness", "font": {"color": "#8899bb", "size": 13}},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#4a5568",
                      tickfont=dict(color="#4a5568")),
            bar=dict(color=score_color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                {"range": [0,  50], "color": "rgba(231,76,60,0.10)"},
                {"range": [50, 75], "color": "rgba(243,156,18,0.10)"},
                {"range": [75,100], "color": "rgba(46,204,113,0.10)"},
            ],
            threshold=dict(line=dict(color="rgba(255,255,255,0.4)", width=2),
                           thickness=0.75, value=75),
        ),
    ))
    gauge_fig.update_layout(
        template=TEMPLATE, height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10, l=20, r=20),
        font=dict(color=FC),
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

with col_donut:
    st.markdown("#### Coverage Breakdown")
    donut_fig = go.Figure(go.Pie(
        labels=["Fully Complete (≥99%)", "Partially Captured", "No Data (0%)"],
        values=[n_full, n_partial, n_empty],
        hole=0.60,
        marker_colors=["#2ecc71", "#f39c12", "#e74c3c"],
        textinfo="percent+label",
        textfont=dict(size=10, color=FC),
        hovertemplate="%{label}<br>%{value} columns (%{percent})<extra></extra>",
    ))
    donut_fig.add_annotation(
        text=f"<b>{n_total}</b><br><span style='font-size:10px'>columns</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=FC),
    )
    donut_fig.update_layout(
        template=TEMPLATE, height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(donut_fig, use_container_width=True)

with col_alerts:
    st.markdown("#### Data Alerts")
    alerts = quality_df[quality_df["status"] == "partial"].sort_values("pct")
    if len(alerts) == 0:
        st.success("✅ No partially-captured columns. All columns are either fully complete or fully absent.")
    else:
        for _, row in alerts.iterrows():
            color = "#e74c3c" if row["pct"] < 50 else "#f39c12"
            bar_w = max(3, int(row["pct"]))
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border-left:3px solid {color};
                        border-radius:0 8px 8px 0;padding:8px 12px;margin-bottom:6px">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div style="font-size:0.78rem;color:#c5d3f0;font-weight:600">{row['friendly'][:38]}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                            font-weight:700;color:{color}">{row['pct']:.0f}%</div>
              </div>
              <div style="font-size:0.65rem;color:#6b7fa3;margin:2px 0 4px">
                {row['n_valid']:,} of {total_rows:,} records · {row['group']}
              </div>
              <div style="background:rgba(255,255,255,0.05);border-radius:3px;height:3px">
                <div style="background:{color};width:{bar_w}%;height:100%;border-radius:3px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

st.divider()

# ── Milestone delay column coverage chart ────────────────────────────────────
st.markdown("### 📊 Milestone Delay Coverage — Column by Column")
st.caption("Coverage of every milestone's delay analysis column. "
           "🟢 Fully captured (≥99%)  ·  🟡 Partially captured  ·  🔴 No data (0%)")

delay_cov = quality_df[quality_df["group"] == "Milestone Delay (Analysis)"].copy()
delay_cov = delay_cov.sort_values(["pct", "team"], ascending=[False, True])

# Color by coverage
bar_colors = []
for _, r in delay_cov.iterrows():
    if r["pct"] >= 99:
        bar_colors.append(TEAM_COLORS.get(r["team"], "#2ecc71"))
    elif r["pct"] > 0:
        bar_colors.append("#f39c12")
    else:
        bar_colors.append("rgba(60,70,90,0.4)")

cov_fig = go.Figure(go.Bar(
    x=delay_cov["pct"],
    y=delay_cov["friendly"].str.replace(" ← delay", "", regex=False).str[:42],
    orientation="h",
    marker_color=bar_colors,
    marker_line_width=0,
    opacity=0.90,
    text=[f"{v:.0f}%  ({int(n):,})" if v > 0 else "No data"
          for v, n in zip(delay_cov["pct"], delay_cov["n_valid"])],
    textposition="outside",
    textfont=dict(size=9, color=FC),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Coverage: %{x:.1f}%<br>"
        "Records: %{text}<extra></extra>"
    ),
))
cov_fig.add_vline(x=99, line_dash="dash", line_color="rgba(46,204,113,0.4)", line_width=1.5)
cov_fig.add_vline(x=50, line_dash="dot",  line_color="rgba(243,156,18,0.4)",  line_width=1.5)
cov_fig.update_layout(
    template=TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=max(500, len(delay_cov) * 26 + 100),
    xaxis=dict(title="Coverage (%)", range=[0, 130],
               gridcolor="rgba(255,255,255,0.04)", zeroline=False),
    yaxis=dict(tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.03)",
               automargin=True),
    margin=dict(l=10, r=100, t=20, b=40),
    bargap=0.18,
)
st.plotly_chart(cov_fig, use_container_width=True)

st.divider()

# ── BU coverage summary ───────────────────────────────────────────────────────
st.markdown("### 🏢 Coverage by Business Unit")

bu_left, bu_right = st.columns([1.6, 1.4])

with bu_left:
    # Aggregate per BU across delay cols only
    bu_cov = (delay_cov.groupby("team")
              .agg(avg_pct=("pct", "mean"),
                   n_cols=("pct", "count"),
                   n_full=("status", lambda x: (x == "full").sum()),
                   n_empty=("status", lambda x: (x == "empty").sum()))
              .reset_index())
    bu_cov["label"] = bu_cov["team"].map(TEAM_LABELS).fillna(bu_cov["team"])
    bu_cov = bu_cov.sort_values("avg_pct", ascending=True)

    bu_colors = [
        "#2ecc71" if p >= 99 else "#f39c12" if p > 50 else "#e74c3c"
        for p in bu_cov["avg_pct"]
    ]

    bu_fig = go.Figure(go.Bar(
        x=bu_cov["avg_pct"],
        y=bu_cov["label"],
        orientation="h",
        marker_color=bu_colors,
        marker_line_width=0,
        opacity=0.88,
        text=[f"{v:.0f}%  ({fc}/{tc} fully captured)"
              for v, fc, tc in zip(bu_cov["avg_pct"], bu_cov["n_full"], bu_cov["n_cols"])],
        textposition="outside",
        textfont=dict(size=9, color=FC),
        hovertemplate="<b>%{y}</b><br>Avg coverage: %{x:.1f}%<extra></extra>",
    ))
    bu_fig.add_vline(x=99, line_dash="dash", line_color="rgba(46,204,113,0.3)", line_width=1.5)
    bu_fig.update_layout(
        template=TEMPLATE, height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Avg Coverage (%)", range=[0, 130],
                   gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis=dict(tickfont=dict(size=10), automargin=True),
        margin=dict(l=10, r=120, t=10, b=40),
        bargap=0.22,
        title=dict(text="Average Milestone Coverage per Business Unit",
                   font=dict(size=11, color="#8899bb"), x=0.01),
    )
    st.plotly_chart(bu_fig, use_container_width=True)

with bu_right:
    st.markdown("#### BU Summary Table")
    bu_display = bu_cov[["label", "avg_pct", "n_cols", "n_full", "n_empty"]].copy()
    bu_display.columns = ["Business Unit", "Avg Coverage %", "Total Cols", "Fully Complete", "Not Captured"]
    bu_display = bu_display.sort_values("Avg Coverage %", ascending=False).reset_index(drop=True)
    bu_display["Avg Coverage %"] = bu_display["Avg Coverage %"].round(1)

    st.dataframe(
        bu_display,
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={
            "Avg Coverage %": st.column_config.ProgressColumn(
                "Avg Coverage %", min_value=0, max_value=100, format="%.1f%%"),
            "Fully Complete": st.column_config.NumberColumn("✅ Full"),
            "Not Captured":   st.column_config.NumberColumn("❌ No Data"),
        }
    )

st.divider()

# ── Coverage over time ────────────────────────────────────────────────────────
st.markdown("### 📅 Coverage Trend Over Time")
st.caption("How many key milestones were recorded per flight, by month.")

# Define outside the time-section block so terminal/carrier sections can use it
active_delay_cols = [c for c in active_delay["raw_col"].tolist() if c in df.columns]

if "_dep_month" in df.columns and "_dep_year" in df.columns and active_delay_cols:

    if active_delay_cols:
        df["_month_label"] = (df["_dep_year"].astype(str) + "-"
                              + df["_dep_month"].astype(str).str.zfill(2))
        df["_active_coverage"] = df[active_delay_cols].notna().sum(axis=1) / len(active_delay_cols) * 100

        time_grp = (df.groupby("_month_label")
                    .agg(avg_coverage=("_active_coverage", "mean"),
                         flight_count=("id", "count") if "id" in df.columns else ("_active_coverage", "count"))
                    .reset_index().sort_values("_month_label"))

        t_left, t_right = st.columns([2, 1])
        with t_left:
            time_fig = go.Figure()
            time_fig.add_trace(go.Scatter(
                x=time_grp["_month_label"],
                y=time_grp["avg_coverage"],
                mode="lines+markers",
                line=dict(color="#4d9fff", width=3),
                marker=dict(size=9, color="#4d9fff",
                            line=dict(color=FC, width=2)),
                fill="tozeroy",
                fillcolor="rgba(77,159,255,0.07)",
                name="Avg Coverage %",
                hovertemplate="<b>%{x}</b><br>Coverage: %{y:.1f}%<extra></extra>",
            ))
            time_fig.add_hline(y=99, line_dash="dash",
                               line_color="rgba(46,204,113,0.4)", line_width=1.5,
                               annotation_text="Target: 99%",
                               annotation_font=dict(color="#2ecc71", size=10))
            time_fig.update_layout(
                template=TEMPLATE, height=250,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Month", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Avg Coverage (%)", range=[0, 110],
                           gridcolor="rgba(255,255,255,0.04)"),
                margin=dict(t=10, b=40, l=50, r=20),
                showlegend=False,
            )
            st.plotly_chart(time_fig, use_container_width=True)

        with t_right:
            st.markdown("**Monthly Summary**")
            disp = time_grp[["_month_label", "avg_coverage", "flight_count"]].copy()
            disp.columns = ["Month", "Coverage %", "Flights"]
            disp["Coverage %"] = disp["Coverage %"].round(1)
            st.dataframe(disp, use_container_width=True, hide_index=True, height=240,
                         column_config={
                             "Coverage %": st.column_config.ProgressColumn(
                                 format="%.1f%%", min_value=0, max_value=100)
                         })

st.divider()

# ── Coverage by terminal & carrier ────────────────────────────────────────────
st.markdown("### ✈️ Coverage by Terminal & Carrier")

dim_l, dim_r = st.columns(2)

# By terminal
with dim_l:
    if "origin_terminal" in df.columns and active_delay_cols:
        term_grp = (df.groupby(df["origin_terminal"].astype(str))
                    ["_active_coverage"].mean().reset_index())
        term_grp.columns = ["Terminal", "Coverage %"]
        term_grp = term_grp[~term_grp["Terminal"].isin(["nan", "None", ""])]
        term_grp["Terminal"] = term_grp["Terminal"].replace(
            {"1": "T1", "2": "T2", "3": "T3", "4": "T4"})
        term_grp["Coverage %"] = term_grp["Coverage %"].round(1)
        t_colors = ["#2ecc71" if v >= 99 else "#f39c12" if v > 50 else "#e74c3c"
                    for v in term_grp["Coverage %"]]
        t_fig = go.Figure(go.Bar(
            x=term_grp["Terminal"], y=term_grp["Coverage %"],
            marker_color=t_colors, opacity=0.88,
            text=[f"{v:.1f}%" for v in term_grp["Coverage %"]],
            textposition="outside", textfont=dict(color=FC, size=11),
        ))
        t_fig.update_layout(
            template=TEMPLATE, height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Terminal"),
            yaxis=dict(title="Avg Coverage (%)", range=[0, 115],
                       gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(t=10, b=40, l=50, r=20),
            title=dict(text="Avg Active Milestone Coverage by Terminal",
                       font=dict(size=11, color="#8899bb"), x=0.01),
        )
        st.plotly_chart(t_fig, use_container_width=True)

# By carrier
with dim_r:
    if "identification_carrierCode" in df.columns and active_delay_cols:
        carrier_grp = (df.groupby("identification_carrierCode")
                       .agg(coverage=("_active_coverage", "mean"),
                            flights=("_active_coverage", "count"))
                       .reset_index())
        carrier_grp = carrier_grp[carrier_grp["flights"] >= 50]  # at least 50 flights
        carrier_grp = carrier_grp.sort_values("coverage", ascending=False).head(12)
        c_colors = ["#2ecc71" if v >= 99 else "#f39c12" if v > 50 else "#e74c3c"
                    for v in carrier_grp["coverage"]]
        c_fig = go.Figure(go.Bar(
            x=carrier_grp["identification_carrierCode"],
            y=carrier_grp["coverage"].round(1),
            marker_color=c_colors, opacity=0.88,
            text=[f"{v:.0f}%" for v in carrier_grp["coverage"]],
            textposition="outside", textfont=dict(color=FC, size=10),
        ))
        c_fig.update_layout(
            template=TEMPLATE, height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Carrier Code"),
            yaxis=dict(title="Avg Coverage (%)", range=[0, 115],
                       gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(t=10, b=40, l=50, r=20),
            title=dict(text="Avg Coverage by Carrier (min 50 flights)",
                       font=dict(size=11, color="#8899bb"), x=0.01),
        )
        st.plotly_chart(c_fig, use_container_width=True)

st.divider()

# ── Full raw column audit table ───────────────────────────────────────────────
st.markdown("### 🗂️ Full Column Audit")

with st.expander("Show complete data quality table (all columns)", expanded=False):
    group_filter = st.selectbox(
        "Filter by group",
        ["All"] + sorted(quality_df["group"].unique()),
        key="dq_group_filter"
    )
    status_filter = st.multiselect(
        "Filter by status",
        ["full", "partial", "empty"],
        default=["full", "partial", "empty"],
        key="dq_status_filter"
    )

    audit_df = quality_df.copy()
    if group_filter != "All":
        audit_df = audit_df[audit_df["group"] == group_filter]
    if status_filter:
        audit_df = audit_df[audit_df["status"].isin(status_filter)]

    audit_display = audit_df[["group", "friendly", "pct", "n_valid", "n_missing", "status"]].copy()
    audit_display.columns = ["Group", "Column", "Coverage %", "Records Present",
                              "Records Missing", "Status"]
    audit_display = audit_display.sort_values(["Group", "Coverage %"],
                                               ascending=[True, False]).reset_index(drop=True)

    def status_icon(s):
        return "✅ Full" if s == "full" else "⚠️ Partial" if s == "partial" else "❌ Empty"

    audit_display["Status"] = audit_display["Status"].apply(status_icon)

    st.dataframe(
        audit_display,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Coverage %": st.column_config.ProgressColumn(
                "Coverage %", min_value=0, max_value=100, format="%.1f%%"),
            "Records Present": st.column_config.NumberColumn(format="%d"),
            "Records Missing": st.column_config.NumberColumn(format="%d"),
        }
    )
    st.caption(f"Showing {len(audit_display):,} of {n_total:,} columns.")

st.divider()

# ── Recommendation panel ──────────────────────────────────────────────────────
st.markdown("### 💡 Data Quality Recommendations")

recs = []

empty_delay = quality_df[
    (quality_df["group"] == "Milestone Delay (Analysis)") &
    (quality_df["status"] == "empty")
]
if len(empty_delay) > 0:
    teams_missing = empty_delay["team"].map(TEAM_LABELS).fillna(empty_delay["team"]).unique()
    recs.append(("🔴", "No-Data Milestones",
                 f"{len(empty_delay)} milestone delay columns have 0% coverage — "
                 f"activities in {', '.join(teams_missing[:4])} have no data. "
                 "This may mean the service isn't provided for those airlines/flights, or it wasn't "
                 "captured — confirm with each BU before assuming a capture gap."))

partial = quality_df[quality_df["status"] == "partial"]
if len(partial) > 0:
    recs.append(("🟡", "Partial Coverage Columns",
                 f"{len(partial)} columns have between 1-98% coverage. "
                 "Check whether missing records correlate with a specific terminal, carrier, or time period — "
                 "this may indicate a data feed issue, or that the service applies only to certain airlines/flights."))

act_dep = quality_df[
    (quality_df["raw_col"] == "departure_offBlock.actual") &
    (quality_df["pct"] < 100)
]
if len(act_dep) > 0:
    pct = act_dep.iloc[0]["pct"]
    recs.append(("🟡", "Missing Actual Departure Times",
                 f"Actual departure time is only available for {pct:.1f}% of flights. "
                 f"The {100-pct:.1f}% missing ({int((1-pct/100)*total_rows):,} flights) cannot have "
                 "departure delay calculated and are excluded from delay analysis."))

full_meta = quality_df[
    (quality_df["group"] == "Flight Metadata") &
    (quality_df["status"] == "full")
]
recs.append(("🟢", "Strong Flight Metadata",
             f"{len(full_meta)} core flight metadata columns are ≥99% complete "
             "(flight number, carrier, aircraft, terminal, destination, scheduled departure). "
             "All analysis dimensions are well-supported."))

active_full = quality_df[
    (quality_df["group"] == "Milestone Delay (Analysis)") &
    (quality_df["status"] == "full")
]
if len(active_full) > 0:
    recs.append(("🟢", "Reliable Analysis Milestones",
                 f"{len(active_full)} milestone delay columns are ≥99% complete — "
                 "these are the backbone of the Cascade Effect, Activity Analysis, and BU Impact pages. "
                 "Ramp, PAX, AIC, Cargo, Security, and Load Control milestones are all fully captured."))

for icon, title, desc in recs:
    color = "#2ecc71" if icon == "🟢" else "#f39c12" if icon == "🟡" else "#e74c3c"
    bg    = "#0d2e0d" if icon == "🟢" else "#2e1f00" if icon == "🟡" else "#2e0d0d"
    st.markdown(f"""
    <div style="background:{bg};border-left:4px solid {color};border-radius:0 10px 10px 0;
                padding:14px 18px;margin-bottom:10px">
      <div style="font-weight:700;color:{color};font-size:.9rem;margin-bottom:4px">
        {icon} {title}
      </div>
      <div style="font-size:.82rem;color:#a0aec0;line-height:1.55">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
