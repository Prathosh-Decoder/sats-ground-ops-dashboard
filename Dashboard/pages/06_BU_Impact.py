"""
pages/8_BU_Impact.py
Combined BU Impact Analyser — Business Unit performance + Cascade position.
Pick a BU to see which of its activities are late and how they cascade downstream.
Pick an activity to see its delay distribution, time patterns, and regression impact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Ridge
import streamlit as st

from utils.loader       import load_data, render_date_filters
from utils.style        import inject_css, chart_template, chart_fc, chart_gc, is_light, card_bg, card_text, card_sub, header_bg, header_border
from utils.insights     import insight_card, insight_strip, no_data_metric
from utils.cascade      import (NODES, NODE_MAP, TEAM_COLORS, TEAM_LABELS, STRUCTURAL, ROUTES,
                                COLUMN_HEADERS, COLUMN_BANDS_Y, DEPARTURE_NODE)
from utils.cascade      import _wrap as _wrap_label
from utils.crossfilter  import init_xf, apply_xf, render_xf_bar

st.set_page_config(page_title="BU Impact | SATS", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
TEMPLATE = chart_template()
FC       = chart_fc()
BU_ICONS = {
    "techramp": "🛠️", "ramp": "🔩", "pax": "🧳", "aic": "🧹",
    "cabin": "💺", "cargo": "📦", "baggage": "🛄", "security": "🔒", "loadctrl": "⚖️",
}
# A team-level number is only reliable once the activities backing it have
# recorded enough flights combined — number of distinct activities doesn't
# matter (matches the same threshold used on Activity Analysis).
BU_MIN_TOTAL_FLIGHTS = 500

# ── Data ─────────────────────────────────────────────────────────────────────
df = load_data()
df = render_date_filters(df, page_key="bu_impact")
init_xf()
df = apply_xf(df)

# ── Session state ─────────────────────────────────────────────────────────────
if "bui_bu"       not in st.session_state: st.session_state["bui_bu"]       = list(TEAM_LABELS)[0]
if "bui_activity" not in st.session_state: st.session_state["bui_activity"] = None

# ── Activity stats ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing activity statistics…")
def compute_activity_stats(_df):
    dep_col = "Target_Departure_Delay_mins"
    stats = {}
    for name, team, col, *_ in NODES:
        if not col or col not in _df.columns:
            continue
        data = _df[[col, dep_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 50:
            continue
        vals = data[col]
        X = vals.values.reshape(-1, 1)
        y = data[dep_col].values
        reg = Ridge(alpha=10.0).fit(X, y)
        ss_res = np.sum((y - reg.predict(X)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(max(0., 1 - ss_res / ss_tot)) if ss_tot > 1 else 0.
        stats[name] = {
            "team":        team,
            "col":         col,
            "pct_late":    float((vals > 0).mean() * 100),
            "median_delay":float(vals.clip(lower=0).median()),
            "mean_delay":  float(vals.clip(lower=0).mean()),
            "coef":        float(np.asarray(reg.coef_).ravel()[0]),
            "r2":          r2,
            "n":           len(data),
        }
    return stats

activity_stats = compute_activity_stats(df)

# ── BU performance roll-up ────────────────────────────────────────────────────
bu_perf = {}
for team, label in TEAM_LABELS.items():
    nodes = [(n, s) for n, s in activity_stats.items() if s["team"] == team]
    if not nodes:
        continue
    _w = np.array([s["n"] for _, s in nodes], dtype=float)
    _wsum = _w.sum()
    bu_perf[team] = {
        "label":       label,
        "icon":        BU_ICONS.get(team, "⚙️"),
        "color":       TEAM_COLORS[team],
        # Flight-count-weighted — an activity with more recorded flights
        # should count for more than one with barely enough to qualify.
        "pct_late":    float(np.sum([s["pct_late"]   for _, s in nodes] * _w) / _wsum),
        "avg_delay":   float(np.sum([s["mean_delay"] for _, s in nodes] * _w) / _wsum),
        "avg_coef":    float(np.sum([s["coef"]        for _, s in nodes] * _w) / _wsum),
        "total_flights": int(_wsum),
        "activities":  nodes,
    }

# ── Flowchart ─────────────────────────────────────────────────────────────────
def build_combined_flowchart(selected_bu, selected_activity=None):
    """
    Flowchart where the selected BU's nodes are highlighted with their team colour
    and coloured by % late. All other nodes are dimmed.
    """
    fig = go.Figure()
    pos = {n[0]: (n[3], n[4]) for n in NODES}
    BW, BH = 0.57, 0.42

    _light = is_light()
    _node_normal_fill = "#f1f3f9" if _light else "#11162e"
    _node_normal_text = "#5a7090" if _light else "#7185a9"
    _node_normal_bdr  = "#d0d7e5" if _light else "#1c2445"
    _ref_fill  = "#d0e4ff" if _light else "#1a3a6e"
    _ref_text  = "#1a3a6e" if _light else "#cce5ff"
    _dep_fill  = "#d4f5e2" if _light else "#0d3b20"
    _dep_text  = "#1a2340" if _light else "#ffffff"
    _bu_text   = "#1a2340" if _light else "#ffffff"
    _title_clr = "#4a5a7a" if _light else "#8899bb"
    _edge_dim  = "rgba(90,100,130,0.30)" if _light else "rgba(90,100,130,0.18)"

    # Edges — routed through the shared dummy-waypoint splines (see utils/cascade.py)
    for (parent, child), waypts in ROUTES.items():
        if parent not in pos or child not in pos:
            continue
        c_team = NODE_MAP[child][1]
        p_team = NODE_MAP[parent][1]
        both_bu = (c_team == selected_bu and p_team == selected_bu)
        clr = TEAM_COLORS.get(selected_bu, "#e74c3c") if both_bu else _edge_dim
        wid = 2.5 if both_bu else 0.8
        (px_, py_), (cx, cy) = waypts[0], waypts[-1]
        fig.add_trace(go.Scatter(
            x=[px_, cx], y=[py_ - BH, cy + BH], mode="lines",
            line=dict(color=clr, width=wid),
            showlegend=False, hoverinfo="skip",
        ))

    shapes, annotations = [], []

    for name, team, col, x, y, *_ in NODES:
        is_selected_bu   = (team == selected_bu)
        is_sel_activity  = (name == selected_activity)
        is_struct        = name in STRUCTURAL
        stat             = activity_stats.get(name, {})
        pct_late         = stat.get("pct_late", 0)
        dash             = None

        if name == "Aircraft Arrives":
            fill, border, bw = _ref_fill, "#5dade2", 2
            fc    = _ref_text
            label = "<b>✈️ Aircraft Arrives</b><br>(AIBT · On-Chocks)"
        elif name == "✈️  DEPARTURE":
            fill, border, bw = _dep_fill, "#2ecc71", 2
            fc    = _dep_text
            label = "<b>✈️ PUSHBACK</b>"
        elif is_struct:
            # No-data process step — neutral dashed box, never a metric.
            fill, border, bw = _node_normal_fill, _node_normal_bdr, 1
            fc, dash = _node_normal_text, "dot"
            label = f"<b>{_wrap_label(name)}</b>"
        elif is_selected_bu:
            color = TEAM_COLORS[team]
            if pct_late > 40:
                fill   = "rgba(255,220,220,0.90)" if _light else "rgba(120,20,20,0.85)"
                border = "#e74c3c"
            elif pct_late > 20:
                fill   = "rgba(255,243,200,0.90)" if _light else "rgba(100,65,0,0.85)"
                border = "#f39c12"
            else:
                fill   = "rgba(200,255,225,0.90)" if _light else "rgba(10,50,30,0.85)"
                border = color
            bw = 3
            fc = _bu_text
            label = (f"<b>{_wrap_label(name)}</b>"
                     + (f"<br>{pct_late:.0f}% late" if pct_late > 0 else ""))
        else:
            fill   = _node_normal_fill
            border = _node_normal_bdr
            bw, fc = 1, _node_normal_text
            label  = f"<b>{_wrap_label(name)}</b>"

        # Glow for the specifically selected activity
        if is_sel_activity:
            shapes.append(dict(
                type="rect",
                x0=x - BW - 0.07, y0=y - BH - 0.07,
                x1=x + BW + 0.07, y1=y + BH + 0.07,
                fillcolor="rgba(255,255,255,0.07)", opacity=1.0,
                line=dict(color="#ffffff", width=5),
                xref="x", yref="y",
            ))
            border, bw = "#ffffff", 4

        shapes.append(dict(
            type="rect",
            x0=x - BW, y0=y - BH, x1=x + BW, y1=y + BH,
            fillcolor=fill, opacity=0.95 if (is_selected_bu and not is_struct) else 0.5,
            line=dict(color=border, width=bw, dash=dash),
            xref="x", yref="y",
        ))
        annotations.append(dict(
            x=x, y=y, text=label, showarrow=False,
            font=dict(size=7.6, color=fc, family="Inter, Arial"),
            xref="x", yref="y", align="center",
        ))

    # Swimlane column bands + BU headers (the selected BU's column is emphasised)
    def _faint(hexc, a):
        h = hexc.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

    _band_lo, _band_hi = COLUMN_BANDS_Y
    _band_shapes = []
    for _hlabel, _hx, _hy, _hcolor in COLUMN_HEADERS:
        _sel = (TEAM_LABELS.get(selected_bu, selected_bu) == _hlabel)
        _band_shapes.append(dict(
            type="rect", x0=_hx - 0.62, x1=_hx + 0.62, y0=_band_lo, y1=_band_hi,
            fillcolor=_faint(_hcolor, 0.20 if _sel else (0.08 if _light else 0.05)),
            line=dict(width=0), layer="below", xref="x", yref="y",
        ))
        annotations.append(dict(
            x=_hx, y=_hy, text=f"<b>{_hlabel}</b>", showarrow=False,
            font=dict(size=10.5, color=_hcolor if _sel else _node_normal_text,
                      family="Inter, Arial"),
            xref="x", yref="y",
        ))
    shapes = _band_shapes + shapes

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        template=TEMPLATE,
        height=1080,
        xaxis=dict(range=[0.0, 12.0], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[pos[DEPARTURE_NODE][1] - 0.85, _band_hi + 1.55],
                   showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=44, b=5),
        showlegend=False,
        title=dict(
            text=f"Ground Ops Flow  ·  <b style='color:{TEAM_COLORS.get(selected_bu,'#f39c12')}'>"
                 f"{TEAM_LABELS.get(selected_bu, selected_bu)} column highlighted</b>  ·  "
                 "🔴 >40% late  🟡 20-40%  🟢 <20%",
            font=dict(size=12, color=_title_clr), x=0.01, xanchor="left",
        ),
    )
    return fig


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{header_bg()};
            border-left:4px solid #a78bfa;padding:18px 24px;
            border-radius:0 12px 12px 0;margin-bottom:22px">
  <h2 style="margin:0;color:{card_text()}">🎯 BU Impact Analyser</h2>
  <p style="margin:8px 0 0;color:{card_sub()};font-size:.93rem">
    Select a Business Unit to see its position in the ground ops flow and
    which activities are driving departure delays.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Page Insight ─────────────────────────────────────────────────────────────
if bu_perf:
    _sorted = sorted(bu_perf.items(), key=lambda x: x[1]["pct_late"], reverse=True)
    _w_key, _w = _sorted[0]
    _b_key, _b = _sorted[-1]
    _worst_act_name = (max(_w["activities"], key=lambda x: x[1]["pct_late"])[0]
                       if _w["activities"] else "N/A")
    insight_card(
        problem=(f"**{_w['label']}** ({_w['icon']}) has the highest late rate at "
                 f"**{_w['pct_late']:.0f}%** of activities running behind schedule. "
                 f"Their worst activity in the flow is **{_worst_act_name}**."),
        impact=(f"Each activity late-rate point translates to cascading pressure on all downstream "
                f"nodes — especially ones with sequential dependencies like Load Control and PAX Boarding. "
                f"Best performer is **{_b['label']}** at {_b['pct_late']:.0f}% late."),
        action=(f"Select **{_w['label']}** below, then click **{_worst_act_name}** to see "
                "its delay distribution, regression impact, and peak overrun hours. "
                "Use that data to build a targeted SOP improvement."),
        icon="🎯", severity="red" if _w["pct_late"] > 45 else "amber",
    )

# ── BU selector cards ─────────────────────────────────────────────────────────
render_xf_bar()
st.markdown("### Business Units")
bu_keys = list(bu_perf.keys())

if not bu_keys:
    st.info("📊 Not enough data in the current filter to analyse any Business Unit (need 50+ flights per activity).")
    st.stop()

bu_cols = st.columns(len(bu_keys))

for bc, bu in zip(bu_cols, bu_keys):
    info    = bu_perf[bu]
    active  = st.session_state["bui_bu"] == bu
    color   = info["color"]
    _bu_total_card = info.get("total_flights", 0)
    _enough      = _bu_total_card >= BU_MIN_TOTAL_FLIGHTS
    pct_ok  = 100 - info["pct_late"] if _enough else 0
    bar_color = ("#e74c3c" if info["pct_late"] > 40 else "#f39c12" if info["pct_late"] > 20 else "#2ecc71") if _enough else "#6b7fa3"
    border_style = f"3px solid {color}" if active else f"1px solid {color}44"
    bg_style     = (f"rgba(0,0,0,0.08)" if is_light() else f"rgba(0,0,0,0.5)") if active else (
                    "rgba(0,0,0,0.03)" if is_light() else "rgba(255,255,255,0.02)")
    bar_track    = "rgba(0,0,0,0.08)" if is_light() else "rgba(255,255,255,0.06)"
    _metric_html = (
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.1rem;'
        f'font-weight:700;color:{bar_color}">{pct_ok:.0f}%</div>'
        f'<div style="font-size:0.62rem;color:{card_sub()};margin-bottom:6px">on-time</div>'
        f'<div style="background:{bar_track};border-radius:3px;height:3px">'
        f'<div style="background:{bar_color};width:{pct_ok:.0f}%;height:100%;border-radius:3px"></div></div>'
        if _enough else
        f'<div style="font-size:0.85rem;font-weight:700;color:{bar_color}">No data</div>'
        f'<div style="font-size:0.62rem;color:{card_sub()};margin-bottom:6px">'
        f'only {_bu_total_card:,} flights recorded</div>'
    )

    with bc:
        st.markdown(f"""
        <div style="background:{bg_style};border:{border_style};border-radius:14px;
                    padding:14px 16px;text-align:center;
                    box-shadow:{'0 0 16px ' + color + '44' if active else 'none'}">
          <div style="font-size:1.4rem">{info['icon']}</div>
          <div style="font-size:0.78rem;font-weight:700;color:{card_text()};
                      margin:4px 0 2px;letter-spacing:0.3px">{info['label']}</div>
          {_metric_html}
        </div>
        """, unsafe_allow_html=True)
        if st.button(
            f"{'✓ ' if active else ''}{info['label']}",
            key=f"bui_select_{bu}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state["bui_bu"]       = bu
            st.session_state["bui_activity"] = None
            st.rerun()

st.divider()

# ── Main two-column layout ────────────────────────────────────────────────────
selected_bu = st.session_state["bui_bu"]
bu_info     = bu_perf.get(selected_bu, {})
bu_activities = bu_info.get("activities", [])

left_col, right_col = st.columns([1.1, 2], gap="large")

# ── Left: activity list + selector ───────────────────────────────────────────
with left_col:
    color = bu_info.get("color", "#4d9fff")
    st.markdown(f"""
    <div style="border-left:3px solid {color};padding:4px 0 4px 14px;margin-bottom:16px">
      <div style="font-size:1rem;font-weight:800;color:{card_text()}">
        {bu_info.get('icon','')} {bu_info.get('label','')}
      </div>
      <div style="font-size:0.78rem;color:{card_sub()};margin-top:2px">
        {len(bu_activities)} tracked activities
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    k1, k2, k3 = st.columns(3)
    _bu_total = bu_info.get("total_flights", 0)
    if _bu_total < BU_MIN_TOTAL_FLIGHTS:
        with k1: no_data_metric("On-Time", _bu_total, min_n=BU_MIN_TOTAL_FLIGHTS)
        with k2: no_data_metric("Avg Delay", _bu_total, min_n=BU_MIN_TOTAL_FLIGHTS)
        with k3: no_data_metric("Dep Impact", _bu_total, min_n=BU_MIN_TOTAL_FLIGHTS)
    else:
        k1.metric("On-Time",   f"{100 - bu_info.get('pct_late', 0):.0f}%")
        k2.metric("Avg Delay", f"{bu_info.get('avg_delay', 0):.1f} min")
        k3.metric("Dep Impact", f"{bu_info.get('avg_coef', 0):.2f}x")

    st.markdown("#### Activities")
    st.caption("Select an activity to see its full analysis below.")

    sel_activity = st.session_state.get("bui_activity")

    for name, stat in sorted(bu_activities, key=lambda x: -x[1]["pct_late"]):
        pct = stat["pct_late"]
        bar_c = "#e74c3c" if pct > 40 else "#f39c12" if pct > 20 else "#2ecc71"
        active_act = sel_activity == name
        bg_act = f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12)" if active_act else "rgba(255,255,255,0.02)"

        _act_bdr = "rgba(0,0,0,0.10)" if (active_act and is_light()) else (
                   "rgba(255,255,255,0.12)" if active_act else (
                   "rgba(0,0,0,0.06)" if is_light() else "rgba(255,255,255,0.05)"))
        _bar_trk = "rgba(0,0,0,0.08)" if is_light() else "rgba(255,255,255,0.05)"
        st.markdown(f"""
        <div style="background:{bg_act};border:1px solid {_act_bdr};
                    border-radius:10px;padding:10px 14px;margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div style="font-size:0.82rem;font-weight:600;color:{card_text()}">{name}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                        font-weight:700;color:{bar_c}">{pct:.0f}%</div>
          </div>
          <div style="background:{_bar_trk};border-radius:3px;height:3px;margin-top:6px">
            <div style="background:{bar_c};width:{min(pct,100):.0f}%;height:100%;border-radius:3px"></div>
          </div>
          <div style="font-size:0.68rem;color:{card_sub()};margin-top:4px">
            avg {stat['mean_delay']:.1f} min late · {stat['n']:,} flights
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            f"{'▼ Viewing' if active_act else '▶ Analyse'} {name}",
            key=f"bui_act_{name.replace(' ','_').replace(':','_')}",
            use_container_width=True,
            type="primary" if active_act else "secondary",
        ):
            if active_act:
                st.session_state["bui_activity"] = None
            else:
                st.session_state["bui_activity"] = name
            st.rerun()

# ── Right: cascade flowchart ──────────────────────────────────────────────────
with right_col:
    sel_activity = st.session_state.get("bui_activity")
    st.plotly_chart(
        build_combined_flowchart(selected_bu, sel_activity),
        use_container_width=True,
    )

    # Downstream impact summary for selected BU
    if bu_info.get("total_flights", 0) >= BU_MIN_TOTAL_FLIGHTS:
        avg_coef = bu_info.get("avg_coef", 0)
        impact_color = "#e74c3c" if avg_coef > 0.5 else "#f39c12" if avg_coef > 0.2 else "#2ecc71"
        st.markdown(f"""
        <div style="background:{card_bg()};border-left:3px solid {impact_color};
                    border-radius:0 10px 10px 0;padding:12px 18px">
          <span style="font-size:0.78rem;color:{card_sub()}">
            On average, a 1-min delay in any <b style="color:{TEAM_COLORS.get(selected_bu,'#fff')}">{bu_info.get('label','')}</b> activity
            pushes departure back by
            <b style="color:{impact_color};font-size:1.1rem">{avg_coef:.2f} min</b>
            per minute of activity lateness (Ridge regression, all flights).
          </span>
        </div>
        """, unsafe_allow_html=True)
    elif bu_activities:
        st.caption(f"ℹ️ Not enough tracked activities ({len(bu_activities)}) for a reliable downstream-impact estimate.")

# ── Activity deep-dive (shown below when activity selected) ───────────────────
sel_activity = st.session_state.get("bui_activity")
if sel_activity and sel_activity in activity_stats:
    st.divider()
    stat    = activity_stats[sel_activity]
    col_key = stat["col"]
    color   = TEAM_COLORS.get(selected_bu, "#4d9fff")

    st.markdown(f"""
    <div style="background:{color}18;border-left:3px solid {color};
                border-radius:0 12px 12px 0;padding:14px 20px;margin-bottom:18px">
      <div style="font-size:1rem;font-weight:800;color:{card_text()}">
        {BU_ICONS.get(selected_bu,'')} {sel_activity} — Activity Analysis
      </div>
      <div style="font-size:0.78rem;color:{card_sub()};margin-top:4px">
        On average, this activity exceeded <b>PTS</b> by <b>{stat['mean_delay']:.1f} minutes</b> ·
        {stat['pct_late']:.1f}% of flights had this activity running late ·
        Ridge regression coef: {stat['coef']:.3f} · R² = {stat['r2']:.2f}
      </div>
    </div>
    """, unsafe_allow_html=True)

    chart1, chart2, chart3 = st.columns(3)

    # ── 1. Delay distribution ─────────────────────────────────────────────────
    with chart1:
        st.markdown("**Delay Distribution**")
        vals = df[col_key].dropna()
        vals_pos = vals[vals > 0].clip(upper=60)
        if len(vals_pos) > 10:
            _p50 = float(vals_pos.quantile(0.50))
            _p75 = float(vals_pos.quantile(0.75))
            _p90 = float(vals_pos.quantile(0.90))
            hist_fig = go.Figure(go.Histogram(
                x=vals_pos, nbinsx=30,
                marker_color=color, opacity=0.85,
                hovertemplate="Delay: %{x:.0f} min<br>Count: %{y}<extra></extra>",
            ))
            for _pv, _pc, _pl, _pos in [
                (_p50, "#2ecc71", "P50", "top left"),
                (_p75, "#f39c12", "P75", "top left"),
                (_p90, "#e74c3c", "P90", "top right"),
            ]:
                hist_fig.add_vline(
                    x=_pv, line_dash="dash", line_color=_pc, line_width=1.5,
                    annotation_text=f"{_pl}={_pv:.0f}m",
                    annotation_font_size=9, annotation_position=_pos,
                )
            hist_fig.update_layout(
                template=TEMPLATE, height=240,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Delay (min)", gridcolor=chart_gc()),
                yaxis=dict(title="Flights",      gridcolor=chart_gc()),
                margin=dict(t=30, b=40, l=40, r=10),
                bargap=0.05,
            )
            st.plotly_chart(hist_fig, use_container_width=True)

    # ── 2. Hour-of-day pattern ────────────────────────────────────────────────
    with chart2:
        st.markdown("**Hour-of-Day Pattern**")
        if "Hour_of_Day" in df.columns and col_key in df.columns:
            hour_df  = df[["Hour_of_Day", col_key]].dropna()
            hour_grp = hour_df.groupby("Hour_of_Day")[col_key].agg(
                PctLate=lambda x: (x > 0).mean() * 100,
                AvgDelay="mean", StdDelay="std",
                P90=lambda x: x.quantile(0.90), n="count"
            ).reset_index().rename(columns={"Hour_of_Day": "Hour"})
            bar_colors = ["#e74c3c" if v > 40 else "#f39c12" if v > 20 else color
                          for v in hour_grp["PctLate"]]
            hr_fig = go.Figure()
            hr_fig.add_trace(go.Bar(
                x=hour_grp["Hour"], y=hour_grp["AvgDelay"],
                name="Avg Delay",
                marker_color=bar_colors, opacity=0.88,
                error_y=dict(type="data", array=hour_grp["StdDelay"].fillna(0),
                             visible=True, color="rgba(180,180,180,0.55)",
                             thickness=1.5, width=3),
                customdata=np.column_stack([
                    hour_grp["PctLate"], hour_grp["StdDelay"].fillna(0)
                ]),
                hovertemplate="Hour %{x}:00<br>Avg: %{y:.1f} min<br>±1σ: %{customdata[1]:.1f} min<br>Late: %{customdata[0]:.0f}%<extra></extra>",
            ))
            hr_fig.add_trace(go.Scatter(
                x=hour_grp["Hour"], y=hour_grp["P90"],
                mode="lines+markers", name="P90",
                line=dict(color="#e74c3c", dash="dot", width=1.5),
                marker=dict(size=4),
                hovertemplate="Hour %{x}:00<br>P90: %{y:.1f} min<extra></extra>",
            ))
            hr_fig.update_layout(
                template=TEMPLATE, height=240,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Hour (UTC)", gridcolor=chart_gc(),
                           tickmode="linear", dtick=3),
                yaxis=dict(title="Delay (min)", gridcolor=chart_gc()),
                margin=dict(t=10, b=40, l=40, r=10),
                bargap=0.1,
                legend=dict(orientation="h", yanchor="top", y=-0.15,
                            xanchor="center", x=0.5, font=dict(size=9)),
            )
            st.plotly_chart(hr_fig, use_container_width=True)

    # ── 3. Correlation with departure delay ───────────────────────────────────
    with chart3:
        st.markdown("**Departure Delay Correlation**")
        dep_col = "Target_Departure_Delay_mins"
        if dep_col in df.columns and col_key in df.columns:
            scatter_df = df[[col_key, dep_col]].dropna()
            scatter_df = scatter_df.sample(
                min(800, len(scatter_df)), random_state=42
            )
            sc_fig = px.scatter(
                scatter_df, x=col_key, y=dep_col,
                opacity=0.35, trendline="ols",
                color_discrete_sequence=[color],
                template=TEMPLATE,
                labels={col_key: "Activity Delay (min)", dep_col: "Dep Delay (min)"},
            )
            sc_fig.update_layout(
                height=240,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[0, 40], gridcolor=chart_gc()),
                yaxis=dict(range=[-10, 80], gridcolor=chart_gc()),
                margin=dict(t=10, b=40, l=40, r=10),
                showlegend=False,
            )
            sc_fig.update_traces(marker_size=5, selector=dict(mode="markers"))
            sc_fig.update_traces(line_color=FC, selector=dict(type="scatter", mode="lines"))
            st.plotly_chart(sc_fig, use_container_width=True)

    # ── Impact summary ─────────────────────────────────────────────────────────
    ia, ib = st.columns(2)
    with ia:
        coef = stat["coef"]
        r2   = stat["r2"]
        imp_color = "#e74c3c" if coef > 0.6 else "#f39c12" if coef > 0.3 else "#2ecc71"
        st.markdown(f"""
        <div style="background:{card_bg()};border-radius:12px;padding:16px 20px">
          <div style="font-size:0.7rem;color:{card_sub()};text-transform:uppercase;
                      letter-spacing:1.2px;margin-bottom:8px">Regression Impact</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;
                      font-weight:700;color:{imp_color}">{coef:.3f}x</div>
          <div style="font-size:0.75rem;color:{card_sub()};margin-top:4px">
            min of departure delay per min of activity delay<br>
            R² = {r2:.2f} ({int(r2*100)}% confidence)
          </div>
        </div>
        """, unsafe_allow_html=True)
    with ib:
        pct = stat["pct_late"]
        med = stat["mean_delay"]
        freq_color = "#e74c3c" if pct > 40 else "#f39c12" if pct > 20 else "#2ecc71"
        st.markdown(f"""
        <div style="background:{card_bg()};border-radius:12px;padding:16px 20px">
          <div style="font-size:0.7rem;color:{card_sub()};text-transform:uppercase;
                      letter-spacing:1.2px;margin-bottom:8px">Frequency & Severity</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;
                      font-weight:700;color:{freq_color}">{pct:.1f}%</div>
          <div style="font-size:0.75rem;color:{card_sub()};margin-top:4px">
            of flights had this activity late<br>
            avg {med:.1f} min delay when late · {stat['n']:,} flights analysed
          </div>
        </div>
        """, unsafe_allow_html=True)
