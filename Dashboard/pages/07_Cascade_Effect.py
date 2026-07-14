"""
Cascade Effect — data-driven delay propagation.
Ridge regression on 25k+ flights shows exactly how one activity's lateness
multiplies through the turnaround chain.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Ridge
import streamlit as st
from utils.loader       import load_data, render_date_filters
from utils.style        import inject_css, chart_template, chart_fc, card_bg, card_text, card_sub, header_bg, header_border
from utils.insights     import insight_card, insight_strip, no_data_info
from utils.cascade      import (NODES, NODE_MAP, DEPS, TEAM_COLORS, TEAM_LABELS,
                                STRUCTURAL, build_flowchart, build_timeline,
                                measured_no_data_nodes, orphan_milestone_summary)
from utils.crossfilter  import init_xf, apply_xf, render_xf_bar

st.set_page_config(page_title="Cascade Effect | SATS", page_icon="🔗", layout="wide", initial_sidebar_state="expanded")
inject_css()
TEMPLATE = chart_template()
FC       = chart_fc()

# ── Regression ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Training cascade regression models…")
def build_regression_models(_df):
    dep_col = "Target_Departure_Delay_mins"
    activity_stats, pairwise = {}, {}

    for name, team, col, *_ in NODES:
        if not col or col not in _df.columns:
            continue
        data = _df[[col, dep_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 50:
            continue

        X, y = data[[col]].values, data[dep_col].values
        reg = Ridge(alpha=10.0).fit(X, y)
        y_pred = reg.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(max(0.0, 1 - ss_res / ss_tot)) if ss_tot > 1 else 0.0

        late_mask = data[col] > 2
        late_data = data[late_mask]

        activity_stats[name] = {
            "col": col,
            "coef": float(np.asarray(reg.coef_).ravel()[0]),
            "intercept": float(np.asarray(reg.intercept_).ravel()[0]),
            "r2": r2,
            "n_total": len(data),
            "pct_late": float(late_mask.mean() * 100),
            "median_dep_when_late": float(late_data[dep_col].median()) if len(late_data) > 5 else 0.0,
        }

    for child, parents in DEPS.items():
        c_node = NODE_MAP.get(child)
        if not c_node or not c_node[2] or c_node[2] not in _df.columns:
            continue
        for parent in parents:
            p_node = NODE_MAP.get(parent)
            if not p_node or not p_node[2] or p_node[2] not in _df.columns:
                continue
            # Use loc to get a single column even when parent == child column
            p_col, c_col = p_node[2], c_node[2]
            cols = list(dict.fromkeys([p_col, c_col]))  # deduplicate but preserve order
            pair = _df[cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 50:
                continue
            Xp = pair[p_col].values.reshape(-1, 1)
            yc = pair[c_col].values
            reg = Ridge(alpha=10.0).fit(Xp, yc)
            ss_res = np.sum((yc - reg.predict(Xp)) ** 2)
            ss_tot = np.sum((yc - yc.mean()) ** 2)
            r2 = float(max(0., 1 - ss_res / ss_tot)) if ss_tot > 1 else 0.
            pairwise[(parent, child)] = {"coef": float(np.asarray(reg.coef_).ravel()[0]), "r2": r2}

    return activity_stats, pairwise


def propagate(source, delay_mins, activity_stats, pairwise):
    nd = {n[0]: 0.0 for n in NODES}
    nd[source] = float(delay_mins)
    order = [(source, float(delay_mins))]   # propagation order for animation
    visited, queue = set(), [source]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        cur_d = nd[cur]
        if cur_d <= 0:
            continue
        for child, parents in DEPS.items():
            if cur not in parents:
                continue
            edge = (cur, child)
            rate = min(pairwise[edge]["coef"], 1.0) if edge in pairwise and pairwise[edge]["coef"] > 0 else 0.80
            propagated = cur_d * rate
            if propagated > nd.get(child, 0):
                nd[child] = propagated
                if child not in visited:
                    order.append((child, propagated))
                queue.append(child)
    # Regression-backed departure estimate
    if source in activity_stats:
        s = activity_stats[source]
        nd["✈️  DEPARTURE"] = max(nd["✈️  DEPARTURE"],
                                  s["coef"] * delay_mins + s["intercept"])
    if ("✈️  DEPARTURE", nd["✈️  DEPARTURE"]) not in order and nd["✈️  DEPARTURE"] > 0:
        order.append(("✈️  DEPARTURE", nd["✈️  DEPARTURE"]))
    return nd, order


# ── Load & compute ─────────────────────────────────────────────────────────────
df = load_data()
df = render_date_filters(df, page_key="cascade")
init_xf()
df = apply_xf(df)
activity_stats, pairwise = build_regression_models(df)

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{header_bg()};
            border-left:4px solid #e74c3c;padding:18px 24px;
            border-radius:0 12px 12px 0;margin-bottom:22px">
  <h2 style="margin:0;color:{card_text()}">🔗 Cascade Effect Simulator</h2>
  <p style="margin:8px 0 0;color:{card_sub()};font-size:.93rem">
    Select any ground activity and set a delay — see exactly which milestones are hit downstream.
    <b style="color:#f4a621">Cascade predictions use Ridge regression trained on 25,000+ real flights.</b>
  </p>
</div>
""", unsafe_allow_html=True)
render_xf_bar()

# ── Page-level insight ────────────────────────────────────────────────────────
if activity_stats:
    top_name, top_s = max(activity_stats.items(), key=lambda x: x[1]["coef"])
    sev = "red" if top_s["coef"] > 0.8 else "amber"
    insight_card(
        problem=(f"**{top_name}** has the highest cascade multiplier — every minute it runs late "
                 f"shifts departure by **{top_s['coef']:.2f} min** on average "
                 f"(R²={top_s['r2']:.2f})."),
        impact=(f"{top_s['pct_late']:.0f}% of flights with this activity are late. "
                f"When late, the median departure delay is {top_s['median_dep_when_late']:.0f} min."),
        action=(f"Prioritise resources on {top_name} first. "
                "Use the simulator below to trace exactly which downstream milestones are hit."),
        severity=sev,
        icon="🔗",
    )
else:
    insight_strip("Cascade regression requires at least 50 flights per activity. Load more data or broaden the date filter to enable insights.", severity="amber")

# ── Controls + Flowchart ───────────────────────────────────────────────────────
ctrl_col, chart_col = st.columns([1, 2.6], gap="large")

# Only measured nodes (those with a delay column) can be simulated/regressed.
selectable = [n[0] for n in NODES if n[2] is not None]

with ctrl_col:
    st.markdown("### ⚙️ Simulate a Delay")

    selected = st.selectbox("Delayed activity", selectable, index=0)
    delay_val = st.slider("Minutes late", 0, 60, 10, 1)

    nd, prop_order = propagate(selected, float(delay_val), activity_stats, pairwise)
    dep_impact = max(nd.get("✈️  DEPARTURE", 0), 0)

    # ── Big impact card ──
    if delay_val == 0:
        st.markdown("""
        <div class="impact-card impact-card-ok glow-ok">
          <div class="title">Departure Impact</div>
          <div class="value">No Delay</div>
          <div class="sub">All activities on schedule</div>
        </div>""", unsafe_allow_html=True)
    elif dep_impact > 4:
        extra = ""
        if selected in activity_stats:
            s = activity_stats[selected]
            extra = (f"<br><span class='sub-extra'>"
                     f"{s['pct_late']:.0f}% of flights with this activity late → Delayed</span>")
        st.markdown(f"""
        <div class="impact-card impact-card-red glow-red">
          <div class="title">⚠️ Departure Delayed</div>
          <div class="value">+{dep_impact:.0f} min</div>
          <div class="sub">Flight will miss on-time target{extra}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="impact-card impact-card-amber">
          <div class="title">Acceptable Delay</div>
          <div class="value">+{dep_impact:.0f} min</div>
          <div class="sub">Within tolerable range (≤ 4 min)</div>
        </div>""", unsafe_allow_html=True)

    # ── Model confidence ──
    if selected in activity_stats:
        s = activity_stats[selected]
        rel = int(s["r2"] * 100)
        bar_clr = "#2ecc71" if rel > 50 else "#f4a621" if rel > 25 else "#e74c3c"
        bar_pct = max(4, rel)
        st.markdown(f"""
        <div style="background:{card_bg()};border-radius:10px;padding:14px 16px;margin-top:12px">
          <div style="font-size:.75rem;color:{card_sub()};margin-bottom:6px">
            Model confidence &nbsp;
            <b style="color:{bar_clr};font-size:1.1rem">{rel}%</b>
          </div>
          <div style="background:#2d3348;border-radius:4px;height:6px;overflow:hidden">
            <div style="background:{bar_clr};width:{bar_pct}%;height:100%;
                        border-radius:4px;transition:width .4s"></div>
          </div>
          <div style="font-size:.72rem;color:{card_sub()};margin-top:6px">
            {s['n_total']:,} flights analysed &nbsp;·&nbsp;
            {s['pct_late']:.0f}% had this activity running late
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(
            f"⚠️ **{selected}** has fewer than 50 recorded flights — the Departure Impact "
            "number above is a rough fallback estimate, not a fitted model. Treat it as "
            "indicative only."
        )

    # ── Cascade summary bullets ──
    affected = sorted(
        [(n, d) for n, d in nd.items()
         if d > 1 and n not in STRUCTURAL and n not in ("Aircraft Arrives", selected)],
        key=lambda x: -x[1]
    )[:5]
    if affected:
        bullets_html = "".join(
            f"<div style='margin:4px 0;font-size:.82rem'>"
            f"<span style='color:#f4a621'>▸</span> "
            f"<b>{n}</b> shifts by <b style='color:#e74c3c'>+{d:.0f} min</b></div>"
            for n, d in affected
        )
        st.markdown(f"""
        <div style="background:{card_bg()};border-radius:10px;padding:14px 16px;margin-top:12px">
          <div style="font-size:.78rem;color:{card_sub()};margin-bottom:6px;text-transform:uppercase;
                      letter-spacing:.8px">Downstream hits</div>
          {bullets_html}
        </div>""", unsafe_allow_html=True)

    # ── Team legend ──
    st.markdown("<div style='margin-top:14px;font-size:.78rem;color:#8892a4'><b>Team legend</b></div>",
                unsafe_allow_html=True)
    legend_html = "".join(
        f"<span style='background:{TEAM_COLORS[t]};padding:3px 10px;border-radius:5px;"
        f"margin:2px 2px 2px 0;display:inline-block;font-size:.72rem;color:white'>"
        f"{TEAM_LABELS[t]}</span>"
        for t in TEAM_COLORS if t not in ("ref", "dep")
    )
    st.markdown(legend_html, unsafe_allow_html=True)

with chart_col:
    # ── Animation controls ────────────────────────────────────────────────────
    anim_col, _ = st.columns([1, 3])
    with anim_col:
        run_anim = st.button("▶  Animate Cascade", use_container_width=True,
                             disabled=(delay_val == 0))

    chart_placeholder = st.empty()

    # Nodes with no data across the dataset — drawn as "NO DATA" (computed once)
    _agg_nodata = measured_no_data_nodes(df)

    if run_anim and delay_val > 0:
        # Step through propagation order, lighting up one node at a time
        partial_nd = {n[0]: 0.0 for n in NODES}
        for idx, (step_name, step_delay) in enumerate(prop_order):
            partial_nd[step_name] = step_delay
            chart_placeholder.plotly_chart(
                build_flowchart(partial_nd, selected, highlight_node=step_name,
                                no_data_nodes=_agg_nodata),
                use_container_width=True,
                key=f"anim_plotly_{selected}_{step_name}_{idx}"
            )
            time.sleep(0.45)
        # Hold final state briefly then hand off to tabs below
        time.sleep(0.4)
        chart_placeholder.empty()

    tab_flow, tab_time = st.tabs(["🔗  Flow Map", "⏰  Timeline View"])

    with tab_flow:
        st.plotly_chart(build_flowchart(nd, selected, no_data_nodes=_agg_nodata),
                        use_container_width=True)

    with tab_time:
        st.plotly_chart(build_timeline(nd, selected), use_container_width=True)

from utils.cascade import NO_DATA_NOTE as _ND_NOTE
st.caption(_ND_NOTE)

# Off-map milestones — recorded in the data but not on the cascade map yet
_orph_summary = orphan_milestone_summary(df)
with st.expander(f"📎 Off-map milestones with data ({len(_orph_summary)})"):
    st.caption("Recorded in the current data but not shown on the map above — "
               "candidates to add as nodes. New milestones appear here automatically.")
    if _orph_summary:
        import pandas as _pd07
        _tot = max(len(df), 1)
        st.dataframe(_pd07.DataFrame([{
            "BU": _bu, "Milestone": _lbl,
            "Flights with data": f"{_nr:,} ({100*_nr/_tot:.0f}%)",
            "Late / over (>0.5m)": f"{_ns:,}",
        } for _bu, _lbl, _nr, _ns, _c in _orph_summary]),
            use_container_width=True, hide_index=True)
    else:
        st.caption("None — every milestone with data is already on the map.")

st.divider()

# ── Detailed impact table ──────────────────────────────────────────────────────
st.markdown("### 📋 Activity-by-Activity Impact")

rows = []
for name, team, col, *_ in NODES:
    delay = nd.get(name, 0)
    struct = name in STRUCTURAL
    if name == "Aircraft Arrives":
        icon, status = "🔵", "Reference"
    elif struct:
        icon, status = "◇", "No data available"
    elif name == selected:
        icon, status = "🔴", "Source of delay"
    elif name == "✈️  DEPARTURE":
        icon = "✈️" ; status = "Final outcome"
    elif delay > 10:
        icon, status = "🔴", "Heavily impacted"
    elif delay > 4:
        icon, status = "🟠", "Impacted"
    elif delay > 0.5:
        icon, status = "🟡", "Minor shift"
    else:
        icon, status = "🟢", "Not affected"

    reg_label = ""
    if name in activity_stats:
        s = activity_stats[name]
        reg_label = f"{s['coef']:.2f}x  (R²={s['r2']:.2f})"

    rows.append({
        "": icon,
        "Activity": name,
        "Team": TEAM_LABELS.get(team, team),
        "Delay Impact": "—" if struct else (f"+{delay:.0f} min" if delay > 0.5 else "On Time"),
        "Status": status,
        "Effect on Departure (regression)": reg_label,
    })

st.dataframe(
    pd.DataFrame(rows), use_container_width=True, hide_index=True,
    column_config={
        "": st.column_config.Column(width="small"),
        "Delay Impact": st.column_config.Column(width="medium"),
    },
)

st.divider()

# ── Regression bar chart ───────────────────────────────────────────────────────
st.markdown("### 📈 Which Activities Most Drive Departure Delays?")
st.caption(
    "Ridge regression coefficient: how many minutes of departure delay result from "
    "each extra minute this activity is late. Higher bar = bigger chain reaction risk."
)

if activity_stats:
    reg_rows = [
        {
            "Activity": name,
            "Team": TEAM_LABELS.get(NODE_MAP[name][1], NODE_MAP[name][1]),
            "Impact (min/min)": round(s["coef"], 3),
            "Reliability R²": round(s["r2"], 3),
            "% Flights Late": round(s["pct_late"], 1),
            "Avg Dep Delay When Late (min)": round(s["median_dep_when_late"], 1),
        }
        for name, s in activity_stats.items()
    ]
    reg_df = pd.DataFrame(reg_rows).sort_values("Impact (min/min)", ascending=False)

    color_map = {v: TEAM_COLORS.get(k, "#888") for k, v in TEAM_LABELS.items()}
    fig_reg = px.bar(
        reg_df, x="Impact (min/min)", y="Activity", color="Team",
        orientation="h", template=TEMPLATE,
        color_discrete_map=color_map, height=420,
    )
    fig_reg.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="", xaxis_title="min of departure delay per min of activity delay",
        margin=dict(l=10, r=20, t=44, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    fig_reg.update_traces(marker_opacity=0.88)
    st.plotly_chart(fig_reg, use_container_width=True)

    with st.expander("Show full regression table"):
        st.dataframe(
            reg_df, use_container_width=True, hide_index=True,
            column_config={
                "Impact (min/min)": st.column_config.NumberColumn(format="%.3f"),
                "Reliability R²": st.column_config.ProgressColumn(
                    format="%.2f", min_value=0, max_value=1),
                "% Flights Late": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

# ── Pairwise edge coefficients ─────────────────────────────────────────────────
if pairwise:
    st.markdown("### 🔗 How Strongly Each Link Transfers Delay")
    st.caption("Each row shows: if the upstream activity is 1 min late, "
               "how many minutes does the downstream activity shift?")

    pair_rows = [
        {
            "From": p, "To": c,
            "Transfer Rate (min/min)": round(v["coef"], 3),
            "Reliability R²": round(v["r2"], 3),
        }
        for (p, c), v in sorted(pairwise.items(), key=lambda x: -x[1]["coef"])
    ]
    st.dataframe(
        pd.DataFrame(pair_rows), use_container_width=True, hide_index=True,
        column_config={
            "Transfer Rate (min/min)": st.column_config.NumberColumn(format="%.3f"),
            "Reliability R²": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        },
    )

st.divider()

# ── Indirect Cascade Correlations ─────────────────────────────────────────────
st.markdown("### 🔀 Indirect Cascade Links")
st.caption(
    "Activities that correlate strongly with departure delay even when **not directly linked** in the flowchart. "
    "These hidden relationships often point to shared resource contention or scheduling knock-on effects."
)

@st.cache_data(show_spinner="Computing indirect correlations…")
def find_indirect_correlations(_df, direct_pairs):
    """
    Returns pairs of activities with high pairwise delay correlation
    that are NOT in the direct DEPS graph.

    Vectorized: a single df.corr() call + one matrix-multiply for pairwise
    non-null counts, instead of an O(N^2) Python loop that built a fresh
    3-column DataFrame and called .corr() three times per pair (was ~2,800
    calls on the production dataset — the source of both slow loads and a
    wall of "invalid value encountered in divide" warnings from numpy under
    the hood, since many point-milestone columns are near-constant).
    """
    if "Target_Departure_Delay_mins" not in _df.columns:
        return pd.DataFrame()

    delay_cols = [
        c for c in _df.columns
        if c.endswith("_analysis_Delay_mins") and
        not any(t in c for t in ["ActualDuration", "PlannedDuration"])
    ]

    # Keep only columns with enough data
    valid = [c for c in delay_cols if _df[c].dropna().__len__() >= 100]
    if not valid:
        return pd.DataFrame()

    # Build name → column map
    def col_to_name(c):
        return c.replace("_analysis_Delay_mins", "").replace("milestone_", "").replace("_", " ").title()

    target_col = "Target_Departure_Delay_mins"
    all_cols   = valid + [target_col]
    sub        = _df[all_cols]

    # One vectorized correlation matrix instead of thousands of Series.corr() calls.
    # Zero-variance columns produce NaN cells (harmless) rather than a printed
    # warning per pair.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = sub.corr()

    # Pairwise non-null counts via matrix multiply — replaces per-pair .dropna().
    notna       = sub.notna().to_numpy(dtype=np.int32)
    pair_counts = notna.T @ notna
    dep_idx     = len(valid)  # index of target_col in all_cols / corr_matrix

    rows = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ca, cb = valid[i], valid[j]
            na, nb = col_to_name(ca), col_to_name(cb)

            # Check if this pair (or reverse) is a direct DEPS edge
            if (na, nb) in direct_pairs or (nb, na) in direct_pairs:
                continue

            n_common = int(pair_counts[i, j])
            if n_common < 80:
                continue

            corr_ab = corr_matrix.iat[i, j]
            if pd.isna(corr_ab) or abs(corr_ab) < 0.25:
                continue

            corr_a_d = corr_matrix.iat[i, dep_idx]
            corr_b_d = corr_matrix.iat[j, dep_idx]

            rows.append({
                "Activity A":            na,
                "Activity B":            nb,
                "Correlation A↔B":       round(float(corr_ab), 3),
                "A → Departure corr":    round(float(corr_a_d), 3) if pd.notna(corr_a_d) else None,
                "B → Departure corr":    round(float(corr_b_d), 3) if pd.notna(corr_b_d) else None,
                "n":                     n_common,
            })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("Correlation A↔B", ascending=False)
    return result.head(30)

indirect_df = find_indirect_correlations(df, set(pairwise.keys()))

if not indirect_df.empty:
    # Colour rows by correlation strength
    def colour_corr(val):
        try:
            v = float(val)
            if v >= 0.5:
                return "color:#e74c3c;font-weight:700"
            elif v >= 0.35:
                return "color:#f39c12;font-weight:600"
            return "color:#6b9bd2"
        except Exception:
            return ""

    styled_ind = (
        indirect_df.drop(columns=["n"])
        .style
        .map(colour_corr, subset=["Correlation A↔B"])
    )
    st.dataframe(
        styled_ind, use_container_width=True, hide_index=True,
        column_config={
            "Correlation A↔B":    st.column_config.NumberColumn(format="%.3f"),
            "A → Departure corr": st.column_config.NumberColumn(format="%.3f"),
            "B → Departure corr": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    insight_strip(
        "High A↔B correlation without a direct link usually means: "
        "(1) both activities depend on the same shared resource (e.g. ground crew, equipment), "
        "(2) an upstream activity not in the model is delaying both, or "
        "(3) they share a scheduling dependency that isn't captured in the flow map. "
        "Investigate these pairs when doing root-cause analysis.",
        severity="amber",
    )
else:
    st.info("No significant indirect correlations found — all strong relationships are already captured in the direct DEPS graph.")

st.divider()

# ── Actionable recommendation ─────────────────────────────────────────────────
if selected in activity_stats and delay_val > 0:
    s = activity_stats[selected]
    predicted = s["coef"] * delay_val + s["intercept"]
    predicted_display = max(predicted, 0)   # regression intercept can be negative; clamp to 0 for display
    rec_color = "#e74c3c" if predicted_display > 4 else "#f4a621" if predicted_display > 0 else "#2ecc71"

    top_hits = sorted(
        [(n, d) for n, d in nd.items()
         if d > 2 and n not in STRUCTURAL and n not in ("Aircraft Arrives", selected)],
        key=lambda x: -x[1],
    )[:4]
    bullets = "".join(
        f"&nbsp;&nbsp;• <b>{n}</b> is pushed back by ~{d:.0f} min<br>"
        for n, d in top_hits
    )

    st.markdown(f"""
    <div style="background:{card_bg()};border-left:4px solid {rec_color};
                border-radius:0 12px 12px 0;padding:18px 24px;margin-top:4px">
      <b style="font-size:1rem;color:{card_text()}">💡 Recovery Guidance</b><br><br>
      <span style="color:{card_text()}">A <b>{delay_val}-minute delay</b> in <b>{selected}</b> cascades as follows:</span><br><br>
      {bullets}
      <br>
      <b style="color:{rec_color}">
        Model predicts departure {'on time' if predicted_display == 0 else f'{predicted_display:.0f} min late'}
        &nbsp;(confidence: {int(s['r2']*100)}%).
      </b><br><br>
      <span style="color:{card_sub()}">
        To recover: deploy additional resources to the most impacted downstream activities,
        especially those feeding into <b>Final Readback</b> and <b>PLB Retract</b>,
        which are the last convergence points before departure clearance.
      </span>
    </div>
    """, unsafe_allow_html=True)
elif delay_val > 0:
    no_data_info(f"Recovery Guidance for {selected}", n=0, min_n=50)
