"""
pages/11_Flight_Monitor.py
Live Flight Monitor — shows recent departures with real-time ML delay probability.

Historic mode: most recent N flights sorted by scheduled departure (desc).
Live mode (future): upcoming flights sorted by ETD (asc), probabilities update
as each milestone timestamp is recorded.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from utils.loader   import load_data, load_model, merge_lw_features, render_date_filters
from utils.style    import inject_css, card_bg, card_text, card_sub, header_bg, header_border
from utils.insights import insight_card, insight_strip
from utils.cascade  import build_flowchart, build_timeline, get_flight_cascade_nd

st.set_page_config(
    page_title="Flight Monitor | SATS",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Data & Model ──────────────────────────────────────────────────────────────
df    = load_data()
df    = render_date_filters(df, page_key="monitor")
df    = merge_lw_features(df)
model = load_model()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{header_bg()};
            border:1px solid {header_border()};border-radius:16px;
            padding:22px 32px;margin-bottom:20px">
  <h2 style="margin:0;color:{card_text()};font-size:1.5rem">🛫 Flight Departure Monitor</h2>
  <p style="margin:8px 0 0;color:{card_sub()};font-size:.9rem">
    Each row shows one departure. The <b style="color:#c5d3f0">Delay Probability</b> is
    computed by the ML model from ground conditions — watch for rows turning
    <b style="color:#e74c3c">red</b> and intervene before pushback.
    When connected to a live feed, this table updates as milestones are recorded.
  </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model not found. Run `python prepare_data.py` first.")
    st.stop()

pipeline      = model["pipeline"]
le            = model["label_encoder"]
feature_cols  = model["feature_columns"]
numeric_feats = model.get("numeric_feats", [])
cat_feats     = model.get("categorical_feats", [])

# ── Sidebar filters ───────────────────────────────────────────────────────────
# Date/Carrier/Terminal/Aircraft Type/ICAO/Destination/Day of Week/Tight Ground
# Time/Incoming Delay all come from render_date_filters above (shared, persists
# across pages). Only Flight Monitor's own unique controls are added here.
with st.sidebar:
    st.markdown("### 🛫 Flight Filters")

    n_flights = st.number_input("Flights to show", 20, 500, 100, step=20)

    st.markdown("#### Risk")
    risk_filter = st.selectbox(
        "Risk level",
        ["All flights", "High risk only (Delayed prob > 50%)", "Any risk (Delayed prob > 30%)"],
    )

    st.divider()
    st.markdown("""
    <div style="background:rgba(77,159,255,0.05);border:1px solid rgba(77,159,255,0.12);
                border-radius:12px;padding:14px 16px">
      <div style="font-size:0.7rem;font-weight:700;color:#4d9fff;
                  text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">
        How to Use
      </div>
      <div style="font-size:0.76rem;color:#6b7fa3;line-height:1.7">
        1. Sort by <b style="color:#e74c3c">🔴 Delayed %</b> to prioritise<br>
        2. <b>GT Tight</b> = ground time below minimum required<br>
        3. <b>Incoming Delay</b> = late inbound compresses turnaround<br>
        4. <b>Milestones</b> = % of ground events recorded<br>
        5. Use risk filter to focus on urgent flights only<br><br>
        <b style="color:#8898b8">Live mode:</b><br>
        When connected to real-time feeds, probabilities update as milestones are recorded.
      </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Select & sort flights ─────────────────────────────────────────────────────
# Carrier/Day of Week/Tight Ground Time/Incoming Delay are already applied to
# df by render_date_filters() above.
working = df.copy()

# Sort by most recent scheduled departure
if "departure_offBlock.scheduled" in working.columns:
    working = working.sort_values("departure_offBlock.scheduled", ascending=False)
working = working.head(int(n_flights)).copy()

if working.empty:
    st.warning("No flights match the current filters.")
    st.stop()

# ── ML Prediction ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing delay probabilities for all flights…")
def batch_predict(_df, _pipeline, _le, _feature_cols, _numeric_feats, _cat_feats):
    inp = _df.reindex(columns=_feature_cols)
    # Fill missing feature columns
    for col in _feature_cols:
        if col not in _df.columns:
            inp[col] = "Unknown" if col in _cat_feats else -999.0
    for c in _numeric_feats:
        if c in inp.columns:
            inp[c] = pd.to_numeric(inp[c], errors="coerce").fillna(-999)
    for c in _cat_feats:
        if c in inp.columns:
            inp[c] = inp[c].astype(str)
    try:
        probas = _pipeline.predict_proba(inp)
        preds  = _pipeline.predict(inp)
        labels = _le.inverse_transform(preds)
        return probas, labels
    except Exception:
        return None, None

probas, pred_labels = batch_predict(
    working, pipeline, le, feature_cols, numeric_feats, cat_feats
)

# ── Milestone scan stats ───────────────────────────────────────────────────────
milestone_actual_cols = [c for c in df.columns
                         if ".actual" in c and "milestone" in c]

def milestone_scan_pct(row):
    recorded = sum(1 for c in milestone_actual_cols if c in row.index and pd.notna(row[c]))
    return int(recorded / max(len(milestone_actual_cols), 1) * 100)

working["_scan_pct"] = working.apply(milestone_scan_pct, axis=1)

# ── Build display rows ────────────────────────────────────────────────────────
class_order = list(le.classes_)  # e.g. ["Acceptable","Delayed","On-Time"]

rows = []
for i, (idx, row) in enumerate(working.iterrows()):
    if probas is not None:
        prob_row   = probas[i]
        prob_dict  = dict(zip(class_order, prob_row))
        pred_label = pred_labels[i]
        delay_prob = prob_dict.get("Delayed", 0) * 100
        ontime_prob = prob_dict.get("On-Time", 0) * 100
        acc_prob    = prob_dict.get("Acceptable", 0) * 100
    else:
        prob_dict = {}
        pred_label = "Unknown"
        delay_prob = ontime_prob = acc_prob = 0.0

    # STD in SGT (UTC+8)
    std_raw = row.get("departure_offBlock.scheduled", pd.NaT)
    if pd.notna(std_raw):
        try:
            std_sgt = pd.to_datetime(std_raw, utc=True).tz_convert("Asia/Singapore")
            std_str = std_sgt.strftime("%d %b  %H:%M")
        except Exception:
            std_str = str(std_raw)[:16]
    else:
        std_str = "—"

    incoming = row.get("Incoming_Delay_mins", np.nan)
    gt_avail = row.get("Available_Ground_Time_mins", np.nan)
    gt_defic = row.get("Is_Ground_Time_Deficient", np.nan)

    scan_pct = int(row.get("_scan_pct", 0))
    scan_bar = "▓" * (scan_pct // 10) + "░" * (10 - scan_pct // 10)

    lw_gw  = row.get("LW_In_Ground_Window", 0)
    lw_dep = row.get("LW_Active_At_Departure", 0)
    lw_day_flag = row.get("LW_Day_Had_Warning", 0)
    if lw_gw == 1 or lw_dep == 1:
        lw_badge = "⚡ ACTIVE"
    elif lw_day_flag == 1:
        lw_badge = "⚡ Today"
    else:
        lw_badge = "—"

    rows.append({
        "Flight":         str(row.get("identification_iata", "—")),
        "Carrier":        str(row.get("identification_carrierCode", "—")),
        "Aircraft":       str(row.get("aircraft_typeICAO", "—")),
        "Body":           str(row.get("aircraft_bodyType", "—")),
        "Destination":    str(row.get("destination_iata", "—")),
        "Terminal":       str(row.get("origin_terminal", "—")),
        "STD (SGT)":      std_str,
        "Incoming Delay": f"{incoming:+.0f} min" if pd.notna(incoming) else "—",
        "GT Buffer":      f"{gt_avail:.0f} min"  if pd.notna(gt_avail) else "—",
        "GT Tight":       "⚠️ YES" if str(gt_defic) == "1" or gt_defic == 1.0 else "OK",
        "⚡ LW":          lw_badge,
        "Milestones":     f"{scan_bar} {scan_pct}%",
        "On-Time %":      f"{ontime_prob:.0f}%",
        "Acceptable %":   f"{acc_prob:.0f}%",
        "🔴 Delayed %":   f"{delay_prob:.1f}%",
        "Predicted":      pred_label,
        "_delay_prob":    delay_prob,
        "_pred":          pred_label,
    })

monitor_df = pd.DataFrame(rows)

# Apply risk filter
if risk_filter == "High risk only (Delayed prob > 50%)":
    monitor_df = monitor_df[monitor_df["_delay_prob"] > 50]
elif risk_filter == "Any risk (Delayed prob > 30%)":
    monitor_df = monitor_df[monitor_df["_delay_prob"] > 30]

if monitor_df.empty:
    st.info("📊 No flights match the current risk filter — try \"All flights\" or widen the other filters.")
    st.stop()

# ── Summary insight row ────────────────────────────────────────────────────────
if not monitor_df.empty:
    n_high_risk  = (monitor_df["_delay_prob"] > 50).sum()
    n_med_risk   = ((monitor_df["_delay_prob"] > 30) & (monitor_df["_delay_prob"] <= 50)).sum()
    n_safe       = (monitor_df["_delay_prob"] <= 30).sum()
    avg_dp       = monitor_df["_delay_prob"].mean()

    if n_high_risk > 0:
        high_risk_flights = monitor_df[monitor_df["_delay_prob"] > 50]["Flight"].tolist()[:5]
        insight_card(
            problem=(f"**{n_high_risk} flight{'s' if n_high_risk > 1 else ''}** in this window have "
                     f"delay probability above 50%: {', '.join(high_risk_flights)}."),
            impact=(f"{n_med_risk} more are in the 30–50% amber zone. "
                    f"Average delay probability across all {len(monitor_df)} flights shown: {avg_dp:.0f}%."),
            action=("Sort by 🔴 Delayed % (click column header) to prioritise intervention. "
                    "Check the GT Tight and Incoming Delay columns first — those are the fastest wins."),
            icon="🛫", severity="red",
        )
    elif n_med_risk > 0:
        insight_card(
            problem=f"No high-risk flights, but **{n_med_risk} flights** are in the amber zone (30–50% delay probability).",
            impact=f"Average delay probability is {avg_dp:.0f}%. Amber flights can be tipped by a single late milestone.",
            action="Monitor amber rows closely. If an incoming delay or tight ground time is flagged, intervene early.",
            icon="🛫", severity="amber",
        )
    else:
        insight_strip(f"✅ All {len(monitor_df)} flights shown have delay probability below 30%. Current conditions look stable.", "green")

# ── Risk colour coding function ────────────────────────────────────────────────
def color_row(row):
    # row comes from display_df (15 visible cols) — look up _delay_prob via index
    dp = monitor_df.loc[row.name, "_delay_prob"]
    if dp > 50:
        bg = "background-color: rgba(231,76,60,0.18)"
    elif dp > 30:
        bg = "background-color: rgba(243,156,18,0.12)"
    else:
        bg = ""
    return [bg] * len(row)   # len(row) == number of display columns

def color_delay_col(val):
    try:
        v = float(str(val).replace("%", ""))
        if v > 50:
            return "color:#e74c3c;font-weight:700"
        elif v > 30:
            return "color:#f39c12;font-weight:600"
        return "color:#2ecc71"
    except Exception:
        return ""

def color_predicted(val):
    if val == "Delayed":
        return "color:#e74c3c;font-weight:700"
    elif val == "Acceptable":
        return "color:#f39c12;font-weight:600"
    elif val == "On-Time":
        return "color:#2ecc71"
    return ""

# ── Render table ──────────────────────────────────────────────────────────────
display_cols = [c for c in monitor_df.columns if not c.startswith("_")]
display_df   = monitor_df[display_cols]

# Build styled dataframe
styled = (
    display_df.style
    .apply(color_row, axis=1)   # passes display_df rows; color_row looks up monitor_df internally
    .map(color_delay_col, subset=["🔴 Delayed %"])
    .map(color_predicted, subset=["Predicted"])
    .set_properties(**{"font-size": "0.83rem"})
)

st.markdown(f"**Showing {len(display_df)} flights** — sorted by most recent scheduled departure. "
            "Rows highlighted 🔴 red have >50% delay probability; 🟡 amber = 30–50%.")
st.dataframe(styled, use_container_width=True, hide_index=True, height=600)

# ── Legend ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;gap:18px;font-size:0.78rem;margin-top:8px;color:#8892a4">
  <span><span style="color:#e74c3c">■</span> Delayed prob &gt; 50% — intervene now</span>
  <span><span style="color:#f39c12">■</span> 30–50% — monitor closely</span>
  <span><span style="color:#2ecc71">■</span> &lt; 30% — on track</span>
  <span style="margin-left:16px">
    <b>GT Tight</b> = ground time below minimum required &nbsp;|&nbsp;
    <b>Milestones</b> = % of milestone timestamps recorded
  </span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Per-flight Cascade Inspector ──────────────────────────────────────────────
st.markdown("### 🔗 Cascade Inspector")
st.caption(
    "Select any flight from the table above to see its actual ground ops cascade — "
    "which milestones ran late and how the delay propagated to departure."
)

if not monitor_df.empty:
    # Build labels: Flight | STD | Dep delay %
    _mon_labels = []
    for _, _mrow in monitor_df.iterrows():
        _lbl = f"{_mrow['Flight']}  |  {_mrow['STD (SGT)']}  |  🔴 {_mrow['🔴 Delayed %']}"
        _mon_labels.append(_lbl)

    _sel_mon = st.selectbox(
        "Flight to inspect",
        options=_mon_labels,
        index=0,
        key="mon_cascade_sel",
    )
    _mon_idx  = _mon_labels.index(_sel_mon)
    _flight_w = working.iloc[_mon_idx]

    # Sidebar: milestones recorded for this flight but not on the cascade map
    from utils.cascade import flight_orphan_milestones as _orphan_ms
    _orph = _orphan_ms(_flight_w, working.columns)
    with st.sidebar:
        st.markdown("### 📎 Off-map milestones")
        st.caption("Recorded for this flight but not on the cascade map.")
        if _orph:
            import pandas as _pdo
            st.dataframe(_pdo.DataFrame([{"BU": _bu, "Milestone": _lbl,
                "Status": (f"+{_v:.0f} min" if _v > 0.5 else "recorded")} for _bu, _lbl, _v, _c in _orph]),
                use_container_width=True, hide_index=True)
        else:
            st.caption("None — all recorded milestones are on the map.")

    _nd, _src = get_flight_cascade_nd(_flight_w)

    _n_late = sum(1 for v in _nd.values() if v > 0)
    _dep_d  = _nd.get("✈️  DEPARTURE", 0)
    _dc     = "#e74c3c" if _dep_d > 4 else "#f39c12" if _dep_d > 0 else "#2ecc71"

    st.markdown(
        f"<div style='background:{card_bg()};border-left:3px solid {_dc};"
        f"border-radius:0 10px 10px 0;padding:10px 18px;margin-bottom:10px'>"
        f"<b style='color:{card_text()}'>{_mon_labels[_mon_idx]}</b> — "
        f"<span style='color:{_dc}'>"
        f"{'On time' if _dep_d <= 0 else f'Departed +{_dep_d:.0f} min late'}"
        f"</span> &nbsp;·&nbsp; "
        f"<span style='color:{card_sub()}'>{_n_late} milestone{'s' if _n_late != 1 else ''} delayed</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _mtab_flow, _mtab_tl = st.tabs(["🔗 Flow Map", "⏰ Timeline View"])
    with _mtab_flow:
        from utils.cascade import flight_no_data_nodes as _fnd
        st.plotly_chart(build_flowchart(_nd, _src, no_data_nodes=_fnd(_flight_w)),
                        use_container_width=True)
    with _mtab_tl:
        st.plotly_chart(build_timeline(_nd, _src), use_container_width=True)

