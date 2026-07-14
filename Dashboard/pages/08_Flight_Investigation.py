"""
pages/7_Flight_Investigation.py
Flight Investigation Table — case-by-case filterable view of every flight.
Ground workers can filter by any combination of activity delay, flight delay,
aircraft type, terminal, date, and destination, then copy the results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from utils.loader       import load_data, render_date_filters, merge_lw_features
from utils.style        import inject_css, card_bg, card_text, card_sub
from utils.insights     import insight_strip
from utils.cascade      import build_flowchart, build_timeline, get_flight_cascade_nd
from utils.crossfilter  import init_xf, apply_xf, render_xf_bar

st.set_page_config(page_title="Flight Investigation | SATS", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ─── Friendly column names for the key activity delays ───────────────────────
ACTIVITY_COLS = {
    "milestone_ramp_manAtBay_analysis_Delay_mins":          "Ramp: At Bay",
    "milestone_aic_manAtBay_analysis_Delay_mins":           "AIC: At Bay",
    "milestone_cabinSvc_manAtBay_analysis_Delay_mins":      "Cabin Svc: At Bay",
    "milestone_pax_openGateTeam_analysis_Delay_mins":       "PAX: Gate Opens",
    "milestone_cargo_DLS_analysis_Delay_mins":              "Cargo: DLS",
    "milestone_security_door2_analysis_Delay_mins":         "Security: Door Check",
    "milestone_pax_boardingLoad_(10)_analysis_Delay_mins":  "PAX: 10% Boarded",
    "milestone_loadControl_loadsheetACK_analysis_Delay_mins": "Load Control: Loadsheet",
    "milestone_pax_lastPaxBoarded_analysis_Delay_mins":     "PAX: Last Pax",
    "milestone_ramp_finalReadback_analysis_Delay_mins":     "Ramp: Final Readback",
}

# ─── Data ────────────────────────────────────────────────────────────────────
df = load_data()
df = render_date_filters(df, page_key="investigation")
df = merge_lw_features(df)
init_xf()
df = apply_xf(df)

# ─── Page header ─────────────────────────────────────────────────────────────
st.markdown("## 🔍 Flight Investigation Table")
render_xf_bar()
st.caption(
    "Filter flights by any combination of activity delays, departure status, "
    "aircraft type, terminal, and date. The table is fully sortable and copyable."
)
st.divider()

# ─── In-page filters ─────────────────────────────────────────────────────────
st.markdown("### Filters")

f1, f2, f3 = st.columns(3)

with f1:
    delay_class_opts = ["All", "On-Time", "Acceptable", "Delayed"]
    sel_delay_class = st.selectbox("Flight Departure Status", delay_class_opts)

with f2:
    min_dep_delay = st.number_input(
        "Min Departure Delay (min)", min_value=0, max_value=300, value=0, step=5,
        help="Show only flights with departure delay ≥ this value."
    )

with f3:
    max_dep_delay = st.number_input(
        "Max Departure Delay (min)", min_value=0, max_value=500, value=500, step=5,
        help="Show only flights with departure delay ≤ this value (500 = no cap)."
    )

st.markdown("#### Activity Delay Thresholds")
st.caption("Show only flights where any of the selected activities exceeded the threshold you set.")

act_filter_expander = st.expander("Set activity delay filters (optional)", expanded=False)
act_filters = {}
with act_filter_expander:
    cols = st.columns(2)
    available_act_cols = [c for c in ACTIVITY_COLS if c in df.columns]
    for i, col_key in enumerate(available_act_cols):
        with cols[i % 2]:
            label = ACTIVITY_COLS[col_key]
            threshold = st.slider(
                f"{label} delay >",
                min_value=0, max_value=60, value=0, step=1,
                key=f"act_thresh_{i}",
                help=f"Filter to flights where {label} was delayed by more than this many minutes."
            )
            if threshold > 0:
                act_filters[col_key] = threshold

st.divider()

# ─── Apply filters ────────────────────────────────────────────────────────────
filtered = df.copy()

# Delay class filter
if sel_delay_class != "All" and "Target_Departure_Delay_Class" in filtered.columns:
    filtered = filtered[filtered["Target_Departure_Delay_Class"].astype(str) == sel_delay_class]

# Departure delay range
if "Target_Departure_Delay_mins" in filtered.columns:
    filtered = filtered[filtered["Target_Departure_Delay_mins"].fillna(0) >= min_dep_delay]
    if max_dep_delay < 500:
        filtered = filtered[filtered["Target_Departure_Delay_mins"].fillna(0) <= max_dep_delay]

# Activity delay thresholds (AND logic: all selected thresholds must be exceeded)
for col_key, threshold in act_filters.items():
    if col_key in filtered.columns:
        filtered = filtered[filtered[col_key].fillna(0) > threshold]

# ─── KPI row ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Flights Shown", f"{len(filtered):,}")

if "Target_Departure_Delay_Class" in filtered.columns:
    n_delayed = (filtered["Target_Departure_Delay_Class"].astype(str) == "Delayed").sum()
    n_ontime  = (filtered["Target_Departure_Delay_Class"].astype(str) == "On-Time").sum()
    k2.metric("Delayed", f"{n_delayed:,}")
    k3.metric("On-Time", f"{n_ontime:,}")

if "Target_Departure_Delay_mins" in filtered.columns:
    del_rows = filtered[filtered["Target_Departure_Delay_Class"].astype(str) == "Delayed"]
    avg_d = del_rows["Target_Departure_Delay_mins"].clip(upper=120).mean()
    k4.metric("Avg Delay (Delayed)", f"{avg_d:.1f} min" if not pd.isna(avg_d) else "—")

if "Incoming_Delay_mins" in filtered.columns:
    avg_inc = filtered["Incoming_Delay_mins"].dropna().clip(upper=120).mean()
    k5.metric("Avg Incoming Delay", f"{avg_inc:.1f} min" if not pd.isna(avg_inc) else "—")

if "LW_Day_Had_Warning" in filtered.columns:
    lw_ct  = filtered["LW_Day_Had_Warning"].sum()
    lw_gw  = filtered["LW_In_Ground_Window"].sum() if "LW_In_Ground_Window" in filtered.columns else 0
    lw_dep = filtered["LW_Active_At_Departure"].sum() if "LW_Active_At_Departure" in filtered.columns else 0
    if lw_ct > 0:
        insight_strip(
            f"⚡ <b>{int(lw_ct)}</b> flights in this view were on a lightning warning day. "
            f"<b>{int(lw_gw)}</b> had LW overlap in their ground-handling window. "
            f"<b>{int(lw_dep)}</b> had LW active at scheduled departure time. "
            "See the <b>Delay Attribution</b> page for full LW impact analysis.",
            severity="amber",
        )

st.divider()

# ─── Insight ─────────────────────────────────────────────────────────────────
if len(filtered) > 0 and "Target_Departure_Delay_Class" in filtered.columns:
    n_del   = (filtered["Target_Departure_Delay_Class"].astype(str) == "Delayed").sum()
    pct_del = n_del / len(filtered) * 100
    avail_act = {k: v for k, v in ACTIVITY_COLS.items() if k in filtered.columns}
    worst_act_str = ""
    if avail_act:
        act_means  = {v: filtered[k].dropna().mean() for k, v in avail_act.items()}
        worst_act  = max(act_means, key=act_means.get)
        worst_val  = act_means[worst_act]
        if worst_val > 0.5:
            worst_act_str = f" Highest avg activity delay: **{worst_act}** ({worst_val:.1f} min)."
    if pct_del > 40:
        insight_strip(f"⚠️ {pct_del:.0f}% of the {len(filtered):,} filtered flights are delayed.{worst_act_str} Sort by that column to isolate the worst offenders.", severity="red")
    elif pct_del > 20:
        insight_strip(f"🟡 {pct_del:.0f}% of the {len(filtered):,} filtered flights are delayed.{worst_act_str}", severity="amber")
    else:
        insight_strip(f"✅ {pct_del:.0f}% of the {len(filtered):,} filtered flights are delayed — within a healthy range.{worst_act_str}", severity="green")

# ─── Build display table ──────────────────────────────────────────────────────
display_rows = filtered.copy()

# Date and time from scheduled departure
if "departure_offBlock.scheduled" in display_rows.columns:
    sched = pd.to_datetime(display_rows["departure_offBlock.scheduled"], errors="coerce", utc=True)
    sched_sgt = sched.dt.tz_convert("Asia/Singapore")
    display_rows["Date"]         = sched_sgt.dt.strftime("%Y-%m-%d")
    display_rows["Dep Time"]     = sched_sgt.dt.strftime("%H:%M")
    display_rows["_sched_epoch"] = sched.astype("int64")   # hidden sort key for "Most Recent"

# Nicely formatted delay class with emoji
def fmt_class(v):
    v = str(v)
    if v == "Delayed":    return "🔴 Delayed"
    if v == "Acceptable": return "🟠 Acceptable"
    if v == "On-Time":    return "🟢 On-Time"
    return "—"

if "Target_Departure_Delay_Class" in display_rows.columns:
    display_rows["Status"] = display_rows["Target_Departure_Delay_Class"].apply(fmt_class)

# Terminal label
if "origin_terminal" in display_rows.columns:
    display_rows["Terminal"] = display_rows["origin_terminal"].astype(str).replace(
        {"1": "T1", "2": "T2", "3": "T3", "4": "T4"}
    )

# Select and rename columns for display
col_map = {
    "identification_iata":            "Flight",
    "Date":                           "Date",
    "Dep Time":                       "Dep Time (UTC)",
    "Terminal":                       "Terminal",
    "destination_iata":               "Destination",
    "aircraft_typeICAO":              "Aircraft",
    "aircraft_bodyType":              "Body Type",
    "Status":                         "Status",
    "Target_Departure_Delay_mins":    "Dep Delay (min)",
    "Incoming_Delay_mins":            "Incoming Delay (min)",
    "Available_Ground_Time_mins":     "Ground Time (min)",
}
# LW columns
lw_col_map = {
    "LW_Day_Had_Warning":     "⚡ LW Day",
    "LW_Active_At_Departure": "⚡ LW at Dep",
    "LW_In_Ground_Window":    "⚡ LW in GW",
    "LW_Overlap_Ground_Mins": "⚡ LW Overlap (min)",
}
for raw_col, friendly_name in lw_col_map.items():
    if raw_col in display_rows.columns:
        col_map[raw_col] = friendly_name

# Activity delay columns
for raw_col, friendly_name in ACTIVITY_COLS.items():
    if raw_col in display_rows.columns:
        col_map[raw_col] = friendly_name

keep = [c for c in col_map if c in display_rows.columns]
# Carry the hidden epoch column through for "Most Recent" sorting
if "_sched_epoch" in display_rows.columns:
    keep_with_sort = keep + ["_sched_epoch"]
else:
    keep_with_sort = keep
table = display_rows[keep_with_sort].rename(columns=col_map).copy()

# Round numeric columns
for col in table.select_dtypes(include="float").columns:
    if col != "_sched_epoch":
        table[col] = table[col].round(1)

# ─── Sort control ─────────────────────────────────────────────────────────────
MOST_RECENT = "Most Recent"
sort_opts = [MOST_RECENT, "Dep Delay (min)", "Date", "Status", "Destination",
             "Aircraft", "Incoming Delay (min)"]
sort_opts = [s for s in sort_opts
             if s == MOST_RECENT or s in table.columns]

sc1, sc2 = st.columns([2, 1])
with sc1:
    sort_col = st.selectbox("Sort by", sort_opts, index=0)
with sc2:
    # "Most Recent" defaults to descending (newest first); user can still flip it
    sort_asc = st.radio("Order", ["Descending ▼", "Ascending ▲"],
                        index=0, horizontal=True) == "Ascending ▲"

if sort_col == MOST_RECENT:
    if "_sched_epoch" in table.columns:
        table = table.sort_values("_sched_epoch", ascending=sort_asc, na_position="last")
    elif "Date" in table.columns:
        table = table.sort_values("Date", ascending=sort_asc, na_position="last")
elif sort_col in table.columns:
    table = table.sort_values(sort_col, ascending=sort_asc, na_position="last")

# Drop the hidden sort key before display
table = table.drop(columns=["_sched_epoch"], errors="ignore")

# ─── Table ────────────────────────────────────────────────────────────────────
if table.empty:
    st.info("📊 No flights match the current filters — try widening the date range or clearing a filter.")
    st.stop()

st.markdown(
    f"**{len(table):,} flights** match your filters. "
    "Select rows and press **Ctrl+C / Cmd+C** to copy to clipboard."
)

# Column config for colour-coding and width hints
col_cfg = {
    "Status": st.column_config.TextColumn("Status", width="medium"),
    "Dep Delay (min)": st.column_config.NumberColumn(
        "Dep Delay (min)", format="%.1f", width="small"
    ),
    "Incoming Delay (min)": st.column_config.NumberColumn(
        "Incoming Delay (min)", format="%.1f", width="small"
    ),
    "Ground Time (min)": st.column_config.NumberColumn(
        "Ground Time (min)", format="%.0f", width="small"
    ),
}
# Progress bars for activity delay columns (max = 30 min for visual scale)
for friendly in ACTIVITY_COLS.values():
    if friendly in table.columns:
        col_cfg[friendly] = st.column_config.ProgressColumn(
            friendly, min_value=0, max_value=30, format="%.1f min", width="medium"
        )

st.dataframe(
    table.reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    height=560,
    column_config=col_cfg,
)

st.caption(
    "💡 Tip: click any column header to sort. "
    "Shift-click to multi-sort. "
    "Select rows then Ctrl+C / Cmd+C to copy the data."
)

st.divider()

# ─── Per-flight Cascade Inspector ────────────────────────────────────────────
st.markdown("### 🔗 Cascade Inspector")
st.caption(
    "Select any flight from the filtered results to see its actual ground ops "
    "cascade — which milestones ran late and how the delay propagated to departure."
)

if filtered.empty:
    st.info("No flights to inspect — adjust the filters above.")
else:
    # Build labels for the selectbox from filtered results
    def _cascade_label(row):
        parts = []
        for _c in ("identification_iata", "identification_carrierCode"):
            if _c in row.index:
                v = str(row.get(_c, ""))
                if v and v not in ("nan", "None"):
                    parts.append(v)
        if "departure_offBlock.scheduled" in row.index:
            try:
                dt = pd.to_datetime(row["departure_offBlock.scheduled"], utc=True).tz_convert("Asia/Singapore")
                parts.append(dt.strftime("%d %b %H:%M"))
            except Exception:
                pass
        dep_d = row.get("Target_Departure_Delay_mins", np.nan)
        if pd.notna(dep_d):
            parts.append(f"dep +{dep_d:.0f}m")
        return "  |  ".join(parts) if parts else str(row.get("id", ""))[:12]

    # Sort by most recent departure for display
    _insp_df = filtered.copy()
    if "departure_offBlock.scheduled" in _insp_df.columns:
        _insp_df = _insp_df.sort_values("departure_offBlock.scheduled", ascending=False)

    _labels = _insp_df.apply(_cascade_label, axis=1).tolist()

    _sel_label = st.selectbox(
        "Flight to inspect",
        options=_labels,
        index=0,
        key="inv_cascade_sel",
    )
    _sel_idx = _labels.index(_sel_label)
    _flight  = _insp_df.iloc[_sel_idx]

    # Sidebar: milestones recorded for this flight but not on the cascade map
    from utils.cascade import flight_orphan_milestones as _orphan_ms
    _orph = _orphan_ms(_flight, df.columns)
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

    _nd, _src = get_flight_cascade_nd(_flight)

    # Mini summary bar
    _n_late  = sum(1 for v in _nd.values() if v > 0)
    _dep_d   = _nd.get("✈️  DEPARTURE", 0)
    _dc      = "#e74c3c" if _dep_d > 4 else "#f39c12" if _dep_d > 0 else "#2ecc71"
    st.markdown(
        f"<div style='background:{card_bg()};border-left:3px solid {_dc};"
        f"border-radius:0 10px 10px 0;padding:10px 18px;margin-bottom:10px'>"
        f"<b style='color:{card_text()}'>{_sel_label}</b> — "
        f"<span style='color:{_dc}'>"
        f"{'On time' if _dep_d <= 0 else f'Departed +{_dep_d:.0f} min late'}"
        f"</span> &nbsp;·&nbsp; "
        f"<span style='color:{card_sub()}'>{_n_late} milestone{'s' if _n_late != 1 else ''} delayed</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _ctab_flow, _ctab_tl = st.tabs(["🔗 Flow Map", "⏰ Timeline View"])
    with _ctab_flow:
        from utils.cascade import flight_no_data_nodes as _fnd
        st.plotly_chart(build_flowchart(_nd, _src, no_data_nodes=_fnd(_flight)),
                        use_container_width=True)
    with _ctab_tl:
        st.plotly_chart(build_timeline(_nd, _src), use_container_width=True)
