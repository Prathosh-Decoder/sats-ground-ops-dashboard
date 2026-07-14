"""
pages/5_Flight_Deep_Dive.py
Flight Deep Dive — Gantt chart and narrative for a single flight.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.loader       import load_data, render_date_filters, merge_lw_features, load_lw_data
from utils.style        import inject_css, chart_template, chart_fc
from utils.insights     import insight_card, insight_strip
from utils.cascade      import build_flowchart, build_timeline, get_flight_cascade_nd
from utils.crossfilter  import init_xf, apply_xf, render_xf_bar

st.set_page_config(page_title="Flight Deep Dive | SATS", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")
inject_css()

TEMPLATE = chart_template()
FC       = chart_fc()
COLOR_MAP = {"On-Time": "#2ecc71", "Acceptable": "#f39c12", "Delayed": "#e74c3c"}

SPECIAL_HANDLING_LABELS = {
    "WCHR": "Wheelchair (ramp)",
    "WCHS": "Wheelchair (steps)",
    "WCHC": "Wheelchair (carry)",
    "UMNR": "Unaccompanied minor",
    "MAAS": "Meet and assist",
    "MEDA": "Medical case",
    "STCR": "Stretcher",
    "BLND": "Blind passenger",
    "DEAF": "Deaf passenger",
    "VIP":  "VIP",
    "VVIP": "VVIP",
    "DPNA": "Disabled passenger",
    "IMPT": "Important person",
}

# ─── Data ────────────────────────────────────────────────────────────────────
df = load_data()
df = render_date_filters(df, page_key="deepdive")
df = merge_lw_features(df)
_, lw_daily = load_lw_data()
init_xf()
df = apply_xf(df)

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("## ✈️ Flight Deep Dive")
render_xf_bar()
st.caption("Select any flight to see a full Gantt chart, delay narrative, and passenger requirements.")
st.divider()

# ─── Flight Selector ─────────────────────────────────────────────────────────
if "id" not in df.columns:
    st.error("Flight ID column not found.")
    st.stop()

# Build flight labels — cached so removing the old 1,000-flight cap stays fast
carrier_col = "identification_carrierCode" if "identification_carrierCode" in df.columns else None
iata_col    = "identification_iata"         if "identification_iata"         in df.columns else None
sched_col   = "departure_offBlock.scheduled"

@st.cache_data(show_spinner="Loading flight list…")
def build_flight_options(_df, carrier_col, iata_col, sched_col, cache_key):
    sample_df = _df.dropna(subset=["Target_Departure_Delay_Class"]).copy()
    sample_df = sample_df[sample_df["Target_Departure_Delay_Class"] != "nan"]

    if sched_col in sample_df.columns:
        sample_df = sample_df.sort_values(sched_col, ascending=False)
        sched_str = (pd.to_datetime(sample_df[sched_col], errors="coerce", utc=True)
                     .dt.tz_convert("Asia/Singapore").dt.strftime("%d %b %Y %H:%M"))
    else:
        sched_str = pd.Series("", index=sample_df.index)

    parts = pd.DataFrame(index=sample_df.index)
    parts["carrier"] = sample_df[carrier_col].astype(str) if carrier_col else ""
    parts["iata"]    = sample_df[iata_col].astype(str)     if iata_col    else ""
    parts["sched"]   = sched_str.fillna("")
    parts["idtag"]   = "(" + sample_df["id"].astype(str).str[:8] + "...)"

    def _join(row):
        return " | ".join(v for v in row if v and v != "nan")

    labels = parts.apply(_join, axis=1).tolist()
    ids    = sample_df["id"].tolist()
    return labels, ids


# Cheap fingerprint of the filtered set (not the whole frame) so the cache
# correctly busts when filters change instead of always returning whatever
# was cached on the very first call.
_cache_key = (len(df), int(pd.util.hash_pandas_object(df["id"], index=False).sum()))
flight_labels, flight_ids = build_flight_options(df, carrier_col, iata_col, sched_col, _cache_key)

id_to_label = dict(zip(flight_ids, flight_labels))
label_to_id = {v: k for k, v in id_to_label.items()}

search_col, _ = st.columns([2, 3])
with search_col:
    selected_label = st.selectbox(
        "Search / Select a Flight",
        options=flight_labels,
        index=0 if flight_labels else None,
        help=f"All {len(flight_labels):,} flights matching the current filters — type to search.",
    )

selected_id  = label_to_id.get(selected_label)
if not selected_id:
    st.warning("Could not resolve flight ID.")
    st.stop()

flight = df[df["id"] == selected_id].iloc[0]

# ─── Sidebar: milestones recorded for THIS flight but not on the cascade map ───
from utils.cascade import flight_orphan_milestones as _orphan_ms
_orph = _orphan_ms(flight, df.columns)
with st.sidebar:
    st.markdown("### 📎 Off-map milestones")
    st.caption("Recorded for this flight but not shown on the cascade map "
               "(e.g. services this airline uses, or newly-added milestones).")
    if _orph:
        import pandas as _pdo
        st.dataframe(
            _pdo.DataFrame([{"BU": bu, "Milestone": lbl,
                             "Status": (f"+{v:.0f} min" if v > 0.5 else "recorded")}
                            for bu, lbl, v, _c in _orph]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption("None — every milestone recorded for this flight is on the map.")

# ─── Flight Summary Cards ─────────────────────────────────────────────────────
st.divider()
st.markdown("### Flight Summary")

delay_class = str(flight.get("Target_Departure_Delay_Class", "Unknown"))
delay_mins  = flight.get("Target_Departure_Delay_mins", np.nan)
delay_color = COLOR_MAP.get(delay_class, "#9e9e9e")

badge_html = f"""
<span style="background:{delay_color};color:white;padding:6px 18px;border-radius:20px;
font-weight:700;font-size:1rem;letter-spacing:0.5px;">{delay_class.upper()}</span>
"""
if not np.isnan(delay_mins) if isinstance(delay_mins, float) else True:
    delay_str = f"  {'+' if delay_mins > 0 else ''}{delay_mins:.0f} min"
else:
    delay_str = ""

# ── LW status for this flight ─────────────────────────────────────────────
lw_day  = int(flight.get("LW_Day_Had_Warning",     0))
lw_dep  = int(flight.get("LW_Active_At_Departure", 0))
lw_gw   = int(flight.get("LW_In_Ground_Window",    0))
lw_mins = float(flight.get("LW_Overlap_Ground_Mins", 0))

if lw_dep or lw_gw:
    lw_badge_color = "#e67e22" if lw_gw else "#f1c40f"
    lw_detail = (f"{lw_mins:.0f} min overlap with ground-handling window" if lw_gw
                 else "active at scheduled departure time")
    lw_badge_html = (
        f'<span style="background:{lw_badge_color};color:#1a1a1a;padding:6px 18px;'
        f'border-radius:20px;font-weight:700;font-size:0.95rem;margin-left:8px;">'
        f'⚡ LW ACTIVE — {lw_detail}</span>'
    )
    st.markdown(f"**Status:** {badge_html} {delay_str} &nbsp; {lw_badge_html}", unsafe_allow_html=True)
elif lw_day:
    st.markdown(
        f"**Status:** {badge_html} {delay_str} &nbsp;"
        f'<span style="background:rgba(241,196,15,0.2);border:1px solid #f1c40f;color:#f1c40f;'
        f'padding:4px 14px;border-radius:20px;font-size:0.85rem;margin-left:6px;">'
        f"⚡ LW day (no direct overlap)</span>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(f"**Status:** {badge_html} {delay_str}", unsafe_allow_html=True)
st.markdown("")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Carrier",     str(flight.get("identification_carrierCode", "N/A")))
c2.metric("Aircraft",    str(flight.get("aircraft_typeICAO", flight.get("aircraft_bodyType", "N/A"))))
c3.metric("Origin",      str(flight.get("origin_terminal",  "N/A")))
c4.metric("Destination", str(flight.get("destination_iata",  "N/A")))

sched_dep = flight.get("departure_offBlock.scheduled")
actual_dep = flight.get("departure_offBlock.actual")
try:
    sched_str  = (pd.to_datetime(sched_dep, utc=True).tz_convert("Asia/Singapore").strftime("%H:%M")
                  if pd.notna(sched_dep) else "N/A")
    actual_str = (pd.to_datetime(actual_dep, utc=True).tz_convert("Asia/Singapore").strftime("%H:%M")
                  if pd.notna(actual_dep) else "N/A")
except Exception:
    sched_str = actual_str = "N/A"

c5.metric("Scheduled Dep", sched_str, delta=f"Actual: {actual_str}")

st.divider()

# ─── Insight ─────────────────────────────────────────────────────────────────
_delay_num = delay_mins if isinstance(delay_mins, (int, float)) and not np.isnan(delay_mins) else None
_lw_note = ""
if lw_gw:
    _lw_note = (f" ⚡ **Lightning warning was active for {lw_mins:.0f} min** of the ground-handling window — "
                f"outdoor ramp activities were suspended during that period.")
elif lw_dep:
    _lw_note = " ⚡ **Lightning warning was active at scheduled departure time** — pushback may have been held."
elif lw_day:
    _lw_note = " ⚡ Lightning warning issued on this departure date (no direct overlap with this flight's ground window)."

if delay_class == "Delayed" and _delay_num is not None:
    _lw_action = (" LW was active — check if the delay overlaps with the warning period before attributing to SATS operations." if (lw_dep or lw_gw)
                  else "Review the Ground Activity Timeline below to identify the first activity that ran over schedule.")
    insight_card(
        problem=f"This flight departed **{_delay_num:.0f} min late** — classified as DELAYED.{_lw_note}",
        impact="A late departure reduces turnaround buffer for the aircraft's next rotation and can cascade to the next flight.",
        action=_lw_action,
        severity="red" if not (lw_dep or lw_gw) else "amber",
        icon="✈️",
    )
elif delay_class == "Acceptable" and _delay_num is not None:
    insight_strip(
        f"🟡 Flight was {_delay_num:.0f} min late — within acceptable range.{_lw_note} "
        "Check the timeline below for any marginal activities.",
        severity="amber",
    )
elif delay_class == "On-Time":
    _ok_msg = "✅ Flight departed on time. All turnaround activities completed within schedule."
    if lw_day:
        _ok_msg += _lw_note
    insight_strip(_ok_msg, severity="green")

# ─── Ground Activity Timeline (cascade milestones) ───────────────────────────
st.markdown("### Ground Activity Timeline")
st.caption("Each tracked ground-ops milestone's planned vs actual time for this flight "
           "(Singapore time). Grey tick = planned, dot = actual (🟢 on time · 🔴 late). "
           "Dashed line = scheduled off-block (SOBT).")

from utils.cascade import NODES as _CN, DEPARTURE_NODE as _DEP_N, ARRIVES_NODE as _ARR_N

_nd_dd, _src_dd = get_flight_cascade_nd(flight)
_sobt = pd.to_datetime(flight.get("departure_offBlock.scheduled"), errors="coerce", utc=True)

# Measured milestones with a planned offset, in process order.
_ms = [(n, off, float(_nd_dd.get(n, 0.0))) for n, t, c, x, y, off in _CN
       if off is not None and n not in (_ARR_N, _DEP_N)]
_ms.sort(key=lambda r: r[1])

if pd.isna(_sobt) or not _ms:
    st.info("Scheduled departure time or milestone data unavailable for this flight.")
else:
    def _clock(mins):
        # ISO string (SGT) — Plotly parses as a date on traces, shapes & annotations
        return (_sobt + pd.Timedelta(minutes=mins)).tz_convert("Asia/Singapore").strftime("%Y-%m-%dT%H:%M:%S")

    _names = [m[0] for m in _ms]
    gantt_fig = go.Figure()

    # Slip bars (planned → actual) for milestones that ran late
    for n, off, dly in _ms:
        if dly > 0.5:
            gantt_fig.add_trace(go.Scatter(
                x=[_clock(off), _clock(off + dly)], y=[n, n], mode="lines",
                line=dict(color="rgba(231,76,60,0.40)", width=6),
                showlegend=False, hoverinfo="skip",
            ))

    # Planned markers
    gantt_fig.add_trace(go.Scatter(
        x=[_clock(off) for _, off, _d in _ms], y=_names,
        mode="markers", name="Planned",
        marker=dict(symbol="line-ns-open", size=13,
                    color="rgba(150,160,185,0.85)", line=dict(width=2)),
        hovertemplate="<b>%{y}</b><br>Planned: %{x|%H:%M} SGT<extra></extra>",
    ))

    # Actual markers, coloured by lateness
    _ac = ["#e74c3c" if d > 4 else "#f39c12" if d > 0.5 else "#2ecc71" for _, _o, d in _ms]
    gantt_fig.add_trace(go.Scatter(
        x=[_clock(off + dly) for _, off, dly in _ms], y=_names,
        mode="markers+text", name="Actual",
        marker=dict(symbol="circle", size=11, color=_ac, line=dict(color=FC, width=1)),
        text=[f" +{d:.0f}m" if d > 0.5 else "" for _, _o, d in _ms],
        textposition="middle right", textfont=dict(size=9, color="#ffb766"),
        hovertemplate="<b>%{y}</b><br>Actual: %{x|%H:%M} SGT<extra></extra>",
    ))

    # SOBT reference line
    gantt_fig.add_shape(type="line", x0=_clock(0), x1=_clock(0), y0=0, y1=1,
                        yref="paper", line=dict(color="#f4a621", dash="dash", width=2))
    gantt_fig.add_annotation(x=_clock(0), y=1.02, yref="paper", text="<b>SOBT</b>",
                             showarrow=False, font=dict(color="#f4a621", size=11),
                             xanchor="center", yanchor="bottom")

    gantt_fig.update_layout(
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(440, len(_ms) * 26 + 130),
        xaxis=dict(title="Time (SGT)", gridcolor="rgba(128,128,128,0.12)"),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(_names)),
                   title="", automargin=True, tickfont=dict(size=10)),
        margin=dict(t=30, b=50, l=20, r=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    st.plotly_chart(gantt_fig, use_container_width=True)

st.divider()

# ─── Delay Narrative (cascade milestones) ────────────────────────────────────
st.markdown("### Delay Narrative")
st.caption("How the delay built up across the ground-ops cascade for this flight.")

_late_ms = [(n, d) for n, off, d in _ms if d > 0.5]
_nparts = []
if not _late_ms:
    _nparts.append("🟢 Every tracked ground-ops milestone ran on time — no cascade delay.")
else:
    _fn, _fd = _late_ms[0]   # first late milestone in process order
    _nparts.append(f"🔴 **First milestone to slip:** {_fn} ran **{_fd:.0f} min** behind plan.")
    _wn, _wd = max(_late_ms, key=lambda x: x[1])
    if _wn != _fn:
        _nparts.append(f"🟠 The biggest single slip was **{_wn}** at **+{_wd:.0f} min**.")
    _nparts.append(
        f"⛓️ **{len(_late_ms)}** milestone{'s' if len(_late_ms) != 1 else ''} ran late in total."
    )

_dep_d = float(_nd_dd.get(_DEP_N, 0))
if _dep_d > 4:
    _nparts.append(f"✈️ **Pushback was {_dep_d:.0f} min late — classified DELAYED.**")
elif _dep_d > 0:
    _nparts.append(f"✈️ Pushback was {_dep_d:.0f} min late — within acceptable range.")
else:
    _nparts.append("✈️ **Pushback was on time.**")

for _p in _nparts:
    st.markdown(_p)

st.divider()

# ─── Special Handling ─────────────────────────────────────────────────────────
st.markdown("### Special Handling Requirements")

sh_cols = [c for c in df.columns if c.startswith("specialHandling_")]
sh_data = {}
for col in sh_cols:
    code = col.replace("specialHandling_", "")
    val  = flight.get(col, 0)
    try:
        val = int(val) if not np.isnan(float(val)) else 0
    except (ValueError, TypeError):
        val = 0
    if val > 0:
        sh_data[code] = val

if sh_data:
    sh_cols_display = st.columns(min(len(sh_data), 5))
    for i, (code, count) in enumerate(sh_data.items()):
        with sh_cols_display[i % 5]:
            label = SPECIAL_HANDLING_LABELS.get(code, code)
            st.metric(label, count)
else:
    st.info("No special handling requirements for this flight.")

st.divider()

# ─── Actual Cascade Flowchart ─────────────────────────────────────────────────
st.markdown("### 🔗 Actual Cascade — Ground Ops Flow")
st.caption(
    "Each node shows this flight's actual milestone delay. "
    "Red nodes ran late; the brightest red node was the biggest contributor. "
    "Red edges show which links were under pressure."
)

_nd, _src = get_flight_cascade_nd(flight)

# Summary: count delayed nodes
_n_delayed = sum(1 for v in _nd.values() if v > 0)
_dep_delay = _nd.get("✈️  DEPARTURE", 0)

if _n_delayed == 0:
    st.success("✅ All ground milestones completed on time for this flight — no cascade delays recorded.")
else:
    _delay_color = "#e74c3c" if _dep_delay > 4 else "#f39c12" if _dep_delay > 0 else "#2ecc71"
    st.markdown(
        f"<span style='color:{_delay_color};font-weight:600'>"
        f"{_n_delayed} milestone{'s' if _n_delayed > 1 else ''} ran late — "
        f"departure was {'on time' if _dep_delay <= 0 else f'+{_dep_delay:.0f} min late'}."
        f"</span>",
        unsafe_allow_html=True,
    )

_tab_flow, _tab_tl = st.tabs(["🔗 Flow Map", "⏰ Timeline View"])

with _tab_flow:
    from utils.cascade import flight_no_data_nodes as _fnd
    st.plotly_chart(build_flowchart(_nd, _src, no_data_nodes=_fnd(flight)),
                    use_container_width=True)

with _tab_tl:
    st.plotly_chart(build_timeline(_nd, _src), use_container_width=True)

from utils.cascade import NO_DATA_NOTE as _ND_NOTE
st.caption(_ND_NOTE)

# Detailed delay table
with st.expander("Show milestone delay details"):
    from utils.cascade import (NODES as _NODES, TEAM_LABELS as _TLABELS,
                               flight_no_data_nodes as _no_data_nodes,
                               NO_DATA_LABEL as _ND_LABEL)
    _nodata = _no_data_nodes(flight)
    _detail_rows = []
    for _name, _team, _col, *_ in _NODES:
        _d = _nd.get(_name, 0)
        if _name in _nodata:
            _icon, _delay_txt = "·", _ND_LABEL
        else:
            _icon = "🔴" if _d > 10 else "🟠" if _d > 4 else "🟡" if _d > 0 else "🟢"
            _delay_txt = f"+{_d:.0f} min" if _d > 0 else "On time"
        _detail_rows.append({
            "": _icon,
            "Milestone": _name,
            "Team": _TLABELS.get(_team, _team),
            "Actual Delay": _delay_txt,
        })
    import pandas as _pd2
    st.dataframe(_pd2.DataFrame(_detail_rows), use_container_width=True, hide_index=True)
