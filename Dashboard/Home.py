"""
Home.py — SATS Ground Operations Intelligence Dashboard
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.loader import load_data
from utils.style  import inject_css

st.set_page_config(
    page_title="SATS Ground Ops Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 24px;border-bottom:1px solid rgba(77,159,255,0.12);margin-bottom:16px">
        <div style="font-size:1.8rem;margin-bottom:6px">✈️</div>
        <div style="font-size:1rem;font-weight:800;color:#c5d3f0;letter-spacing:-0.3px">SATS Ground Ops</div>
        <div style="font-size:0.72rem;color:#6b7fa3;margin-top:2px">Changi Airport · Singapore</div>
        <div style="margin-top:10px">
            <span class="pill pill-green"><span class="live-dot"></span>Live Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Data ────────────────────────────────────────────────────────────────────
df    = load_data()
valid = df.dropna(subset=["Target_Departure_Delay_Class"])
valid = valid[valid["Target_Departure_Delay_Class"].astype(str) != "nan"]

total_flights = len(valid)
n_delayed     = (valid["Target_Departure_Delay_Class"].astype(str) == "Delayed").sum()
n_ontime      = (valid["Target_Departure_Delay_Class"].astype(str) == "On-Time").sum()
on_time_pct   = n_ontime / total_flights * 100
delay_pct     = n_delayed / total_flights * 100

avg_delay = np.nan
if "Target_Departure_Delay_mins" in valid.columns:
    avg_delay = (valid[valid["Target_Departure_Delay_Class"].astype(str) == "Delayed"]
                 ["Target_Departure_Delay_mins"].clip(upper=120).mean())

most_delayed_hour = 0
if "Hour_of_Day" in valid.columns:
    hr = (valid.groupby("Hour_of_Day")["Target_Departure_Delay_Class"]
          .apply(lambda x: (x.astype(str) == "Delayed").mean()))
    most_delayed_hour = int(hr.idxmax())

sched = pd.to_datetime(df.get("departure_offBlock.scheduled"), errors="coerce", utc=True)
date_min       = sched.min().strftime("%d %b %Y") if not pd.isna(sched.min()) else "—"
date_max       = sched.max().strftime("%d %b %Y") if not pd.isna(sched.max()) else "—"
avg_delay_safe = float(avg_delay) if not np.isnan(avg_delay) else 0.0

# ─── Hero ─────────────────────────────────────────────────────────────────────
components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:transparent; font-family:'Inter',sans-serif; }}

  .hero {{
    background: linear-gradient(135deg,#070d28 0%,#0a1640 50%,#070d28 100%);
    border: 1px solid rgba(77,159,255,0.18);
    border-radius: 20px;
    padding: 44px 52px;
    position: relative;
    overflow: hidden;
  }}
  /* Animated radar rings */
  .ring {{
    position:absolute; border-radius:50%;
    border:1px solid rgba(77,159,255,0.08);
    animation: expand 4s ease-out infinite;
  }}
  .ring:nth-child(1){{ width:200px;height:200px;top:-60px;right:80px;animation-delay:0s; }}
  .ring:nth-child(2){{ width:320px;height:320px;top:-120px;right:20px;animation-delay:0.8s; }}
  .ring:nth-child(3){{ width:460px;height:460px;top:-190px;right:-50px;animation-delay:1.6s; }}
  .ring:nth-child(4){{ width:600px;height:600px;top:-260px;right:-120px;animation-delay:2.4s; }}
  @keyframes expand {{
    0%  {{ opacity:0.5; transform:scale(0.97); }}
    100%{{ opacity:0;   transform:scale(1.03); }}
  }}
  /* Sweep line */
  .sweep {{
    position:absolute; top:-60px; right:80px;
    width:200px; height:200px;
    animation: spin 6s linear infinite;
    transform-origin: center;
  }}
  @keyframes spin {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
  .sweep-line {{
    position:absolute; top:50%; left:50%;
    width:50%; height:1px;
    background: linear-gradient(90deg,rgba(77,159,255,0.6) 0%,transparent 100%);
    transform-origin: left center;
  }}

  .badge {{
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(0,217,119,0.1);
    border:1px solid rgba(0,217,119,0.25);
    color:#00d977; padding:4px 14px;
    border-radius:20px; font-size:0.7rem;
    font-weight:700; letter-spacing:1px;
    text-transform:uppercase; margin-bottom:18px;
  }}
  .dot {{
    width:7px;height:7px;background:#00d977;border-radius:50%;
    box-shadow:0 0 8px #00d977;
    animation:pulse 1.5s ease-in-out infinite;
  }}
  @keyframes pulse{{ 0%,100%{{opacity:0.6;transform:scale(1)}} 50%{{opacity:1;transform:scale(1.3)}} }}

  h1 {{
    font-size:2.6rem; font-weight:900; color:#fff;
    letter-spacing:-1px; line-height:1.05; margin-bottom:10px;
  }}
  h1 span {{
    background:linear-gradient(135deg,#4d9fff 0%,#00d4ff 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }}
  .sub {{ font-size:1rem; color:#4a6080; font-weight:400; margin-bottom:36px; }}

  /* KPI strip */
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}
  .kpi {{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:18px 20px;
    position:relative; overflow:hidden;
    animation: fadeUp 0.6s ease forwards;
    opacity:0;
  }}
  @keyframes fadeUp {{ from{{opacity:0;transform:translateY(10px)}} to{{opacity:1;transform:none}} }}
  .kpi:nth-child(1){{ animation-delay:0.1s; --top:linear-gradient(90deg,#4d9fff,#00d4ff); }}
  .kpi:nth-child(2){{ animation-delay:0.2s; --top:linear-gradient(90deg,#00d977,#00ffaa); }}
  .kpi:nth-child(3){{ animation-delay:0.3s; --top:linear-gradient(90deg,#ff4757,#ff6b81); }}
  .kpi:nth-child(4){{ animation-delay:0.4s; --top:linear-gradient(90deg,#ffc107,#ffab00); }}
  .kpi::before {{
    content:''; position:absolute;
    top:0;left:0;right:0;height:2px;
    background:var(--top);
  }}
  .kpi-lbl {{
    font-size:0.63rem; font-weight:700; text-transform:uppercase;
    letter-spacing:1.5px; color:#8898b8; margin-bottom:8px;
  }}
  .kpi-num {{
    font-family:'JetBrains Mono',monospace;
    font-size:1.9rem; font-weight:700; color:#e8eeff;
    letter-spacing:-1px; line-height:1;
  }}
  .kpi:nth-child(1) .kpi-num{{ color:#4d9fff; text-shadow:0 0 20px rgba(77,159,255,0.35); }}
  .kpi:nth-child(2) .kpi-num{{ color:#00d977; text-shadow:0 0 20px rgba(0,217,119,0.35); }}
  .kpi:nth-child(3) .kpi-num{{ color:#ff4757; text-shadow:0 0 20px rgba(255,71,87,0.35); }}
  .kpi:nth-child(4) .kpi-num{{ color:#ffc107; text-shadow:0 0 20px rgba(255,193,7,0.35);  }}
  .kpi-sub {{ font-size:0.68rem; color:#6b7fa3; margin-top:6px; }}
</style>
</head>
<body>
<div class="hero">
  <div class="ring"></div><div class="ring"></div>
  <div class="ring"></div><div class="ring"></div>
  <div class="sweep"><div class="sweep-line"></div></div>

  <div class="badge"><div class="dot"></div>Live Analytics Platform</div>
  <h1>SATS <span>Ground Operations</span></h1>
  <div class="sub">Intelligence Dashboard &nbsp;·&nbsp; All SATS-Handled Airlines &nbsp;·&nbsp; Changi Airport, Singapore</div>

  <div class="kpis">
    <div class="kpi">
      <div class="kpi-lbl">Flights Analysed</div>
      <div class="kpi-num" id="k1">0</div>
      <div class="kpi-sub">{date_min} → {date_max}</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">On-Time Rate</div>
      <div class="kpi-num" id="k2">0%</div>
      <div class="kpi-sub">departures within ±0 min</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Delayed Rate</div>
      <div class="kpi-num" id="k3">0%</div>
      <div class="kpi-sub">&gt; 4 min late</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Avg Delay</div>
      <div class="kpi-num" id="k4">0 min</div>
      <div class="kpi-sub">among delayed flights</div>
    </div>
  </div>
</div>

<script>
function animateCounter(id, target, suffix, decimals, duration) {{
  const el = document.getElementById(id);
  const start = performance.now();
  function tick(now) {{
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    const val  = target * ease;
    el.textContent = decimals
      ? val.toFixed(1) + suffix
      : Math.round(val).toLocaleString() + suffix;
    if (t < 1) requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);
}}
window.addEventListener('load', () => {{
  setTimeout(() => {{
    animateCounter('k1', {total_flights}, '', false, 1400);
    animateCounter('k2', {on_time_pct:.1f}, '%', true,  1200);
    animateCounter('k3', {delay_pct:.1f},   '%', true,  1200);
    animateCounter('k4', {avg_delay_safe:.1f}, ' min', true, 1200);
  }}, 300);
}});
</script>
</body>
</html>
""", height=380)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ─── Navigation tiles ─────────────────────────────────────────────────────────

# Style for page_link anchors and group headers
st.markdown("""
<style>
[data-testid="stPageLink-NavLink"] {
    display:block !important; color:#4a6080 !important;
    font-size:0.72rem !important; font-weight:600 !important;
    text-decoration:none !important; padding:6px 18px 14px !important;
    border-radius:0 0 16px 16px !important;
    background:transparent !important; margin-top:-4px !important;
    transition:color 0.2s !important;
}
[data-testid="stPageLink-NavLink"]:hover { color:#4d9fff !important; }
</style>
""", unsafe_allow_html=True)

def _group_header(label: str, color: str = "#6b7fa3") -> None:
    st.markdown(
        f'<div style="font-size:0.68rem;font-weight:700;color:{color};'
        f'text-transform:uppercase;letter-spacing:1.5px;'
        f'margin:28px 0 12px 2px;border-bottom:1px solid {color}33;padding-bottom:6px">'
        f'{label}</div>',
        unsafe_allow_html=True,
    )

def _tile_row(pages_group: list) -> None:
    cols = st.columns(len(pages_group))
    for col, (icon, title, accent, bg, desc, page_file) in zip(cols, pages_group):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {accent}22;border-top:2px solid {accent};
                        border-radius:16px 16px 0 0;padding:20px 18px 12px;
                        box-shadow:0 4px 20px rgba(0,0,0,0.3)">
              <div style="font-size:1.3rem;margin-bottom:8px">{icon}</div>
              <div style="font-size:0.92rem;font-weight:700;color:#c5d3f0;margin-bottom:6px">{title}</div>
              <div style="font-size:0.73rem;color:#8898b8;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            st.page_link(page_file, label=f"Open {title} →", width="stretch")

# ── Group 1 — Live Operations ──────────────────────────────────────────────────
_group_header("📡  Live Operations", "#e74c3c")
_tile_row([
    ("🛫", "Flight Monitor", "#e74c3c", "rgba(231,76,60,0.08)",
     "Real-time delay probability per flight — spot high-risk departures and intervene before pushback.",
     "pages/01_Flight_Monitor.py"),
])

# ── Group 2 — Performance Analytics ───────────────────────────────────────────
_group_header("📊  Performance Analytics", "#4d9fff")
_tile_row([
    ("📊", "Overview",           "#4d9fff", "rgba(77,159,255,0.08)",
     "High-level KPIs — delay breakdown, top carriers, delay distribution, and aircraft analysis.",
     "pages/02_Overview.py"),
    ("⏰", "When Delays Happen", "#00d4ff", "rgba(0,212,255,0.08)",
     "Hour-of-day and day-of-week heatmaps that reveal exactly when delay risk peaks.",
     "pages/03_When_Delays_Happen.py"),
    ("🔍", "Delay Attribution",  "#9b59b6", "rgba(155,89,182,0.08)",
     "Classify each delay: SATS-caused, propagated from inbound, or tight schedule — know what you can fix.",
     "pages/04_Delay_Attribution.py"),
])

# ── Group 3 — Ground Operations ───────────────────────────────────────────────
_group_header("🔧  Ground Operations", "#a78bfa")
_tile_row([
    ("🔧", "Activity Analysis",  "#a78bfa", "rgba(167,139,250,0.08)",
     "Business unit drill-down — click any team, then any activity for a full delay breakdown.",
     "pages/05_Activity_Analysis.py"),
    ("🎯", "BU Impact Analyser", "#f39c12", "rgba(243,156,18,0.08)",
     "Pick a Business Unit to see its position in the ground ops flow and which activities drive delays.",
     "pages/06_BU_Impact.py"),
    ("🔗", "Cascade Effect",     "#ff4757", "rgba(255,71,87,0.08)",
     "Simulate how a delay in one activity ripples through the entire turnaround process.",
     "pages/07_Cascade_Effect.py"),
])

# ── Group 4 — Flight Tools ─────────────────────────────────────────────────────
_group_header("✈️  Flight Tools", "#ff9f43")
_tile_row([
    ("🔍", "Flight Investigation", "#ff9f43", "rgba(255,159,67,0.08)",
     "Filterable table of every flight — search by carrier, status, activity delay, or aircraft type.",
     "pages/08_Flight_Investigation.py"),
    ("✈️", "Flight Deep Dive",    "#ffc107", "rgba(255,193,7,0.08)",
     "Gantt chart and step-by-step narrative breakdown for any individual flight.",
     "pages/09_Flight_Deep_Dive.py"),
])

# ── Group 5 — AI & Prediction ──────────────────────────────────────────────────
_group_header("🤖  AI & Prediction", "#2ecc71")
_tile_row([
    ("🎯", "Delay Predictor", "#00d977", "rgba(0,217,119,0.08)",
     "ML model: enter flight parameters and get an instant departure delay risk score with probability breakdown.",
     "pages/10_Delay_Predictor.py"),
    ("💬", "Ask the Data",   "#2ecc71", "rgba(46,204,113,0.08)",
     "Type any question in plain English — the AI reads real computed numbers and answers with charts or text.",
     "pages/11_Ask_Data.py"),
])

# ── Group 6 — Data ────────────────────────────────────────────────────────────
_group_header("📋  Data", "#6b7fa3")
_tile_row([
    ("🔬", "Data Quality", "#00d4ff", "rgba(0,212,255,0.08)",
     "Completeness observatory — see null rates for every milestone, BU, terminal, and carrier.",
     "pages/12_Data_Quality.py"),
])

# ─── System status bar ────────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

bu_stats = []
# Tech Ramp activities (e.g. Thumbs Up) are stored under milestone_ramp_* but
# belong to the Tech Ramp team — split them out for a consistent taxonomy.
_techramp_cols = [c for c in df.columns
                  if "thumbsup" in c.lower() and "_analysis_Delay_mins" in c]
for bu, label in [("techramp","Tech Ramp"),("ramp","Ramp"),("pax","Pax Svc"),
                   ("security","Security"),("cargo","Cargo"),("aic","AIC"),
                   ("cabinsvc","Cabin Svc"),("baggage","Baggage"),("loadcontrol","Load Ctrl")]:
    if bu == "techramp":
        delay_cols = _techramp_cols
    else:
        delay_cols = [c for c in df.columns
                      if f"milestone_{bu}" in c.lower() and "_analysis_Delay_mins" in c
                      and "ActualDuration" not in c and "PlannedDuration" not in c
                      and "thumbsup" not in c.lower()]
    if not delay_cols:
        continue
    vals = pd.concat([df[c].dropna() for c in delay_cols])
    if len(vals) < 50:
        continue
    late_rate = float((vals > 4).mean() * 100)
    color = "#ff4757" if late_rate > 35 else "#ffc107" if late_rate > 20 else "#00d977"
    bu_stats.append((label, late_rate, color))

if bu_stats:
    st.markdown("""
    <div style="font-size:0.72rem;font-weight:700;color:#6b7fa3;
                text-transform:uppercase;letter-spacing:1.4px;margin-bottom:12px">
      Team Performance Overview
    </div>""", unsafe_allow_html=True)

    cols = st.columns(len(bu_stats))
    for col, (label, late_rate, color) in zip(cols, bu_stats):
        on_time = 100 - late_rate
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
                        border-radius:12px;padding:14px 16px;text-align:center">
              <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:1px;color:#6b7fa3;margin-bottom:8px">{label}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;
                          font-weight:700;color:{color};
                          text-shadow:0 0 14px {color}66">{on_time:.0f}%</div>
              <div style="font-size:0.62rem;color:#6b7fa3;margin-top:4px">on-time</div>
              <div style="background:rgba(255,255,255,0.05);border-radius:4px;
                          height:3px;margin-top:8px;overflow:hidden">
                <div style="background:{color};width:{on_time:.0f}%;height:100%;
                            border-radius:4px;box-shadow:0 0 6px {color}"></div>
              </div>
            </div>""", unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:#6b7fa3;font-size:0.72rem;
            margin-top:48px;padding:20px 0;
            border-top:1px solid rgba(255,255,255,0.04)">
    SATS Ltd &nbsp;·&nbsp; Ground Operations Intelligence &nbsp;·&nbsp;
    Changi Airport, Singapore &nbsp;·&nbsp;
    {total_flights:,} flights &nbsp;·&nbsp; {date_min} – {date_max}
</div>
""", unsafe_allow_html=True)
