"""
utils/insights.py
Reusable insight card and strip renderers for the SATS dashboard.

Usage:
    from utils.insights import insight_card, insight_strip
    insight_card(
        problem="41% of flights depart late. Carrier MI is worst at 58%.",
        impact="On a busy day that is ~28 extra delays affecting passengers and crew.",
        action="Audit MI turnaround process — focus on ramp loading and pax boarding.",
        severity="red",
    )

Not-enough-data guards:
    from utils.insights import MIN_SAMPLE_SIZE, has_enough_data, no_data_metric, no_data_card, no_data_info
    n = int(row["count"])   # the actual recorded-observation count for THIS specific
                             # activity/milestone/BU — never the overall filtered flight total
    if has_enough_data(n):
        st.metric("Avg Delay", f"{avg:.1f} min")
    else:
        no_data_metric("Avg Delay", n)
"""
import streamlit as st
import numpy as np
import pandas as pd

from utils.style import card_bg, card_text, card_sub


# ── Palette ────────────────────────────────────────────────────────────────────
_COLORS = {
    "red":   ("#e74c3c", "rgba(231,76,60,0.09)",  "rgba(231,76,60,0.18)"),
    "amber": ("#f39c12", "rgba(243,156,18,0.09)", "rgba(243,156,18,0.18)"),
    "green": ("#2ecc71", "rgba(46,204,113,0.09)", "rgba(46,204,113,0.18)"),
    "blue":  ("#1a73e8", "rgba(26,115,232,0.09)", "rgba(26,115,232,0.18)"),
}


# ── Not-enough-data guards ──────────────────────────────────────────────────────
# App-wide: below this many recorded observations for a SPECIFIC activity/
# milestone/BU, show "not enough data" instead of a real-looking but unstable
# number. Always pass the actual recorded count for the thing being shown —
# never the overall filtered flight total, and never another item's count.
MIN_SAMPLE_SIZE = 50


def has_enough_data(n, min_n: int = MIN_SAMPLE_SIZE) -> bool:
    """True if `n` (the recorded-observation count for one specific
    activity/milestone/BU) meets the reliability floor."""
    try:
        return n is not None and int(n) >= min_n
    except (TypeError, ValueError):
        return False


def no_data_metric(label: str, n, min_n: int = MIN_SAMPLE_SIZE) -> None:
    """Drop-in replacement for st.metric when there isn't enough data."""
    n_disp = int(n) if pd.notna(n) else 0
    st.metric(label, "—", help=f"Not enough data — only {n_disp:,} recorded (need {min_n}+).")


def no_data_card(label: str, n, icon: str = "📊", min_n: int = MIN_SAMPLE_SIZE) -> None:
    """Custom-HTML 'not enough data' card matching the app's existing card
    shell (same shape as the BU/activity cards on Activity Analysis, BU
    Impact, and Home), so a no-data card reads as a sibling of the real
    cards rather than a jarring inline warning."""
    n_disp = int(n) if pd.notna(n) else 0
    st.markdown(
        f"""<div style="background:{card_bg()};border-left:4px solid #6b7fa3;
            border-radius:10px;padding:14px 16px;margin-bottom:8px">
          <div style="font-size:.8rem;color:{card_sub()};text-transform:uppercase;letter-spacing:1px">
            {icon} {label}
          </div>
          <div style="font-size:1.1rem;font-weight:700;color:{card_sub()};margin:6px 0">
            No data
          </div>
          <div style="font-size:.72rem;color:{card_sub()}">
            Only {n_disp:,} recorded — need {min_n}+ to show a reliable figure.
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def no_data_info(label: str, n=0, min_n: int = MIN_SAMPLE_SIZE) -> None:
    """st.info-style fallback for a chart/section that has too little data
    to render meaningfully."""
    n_disp = int(n) if pd.notna(n) else 0
    st.info(f"📊 Not enough data for {label} — only {n_disp:,} recorded (need {min_n}+).")


def insight_card(problem: str, impact: str, action: str,
                 icon: str = "⚠️", severity: str = "amber") -> None:
    """
    Full 3-row insight card: Problem / Impact / Action.
    severity: "red" | "amber" | "green" | "blue"
    """
    c, bg, _ = _COLORS.get(severity, _COLORS["amber"])
    st.markdown(f"""
    <div style="background:{bg};border-left:4px solid {c};border-radius:0 10px 10px 0;
                padding:14px 20px;margin:0 0 18px 0">
      <div style="font-size:0.7rem;font-weight:700;color:{c};text-transform:uppercase;
                  letter-spacing:1.2px;margin-bottom:10px">{icon}&nbsp;&nbsp;Insight</div>
      <div style="display:grid;grid-template-columns:72px 1fr;gap:6px;font-size:0.875rem;
                  align-items:start">
        <span style="color:#8892a4;font-size:0.65rem;font-weight:700;text-transform:uppercase;
                     padding-top:3px;letter-spacing:.8px">Problem</span>
        <span style="color:#dde8ff;line-height:1.5">{problem}</span>
        <span style="color:#8892a4;font-size:0.65rem;font-weight:700;text-transform:uppercase;
                     padding-top:3px;letter-spacing:.8px">Impact</span>
        <span style="color:#c5d3f0;line-height:1.5">{impact}</span>
        <span style="color:{c};font-size:0.65rem;font-weight:700;text-transform:uppercase;
                     padding-top:3px;letter-spacing:.8px">Action</span>
        <span style="color:#d5ecd5;font-style:italic;line-height:1.5">{action}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def insight_strip(text: str, severity: str = "amber") -> None:
    """Single-line callout — sits directly above a chart or table."""
    c, bg, _ = _COLORS.get(severity, _COLORS["amber"])
    st.markdown(
        f'<div style="background:{bg};border:1px solid {c}50;border-radius:6px;'
        f'padding:7px 14px;margin:4px 0 8px 0;font-size:0.84rem;color:{c}">{text}</div>',
        unsafe_allow_html=True,
    )


def multi_insight_row(insights: list) -> None:
    """
    Display 2-4 compact insight chips in a horizontal row.
    Each item: {"text": str, "severity": str}
    """
    c, bg, border = _COLORS.get("blue", _COLORS["blue"])
    cols = st.columns(len(insights))
    for col, ins in zip(cols, insights):
        sev = ins.get("severity", "amber")
        cc, cbg, _ = _COLORS.get(sev, _COLORS["amber"])
        col.markdown(
            f'<div style="background:{cbg};border:1px solid {cc}50;border-radius:8px;'
            f'padding:10px 14px;font-size:0.82rem;color:{cc};line-height:1.5">'
            f'{ins["text"]}</div>',
            unsafe_allow_html=True,
        )


# ── Per-page stat helpers ──────────────────────────────────────────────────────

def compute_overview_stats(valid: pd.DataFrame) -> dict:
    """Stats needed for Overview page insights."""
    if valid.empty:
        return {}
    total = len(valid)
    delay_pct  = (valid["Target_Departure_Delay_Class"] == "Delayed").mean() * 100
    ontime_pct = (valid["Target_Departure_Delay_Class"] == "On-Time").mean() * 100

    worst_c = best_c = "N/A"
    worst_r = best_r = 0.0
    if "identification_carrierCode" in valid.columns:
        grp = valid.groupby("identification_carrierCode")
        sized = grp.filter(lambda x: len(x) >= 20)
        if not sized.empty:
            cr = (sized.groupby("identification_carrierCode")["Target_Departure_Delay_Class"]
                  .apply(lambda x: (x == "Delayed").mean() * 100))
            worst_c, worst_r = cr.idxmax(), cr.max()
            best_c,  best_r  = cr.idxmin(), cr.min()

    nb_rate = wb_rate = 0.0
    if "aircraft_bodyType" in valid.columns:
        bt = (valid.groupby("aircraft_bodyType")["Target_Departure_Delay_Class"]
              .apply(lambda x: (x == "Delayed").mean() * 100))
        nb_rate = float(bt.get("Narrowbody", 0))
        wb_rate = float(bt.get("Widebody",   0))

    return {
        "total": total, "delay_pct": delay_pct, "ontime_pct": ontime_pct,
        "worst_c": worst_c, "worst_r": worst_r,
        "best_c": best_c,  "best_r":  best_r,
        "nb_rate": nb_rate, "wb_rate": wb_rate,
    }


def compute_when_stats(valid: pd.DataFrame) -> dict:
    """Stats needed for When Delays Happen page insights."""
    if valid.empty or "Hour_of_Day" not in valid.columns or "Day_of_Week" not in valid.columns:
        return {}

    hm = (valid.groupby(["Day_of_Week", "Hour_of_Day"])["Target_Departure_Delay_Class"]
          .agg([("rate", lambda x: (x == "Delayed").mean() * 100), ("n", "count")])
          .reset_index())
    if hm.empty:
        return {}
    # Require at least 30 flights per bucket to avoid misleading 100% rates from tiny samples
    reliable = hm[hm["n"] >= 30]
    if reliable.empty:
        reliable = hm  # fall back if no bucket has 30+ flights
    peak_idx  = reliable["rate"].idxmax()
    peak_day  = reliable.loc[peak_idx, "Day_of_Week"]
    peak_hour = int(reliable.loc[peak_idx, "Hour_of_Day"])
    peak_rate = reliable.loc[peak_idx, "rate"]

    dow = (valid.groupby("Day_of_Week")["Target_Departure_Delay_Class"]
           .apply(lambda x: (x == "Delayed").mean() * 100))
    best_day  = dow.idxmin()
    worst_day = dow.idxmax()

    overall = (valid["Target_Departure_Delay_Class"] == "Delayed").mean() * 100

    return {
        "peak_day": peak_day, "peak_hour": peak_hour, "peak_rate": peak_rate,
        "overall": overall,
        "best_day": best_day,  "best_day_rate":  float(dow.min()),
        "worst_day": worst_day, "worst_day_rate": float(dow.max()),
    }


def compute_activity_stats(bu_stats: pd.DataFrame) -> dict:
    """Stats for Activity Analysis overview level."""
    if bu_stats.empty:
        return {}
    worst = bu_stats.sort_values("avg_delay", ascending=False).iloc[0]
    return {
        "worst_label": worst.get("label", "N/A"),
        "worst_icon":  worst.get("icon",  ""),
        "worst_delay": float(worst.get("avg_delay", 0)),
        "worst_rate":  float(worst.get("avg_late_rate", 0)) * 100,
        "worst_act":   worst.get("worst_activity", "N/A"),
    }
