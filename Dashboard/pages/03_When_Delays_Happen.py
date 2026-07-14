"""
pages/2_When_Delays_Happen.py
Temporal analysis — heatmaps, day-of-week, month, IQR box plot, time-band table.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.loader       import load_data, render_date_filters, merge_lw_features, load_lw_data
from utils.style        import inject_css, chart_template, chart_fc
from utils.insights     import insight_card, insight_strip, compute_when_stats
from utils.crossfilter  import init_xf, apply_xf, render_xf_bar, handle_selection, get_xf, bar_colors

st.set_page_config(page_title="When Delays Happen | SATS", page_icon="⏰", layout="wide", initial_sidebar_state="expanded")
inject_css()

TEMPLATE  = chart_template()
FC        = chart_fc()
COLOR_MAP = {"On-Time": "#00d977", "Acceptable": "#ffc107", "Delayed": "#ff4757"}

# ─── Data ────────────────────────────────────────────────────────────────────
df    = load_data()
df    = render_date_filters(df, page_key="when")
df    = merge_lw_features(df)
init_xf()
df    = apply_xf(df)
valid = df.dropna(subset=["Target_Departure_Delay_Class"])
valid = valid[valid["Target_Departure_Delay_Class"] != "nan"]

_, lw_daily = load_lw_data()
lw_has_data = len(lw_daily) > 0

if "Hour_of_Day" not in valid.columns or "Day_of_Week" not in valid.columns:
    st.error("Time feature columns are missing. Please re-run prepare_data.py.")
    st.stop()

if valid.empty:
    st.warning("No flights match the current filters.")
    st.stop()

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px">
  <div style="font-size:1.6rem;font-weight:800;color:#dde8ff;letter-spacing:-0.5px">⏰ When Delays Happen</div>
  <div style="font-size:0.78rem;color:#6b7fa3;margin-top:2px">
    Pinpoint the hours and days where delay risk peaks.
  </div>
</div>
""", unsafe_allow_html=True)
render_xf_bar()

# ─── Page-level Insight ───────────────────────────────────────────────────────
_ws = compute_when_stats(valid)
if _ws:
    _sgt = (_ws["peak_hour"] + 8) % 24
    _uplift = _ws["peak_rate"] - _ws["overall"]
    insight_card(
        problem=(f"The single worst delay window is **{_ws['peak_day']} at {_ws['peak_hour']:02d}:00 UTC "
                 f"({_sgt:02d}:00 SGT)** — {_ws['peak_rate']:.0f}% of flights at that slot are delayed."),
        impact=(f"That is {_uplift:.0f} percentage points above the overall average of {_ws['overall']:.0f}%. "
                f"**{_ws['worst_day']}** is consistently the worst day "
                f"({_ws['worst_day_rate']:.0f}% delayed) while **{_ws['best_day']}** is the best "
                f"({_ws['best_day_rate']:.0f}% delayed)."),
        action=(f"Pre-position an additional ground crew shift before {_ws['peak_hour']:02d}:00 UTC on "
                f"{_ws['peak_day']}s. Avoid scheduling maintenance windows or staff breaks during "
                f"{_ws['peak_hour']:02d}:00–{(_ws['peak_hour']+2)%24:02d}:00 UTC."),
        icon="⏰",
        severity="red" if _ws["peak_rate"] > 55 else "amber",
    )

# ─── Hero: Heatmap ───────────────────────────────────────────────────────────
st.markdown("### Delay Rate Heatmap — Hour of Day vs Day of Week")
st.caption("Colour intensity = % of flights that were delayed. Darker red = higher delay risk.")

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

heatmap_data = (
    valid.groupby(["Day_of_Week", "Hour_of_Day"])["Target_Departure_Delay_Class"]
    .apply(lambda x: (x == "Delayed").mean() * 100)
    .reset_index()
    .rename(columns={"Target_Departure_Delay_Class": "delay_rate"})
)

# Pivot to matrix
pivot = heatmap_data.pivot(index="Day_of_Week", columns="Hour_of_Day", values="delay_rate")
pivot = pivot.reindex(day_order).fillna(0)

heatmap_fig = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[f"{h:02d}:00" for h in pivot.columns],
    y=pivot.index.tolist(),
    colorscale=[
        [0.0,  "#1a3a1a"],
        [0.35, "#2ecc71"],
        [0.55, "#f39c12"],
        [0.75, "#e74c3c"],
        [1.0,  "#8b0000"],
    ],
    text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=10),
    hoverongaps=False,
    colorbar=dict(title="% Delayed", ticksuffix="%"),
))
heatmap_fig.update_layout(
    template=TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Hour of Day",
    yaxis_title="",
    height=380,
    margin=dict(t=20, b=40, l=80, r=20),
)
st.plotly_chart(heatmap_fig, use_container_width=True)

# Callout: peak window
peak_idx = heatmap_data.loc[heatmap_data["delay_rate"].idxmax()]
peak_day  = peak_idx["Day_of_Week"]
peak_hour = int(peak_idx["Hour_of_Day"])
peak_rate = peak_idx["delay_rate"]

st.error(f"🚨 **Peak delay window:** {peak_day} at {peak_hour:02d}:00 — {peak_rate:.0f}% of flights are delayed")

st.divider()

# ─── Row 2: Day of Week + Month ───────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Delay Rate by Day of Week")
    st.caption("Click a bar to cross-filter all charts.")
    if _ws:
        insight_strip(
            f"⚡ <b>{_ws['worst_day']}</b> is the hardest day ({_ws['worst_day_rate']:.0f}% delayed) "
            f"vs <b>{_ws['best_day']}</b> the easiest ({_ws['best_day_rate']:.0f}%). "
            f"That is a {_ws['worst_day_rate'] - _ws['best_day_rate']:.0f} pp gap — "
            f"review staffing levels for {_ws['worst_day']}s specifically.",
            severity="amber",
        )
    dow_stats = (
        valid.groupby("Day_of_Week")["Target_Departure_Delay_Class"]
        .apply(lambda x: (x == "Delayed").mean() * 100)
        .reindex(day_order)
        .reset_index()
    )
    dow_stats.columns = ["Day", "delay_rate"]

    _active_day = get_xf("day_of_week")
    _dow_colors = bar_colors(dow_stats["Day"], _active_day)
    _dow_marker = dict(color=_dow_colors) if _dow_colors else dict(
        color=dow_stats["delay_rate"],
        colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
        showscale=False,
    )

    dow_fig = go.Figure(go.Bar(
        x=dow_stats["Day"],
        y=dow_stats["delay_rate"],
        marker=_dow_marker,
        text=[f"{v:.1f}%" for v in dow_stats["delay_rate"]],
        textposition="outside",
    ))
    dow_fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Delay Rate (%)",
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        yaxis=dict(range=[0, dow_stats["delay_rate"].max() * 1.2]),
    )
    _dow_evt = st.plotly_chart(dow_fig, use_container_width=True,
                               on_select="rerun", key="xf_when_dow")
    handle_selection(_dow_evt, "day_of_week", axis="x")

with col_right:
    st.markdown("#### Delay Rate by Month")
    st.caption("Click a bar to cross-filter all charts.")
    if _ws and "Month" in valid.columns:
        month_names_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        _mr = (valid.groupby("Month")["Target_Departure_Delay_Class"]
               .apply(lambda x: (x == "Delayed").mean() * 100))
        if not _mr.empty:
            _pm, _bm = _mr.idxmax(), _mr.idxmin()
            insight_strip(
                f"📅 Peak month: <b>{month_names_map.get(_pm,'?')}</b> ({_mr.max():.0f}% delayed). "
                f"Best month: <b>{month_names_map.get(_bm,'?')}</b> ({_mr.min():.0f}%). "
                f"{'Seasonal pattern — review resourcing plan for peak months.' if _mr.max() - _mr.min() > 8 else 'Delay rate is fairly consistent across months.'}",
                severity="amber" if _mr.max() - _mr.min() > 8 else "blue",
            )
    if "Month" in valid.columns:
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        month_stats = (
            valid.groupby("Month")["Target_Departure_Delay_Class"]
            .apply(lambda x: (x == "Delayed").mean() * 100)
            .reset_index()
        )
        month_stats.columns = ["Month", "delay_rate"]
        month_stats["Month_Name"] = month_stats["Month"].map(month_names)

        _active_month = get_xf("month")
        _active_month_name = month_names.get(int(_active_month), None) if _active_month is not None else None
        _month_colors = bar_colors(month_stats["Month_Name"], _active_month_name)
        _month_marker = dict(color=_month_colors) if _month_colors else dict(
            color=month_stats["delay_rate"],
            colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
            showscale=False,
        )

        month_fig = go.Figure(go.Bar(
            x=month_stats["Month_Name"],
            y=month_stats["delay_rate"],
            marker=_month_marker,
            text=[f"{v:.1f}%" for v in month_stats["delay_rate"]],
            textposition="outside",
        ))
        month_fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Delay Rate (%)",
            height=300,
            margin=dict(t=20, b=40, l=40, r=20),
            yaxis=dict(range=[0, month_stats["delay_rate"].max() * 1.2]),
        )
        _month_evt = st.plotly_chart(month_fig, use_container_width=True,
                                     on_select="rerun", key="xf_when_month")
        handle_selection(_month_evt, "month", axis="x")

st.divider()

# ─── Row 3: IQR Box Plot by Hour ─────────────────────────────────────────────
st.markdown("#### Delay Spread by Hour of Day")
st.caption("Box shows Q1–Q3 range. Whiskers extend to 1.5× IQR. Circles are outliers.")

if "Target_Departure_Delay_mins" in valid.columns:
    delay_vals = valid[["Hour_of_Day", "Target_Departure_Delay_mins"]].dropna()
    delay_clipped = delay_vals.copy()
    delay_clipped["Target_Departure_Delay_mins"] = delay_clipped["Target_Departure_Delay_mins"].clip(-30, 90)

    box_hour = go.Figure()
    for hour in range(24):
        subset = delay_clipped[delay_clipped["Hour_of_Day"] == hour]["Target_Departure_Delay_mins"]
        if subset.empty:
            continue
        box_hour.add_trace(go.Box(
            y=subset,
            name=f"{hour:02d}",
            marker_color="#1a73e8",
            line_color="#1a73e8",
            fillcolor="rgba(26,115,232,0.3)",
            boxpoints=False,
            showlegend=False,
        ))
    box_hour.add_hline(y=0, line_dash="dash", line_color="#2ecc71", line_width=1.5)
    box_hour.add_hline(y=4, line_dash="dot",  line_color="#f39c12", line_width=1.5)
    box_hour.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Hour of Day",
        yaxis_title="Delay (minutes)",
        height=350,
        margin=dict(t=20, b=40, l=50, r=20),
    )
    st.plotly_chart(box_hour, use_container_width=True)

st.divider()

# ─── Row 4: Time Band Stats Table ─────────────────────────────────────────────
st.markdown("#### Delay Statistics by Time of Day")

if "Target_Departure_Delay_mins" in valid.columns:
    bands = [
        ("Early Morning", 0,  6),
        ("Morning",       6,  12),
        ("Afternoon",     12, 18),
        ("Evening",       18, 24),
    ]
    table_rows = []
    for band_name, h_start, h_end in bands:
        mask = valid["Hour_of_Day"].between(h_start, h_end - 1)
        subset = valid.loc[mask, "Target_Departure_Delay_mins"].dropna()
        if subset.empty:
            continue
        table_rows.append({
            "Time Band":           f"{band_name} ({h_start:02d}:00–{h_end:02d}:00)",
            "Flights":             f"{len(subset):,}",
            "Min (min)":           f"{subset.min():.0f}",
            "Q1 (min)":            f"{subset.quantile(0.25):.1f}",
            "Median (min)":        f"{subset.median():.1f}",
            "Q3 (min)":            f"{subset.quantile(0.75):.1f}",
            "Max (min)":           f"{subset.max():.0f}",
            "Mean (min)":          f"{subset.mean():.1f}",
            "Delayed %":           f"{(valid.loc[mask,'Target_Departure_Delay_Class'] == 'Delayed').mean()*100:.1f}%",
        })

    if table_rows:
        band_df = pd.DataFrame(table_rows)
        st.dataframe(band_df, use_container_width=True, hide_index=True)

st.divider()

# ─── Lightning Warning Temporal Analysis ──────────────────────────────────────
st.markdown("### ⚡ Lightning Warning — When They Strike")
st.caption(
    "Lightning warnings (LW) are issued by MSS and suspend all outdoor ground operations. "
    "This section shows which hours and days are most exposed to LW events and their delay impact."
)

if not lw_has_data:
    st.info("No lightning warning data available for the current date range.")
else:
    lw_flights = valid[valid["LW_Day_Had_Warning"] == 1]

    if len(lw_flights) == 0:
        st.info(
            "No flights in the current filter overlap with the LW data period. "
            f"LW data covers {lw_daily['date'].min()} to {lw_daily['date'].max()}."
        )
    else:
        lw_c1, lw_c2 = st.columns(2)

        with lw_c1:
            st.markdown("#### LW Active at Departure — Delay Rate by Hour")
            insight_strip(
                "Yellow bars = hours where at least one flight had an active LW at its scheduled "
                "departure time. Blue = no LW active at departure. "
                "Hours with LW consistently show higher delay rates.",
                severity="amber",
            )
            if "LW_Active_At_Departure" in valid.columns:
                h_lw  = valid[valid["LW_Active_At_Departure"] == 1].groupby("Hour_of_Day").agg(
                    rate=("Target_Departure_Delay_Class", lambda x: (x == "Delayed").mean() * 100),
                    n=("Target_Departure_Delay_Class", "count"),
                ).reset_index()
                h_nolw = valid[valid["LW_Active_At_Departure"] == 0].groupby("Hour_of_Day").agg(
                    rate=("Target_Departure_Delay_Class", lambda x: (x == "Delayed").mean() * 100),
                    n=("Target_Departure_Delay_Class", "count"),
                ).reset_index()

                lw_hour_fig = go.Figure()
                lw_hour_fig.add_trace(go.Bar(
                    x=h_nolw["Hour_of_Day"], y=h_nolw["rate"],
                    name="No LW at departure", marker_color="rgba(41,128,185,0.75)",
                    customdata=h_nolw["n"],
                    hovertemplate="Hour %{x}:00 — No LW<br>Delay rate: %{y:.1f}%<br>Flights: %{customdata}<extra></extra>",
                ))
                lw_hour_fig.add_trace(go.Bar(
                    x=h_lw["Hour_of_Day"], y=h_lw["rate"],
                    name="⚡ LW active at departure", marker_color="rgba(241,196,15,0.90)",
                    customdata=h_lw["n"],
                    hovertemplate="Hour %{x}:00 — LW ACTIVE<br>Delay rate: %{y:.1f}%<br>Flights: %{customdata}<extra></extra>",
                ))
                lw_hour_fig.update_layout(
                    template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="overlay",
                    xaxis=dict(title="UTC Departure Hour", dtick=1),
                    yaxis=dict(title="Delay Rate (%)", range=[0, 100]),
                    height=320, margin=dict(t=20, b=60, l=60, r=20),
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(lw_hour_fig, use_container_width=True)

        with lw_c2:
            st.markdown("#### LW Days vs Non-LW Days — Delay Rate by Day of Week")
            insight_strip(
                "Comparing delay rates on days with active lightning warnings vs clear days, "
                "split by day of week. Taller yellow bars signal higher LW exposure on that day.",
                severity="amber",
            )
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if "Day_of_Week" in valid.columns and "LW_Day_Had_Warning" in valid.columns:
                lw_dow  = lw_flights.groupby("Day_of_Week").agg(
                    rate=("Target_Departure_Delay_Class", lambda x: (x == "Delayed").mean() * 100),
                    n=("Target_Departure_Delay_Class", "count"),
                ).reindex(day_order).reset_index()
                nlw_dow = valid[valid["LW_Day_Had_Warning"] == 0].groupby("Day_of_Week").agg(
                    rate=("Target_Departure_Delay_Class", lambda x: (x == "Delayed").mean() * 100),
                    n=("Target_Departure_Delay_Class", "count"),
                ).reindex(day_order).reset_index()

                dow_lw_fig = go.Figure()
                dow_lw_fig.add_trace(go.Bar(
                    x=nlw_dow["Day_of_Week"], y=nlw_dow["rate"].fillna(0),
                    name="No LW", marker_color="rgba(41,128,185,0.75)",
                    customdata=nlw_dow["n"].fillna(0),
                    hovertemplate="%{x} — No LW<br>%{y:.1f}% delayed<br>%{customdata:.0f} flights<extra></extra>",
                ))
                dow_lw_fig.add_trace(go.Bar(
                    x=lw_dow["Day_of_Week"], y=lw_dow["rate"].fillna(0),
                    name="⚡ LW Day", marker_color="rgba(241,196,15,0.90)",
                    customdata=lw_dow["n"].fillna(0),
                    hovertemplate="%{x} — LW Day<br>%{y:.1f}% delayed<br>%{customdata:.0f} flights<extra></extra>",
                ))
                dow_lw_fig.update_layout(
                    template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="group",
                    xaxis=dict(title=""),
                    yaxis=dict(title="Delay Rate (%)", range=[0, 100]),
                    height=320, margin=dict(t=20, b=60, l=60, r=20),
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(dow_lw_fig, use_container_width=True)

        # ── Heatmap of LW count by month × hour ──────────────────────────────
        st.markdown("#### ⚡ Lightning Warning Activity — Month × Hour Heatmap (SGT)")
        insight_strip(
            "This heatmap shows average LW coverage minutes by calendar month and hour of day (SGT). "
            "Darker yellow = more LW exposure. Singapore's thunderstorm season typically peaks "
            "in the afternoon hours (13:00–19:00 SGT) and during Nov–Jan.",
            severity="blue",
        )

        if "Month" in valid.columns and "LW_Active_At_Departure" in valid.columns:
            lw_hm = (valid.groupby(["Month", "Hour_of_Day"])["LW_Active_At_Departure"]
                     .mean().reset_index()
                     .rename(columns={"LW_Active_At_Departure": "lw_rate"}))
            lw_hm["lw_rate"] = lw_hm["lw_rate"] * 100

            month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            lw_hm["Month_Name"] = lw_hm["Month"].map(month_map)

            piv = lw_hm.pivot(index="Month_Name", columns="Hour_of_Day", values="lw_rate")
            month_order = [month_map[m] for m in sorted(lw_hm["Month"].unique())]
            piv = piv.reindex([m for m in month_order if m in piv.index]).fillna(0)

            if not piv.empty and piv.values.max() > 0:
                lw_hm_fig = go.Figure(go.Heatmap(
                    z=piv.values,
                    x=[f"{h:02d}:00 UTC\n({(h+8)%24:02d}:00 SGT)" for h in piv.columns],
                    y=piv.index.tolist(),
                    colorscale=[[0, "rgba(0,0,0,0)"], [0.01, "#3d2e00"], [0.5, "#f39c12"], [1.0, "#f1c40f"]],
                    text=[[f"{v:.0f}%" if v > 0 else "" for v in row] for row in piv.values],
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    hoverongaps=False,
                    colorbar=dict(title="% flights<br>with LW active", ticksuffix="%"),
                ))
                lw_hm_fig.update_layout(
                    template=TEMPLATE,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Departure Hour",
                    yaxis_title="",
                    height=max(250, len(piv) * 35 + 80),
                    margin=dict(t=20, b=60, l=60, r=20),
                )
                st.plotly_chart(lw_hm_fig, use_container_width=True)
            else:
                st.info("Not enough LW-flight overlap in the current filter to build this heatmap.")
