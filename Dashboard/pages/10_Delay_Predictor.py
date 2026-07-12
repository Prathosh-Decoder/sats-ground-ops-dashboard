"""
pages/6_Delay_Predictor.py
ML-powered departure delay predictor.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.loader   import load_data, load_model
from utils.style    import inject_css, chart_template, chart_fc, is_light
from utils.insights import insight_card, insight_strip

st.set_page_config(page_title="Delay Predictor | SATS", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")
inject_css()

TEMPLATE  = chart_template()
FC        = chart_fc()
COLOR_MAP = {"On-Time": "#2ecc71", "Acceptable": "#f39c12", "Delayed": "#e74c3c"}

# ─── Data & Model ────────────────────────────────────────────────────────────
df    = load_data()
model = load_model()

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("## 🎯 Delay Predictor")
st.caption("Enter flight parameters to get an instant ML-powered delay risk assessment.")
st.divider()

if model is None:
    st.error("Model file not found. Please run `python prepare_data.py` first.")
    st.stop()

pipeline      = model["pipeline"]
le            = model["label_encoder"]
feature_cols  = model["feature_columns"]

# ─── Get unique values for dropdowns ─────────────────────────────────────────
def get_options(col, default_list=None):
    if col in df.columns:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        return [v for v in vals if v not in ("nan", "None", "")]
    return default_list or []

carriers     = get_options("identification_carrierCode", ["SQ", "MI", "TR", "SL", "FD"])
body_types   = get_options("aircraft_bodyType",          ["Narrowbody", "Widebody"])
aircraft_icao = get_options("aircraft_typeICAO",         ["B738", "A320", "B77W", "A333"])
terminals    = get_options("origin_terminal",            ["T1", "T2", "T3"])
destinations = get_options("destination_iata",           ["KUL", "BKK", "HKG", "SYD"])
day_of_weeks = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
month_names  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ─── Page insight ────────────────────────────────────────────────────────────
insight_strip(
    "💡 Model trained on 25,000+ historic flights including ⚡ lightning warning features (~56% overall accuracy). "
    "Delayed-class precision ~63% — incoming flight delay, available ground time, and active LW are the strongest predictors. "
    "Set LW conditions accurately for the most reliable result. "
    "<b>Note:</b> LW inputs only affect predictions after running <code>python prepare_data.py</code> to retrain.",
    severity="blue",
)

# ─── Input Form ──────────────────────────────────────────────────────────────
with st.form("predict_form"):
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("#### Flight Details")
        carrier     = st.selectbox("Carrier", options=carriers, index=0)
        aircraft_tp = st.selectbox("Aircraft Type (ICAO)", options=aircraft_icao, index=0)
        body_type   = st.selectbox("Aircraft Body", options=body_types, index=0)
        terminal    = st.selectbox("Origin Terminal", options=terminals, index=0)
        dest        = st.selectbox("Destination", options=destinations, index=0)
        hour        = st.slider("Departure Hour (UTC)", 0, 23, 10, help="Scheduled departure hour in UTC — the model was trained on UTC timestamps. Singapore (SGT) is UTC+8.")
        day_of_week = st.selectbox("Day of Week", options=day_of_weeks, index=0)
        month       = st.selectbox("Month", options=list(month_names.values()), index=0)

    with right_col:
        st.markdown("#### Operational Conditions")
        incoming_delay    = st.slider("Incoming Flight Delay (min)", -30, 120, 0,
                                       help="How many minutes late the inbound aircraft arrived.")
        avail_ground_time = st.slider("Available Ground Time (min)", 0, 180, 90,
                                       help="Time between inbound arrival and scheduled departure.")
        min_ground_time   = st.slider("Minimum Required Ground Time (min)", 0, 120, 45,
                                       help="Minimum time needed for full ground handling.")
        remote_bay        = st.toggle("Remote Bay (no jet bridge)", value=False)
        is_weekend        = 1 if day_of_week in ("Saturday", "Sunday") else 0

        st.markdown("**Special Handling Passengers**")
        sh_wchr = st.number_input("Wheelchair (ramp) — WCHR", 0, 50, 0)
        sh_umnr = st.number_input("Unaccompanied Minors — UMNR", 0, 20, 0)
        sh_meda = st.number_input("Medical Cases — MEDA", 0, 10, 0)
        sh_maas = st.number_input("Meet & Assist — MAAS", 0, 20, 0)

        st.markdown("**⚡ Lightning Warning Conditions**")
        lw_day         = st.toggle("LW warning issued today", value=False,
                                   help="Is there at least one active LW issued for this date?")
        lw_at_dep      = st.toggle("LW active at departure time", value=False,
                                   help="Is the warning window in effect at the scheduled departure hour?")
        lw_in_gw       = st.toggle("LW overlaps ground-handling window", value=False,
                                   help="Does the LW window cover any part of the aircraft ground time?")
        lw_overlap_min = st.slider("LW overlap with ground window (min)", 0, 120, 0,
                                   help="Total minutes the LW covers the ground-handling window.")
        lw_count       = st.number_input("LW warnings on this date (total)", 0, 20, 0)
        lw_total_mins  = st.slider("Total LW duration today (min)", 0, 300, 0,
                                   help="Sum of all LW warning durations on this date.")
        lw_mins_since  = st.slider("Mins since last warning ended (cool-down)", 0, 480, 60,
                                   help="Minutes since the last lightning warning ended before scheduled departure. Set to 0 if warning is still active.")

    predict_btn = st.form_submit_button("🔮 PREDICT DELAY RISK", use_container_width=True)

# ─── Build Input Row ─────────────────────────────────────────────────────────
month_num = [k for k, v in month_names.items() if v == month]
month_num = month_num[0] if month_num else 1

ground_ratio = (avail_ground_time / min_ground_time) if min_ground_time > 0 else 999.0
is_deficient = 1 if avail_ground_time < min_ground_time else 0

input_dict = {
    "identification_carrierCode":  carrier,
    "aircraft_bodyType":           body_type,
    "aircraft_typeICAO":           aircraft_tp,
    "origin_terminal":             terminal,
    "destination_iata":            dest,
    "Day_of_Week":                 day_of_week,
    "status_isRemoteBay":          str(remote_bay),
    "Available_Ground_Time_mins":  float(avail_ground_time),
    "Ground_Time_Ratio":           float(ground_ratio),
    "Incoming_Delay_mins":         float(incoming_delay),
    "Is_Ground_Time_Deficient":    float(is_deficient),
    "Is_Weekend":                  float(is_weekend),
    "Hour_of_Day":                 float(hour),
    "Month":                       float(month_num),
    "status_minGroundTime":        float(min_ground_time),
    "specialHandling_WCHR":        float(sh_wchr),
    "specialHandling_UMNR":        float(sh_umnr),
    "specialHandling_MEDA":        float(sh_meda),
    "specialHandling_MAAS":        float(sh_maas),
    # New interaction term
    "Delay_Pressure_Score":        float(incoming_delay * is_deficient),
    # Lightning Warning features (aligned to new model)
    "LW_Day_Had_Warning":          float(lw_day),
    "LW_Active_At_Sched_Departure":float(lw_at_dep),
    "LW_Active_During_Ground_Time":float(lw_in_gw),
    "LW_Overlap_With_Ground_Window_Mins": float(lw_overlap_min),
    "LW_Count_On_Date":            float(lw_count),
    "Total_LW_Mins_On_Date":       float(lw_total_mins),
    "Mins_Since_Last_LW_Before_Dep":float(0.0 if lw_at_dep else (lw_mins_since if lw_day else -999.0)),
}

if predict_btn:
    # Build input DataFrame aligned to feature_cols
    input_df = pd.DataFrame([input_dict])

    # Add all missing columns from feature_cols with default values
    for col in feature_cols:
        if col not in input_df.columns:
            # Guess numeric vs categorical
            sample = df[col].dropna() if col in df.columns else pd.Series(dtype="object")
            if not sample.empty and pd.api.types.is_numeric_dtype(sample):
                input_df[col] = -999.0
            else:
                input_df[col] = "Unknown"

    input_df = input_df[feature_cols]

    # Coerce types
    numeric_feats     = model.get("numeric_feats", [])
    categorical_feats = model.get("categorical_feats", [])

    for c in numeric_feats:
        if c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce").fillna(-999)
    for c in categorical_feats:
        if c in input_df.columns:
            input_df[c] = input_df[c].astype(str)

    try:
        proba  = pipeline.predict_proba(input_df)[0]
        pred   = pipeline.predict(input_df)[0]
        pred_label = le.inverse_transform([pred])[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Map probabilities to class labels
    class_labels = le.classes_
    prob_dict    = dict(zip(class_labels, proba))

    pred_color = COLOR_MAP.get(pred_label, "#9e9e9e")

    st.divider()
    st.markdown("## Prediction Result")

    # Big badge
    _badge_bg_light = {"On-Time": "#eafaf1", "Acceptable": "#fef9e7", "Delayed": "#fdecea"}
    _badge_bg_dark  = {"On-Time": "#0d2e0d", "Acceptable": "#2e1f00", "Delayed": "#2e0d0d"}
    _badge_bg = _badge_bg_light.get(pred_label, "#f0f4fb") if is_light() else _badge_bg_dark.get(pred_label, "#1e2130")
    _badge_sub = "#5a7090" if is_light() else "#aaa"
    badge_border = COLOR_MAP.get(pred_label, "#9e9e9e")

    st.markdown(f"""
    <div style="background:{_badge_bg};border:3px solid {badge_border};
         border-radius:14px;padding:28px 32px;text-align:center;max-width:480px;margin:0 auto 28px auto;">
        <div style="font-size:0.9rem;color:{_badge_sub};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">
            Predicted Departure Status
        </div>
        <div style="font-size:3.4rem;font-weight:900;color:{pred_color};">{pred_label.upper()}</div>
        <div style="font-size:1rem;color:{_badge_sub};margin-top:6px;">
            {prob_dict.get(pred_label, 0)*100:.1f}% model confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bar chart
    st.markdown("### Probability Breakdown")
    ordered   = ["On-Time", "Acceptable", "Delayed"]
    prob_vals = [prob_dict.get(c, 0) * 100 for c in ordered]
    bar_colors = [COLOR_MAP[c] for c in ordered]

    prob_fig = go.Figure(go.Bar(
        x=prob_vals,
        y=ordered,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in prob_vals],
        textposition="outside",
        textfont=dict(size=13, color=FC),
    ))
    prob_fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Probability (%)", range=[0, 110]),
        yaxis=dict(title=""),
        height=220,
        margin=dict(t=20, b=40, l=80, r=80),
        showlegend=False,
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    # Gauge / Risk meter
    delay_prob = prob_dict.get("Delayed", 0) * 100
    gauge_color = "#e74c3c" if delay_prob > 50 else ("#f39c12" if delay_prob > 30 else "#2ecc71")

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=delay_prob,
        title={"text": "Delay Risk Score", "font": {"color": FC, "size": 16}},
        number={"suffix": "%", "font": {"color": gauge_color, "size": 36}},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#4a5568"),
            bar=dict(color=gauge_color, thickness=0.3),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                {"range": [0, 30],  "color": "rgba(46,204,113,0.15)"},
                {"range": [30, 55], "color": "rgba(243,156,18,0.15)"},
                {"range": [55, 100],"color": "rgba(231,76,60,0.15)"},
            ],
            threshold=dict(line=dict(color=FC, width=2), thickness=0.75, value=50),
        ),
    ))
    gauge_fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(t=30, b=10, l=20, r=20),
        font=dict(color=FC),
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Risk Factors
    st.markdown("### Key Risk Factors")
    risk_flags = []

    if lw_in_gw or lw_at_dep:
        overlap_desc = (f" {lw_overlap_min} min LW overlap with ground window." if lw_overlap_min > 0 else "")
        risk_flags.append(("🚨", "⚡ Lightning Warning Active",
                           f"Ground operations suspended/slowed during active LW.{overlap_desc} "
                           f"{'LW directly covers the ground-handling window.' if lw_in_gw else 'LW active at scheduled departure.'}"))
    elif lw_day:
        risk_flags.append(("⚠️", "⚡ Lightning Warning Day",
                           f"{lw_count} LW warning(s) issued today ({lw_total_mins} min total). "
                           "Ground conditions may be disrupted."))

    if avail_ground_time < min_ground_time:
        risk_flags.append(("🚨", "Tight Ground Time",
                           f"Only {avail_ground_time} min available, need {min_ground_time} min minimum."))
    elif avail_ground_time < min_ground_time * 1.2:
        risk_flags.append(("⚠️", "Ground Time Marginal",
                           f"Only {(avail_ground_time - min_ground_time):.0f} min buffer above minimum."))

    if incoming_delay > 15:
        risk_flags.append(("🚨", "Severely Late Incoming Flight",
                           f"Inbound is {incoming_delay} min late, severely reducing turnaround time."))
    elif incoming_delay > 5:
        risk_flags.append(("⚠️", "Late Incoming Flight",
                           f"Inbound is {incoming_delay} min late."))

    if remote_bay:
        risk_flags.append(("⚠️", "Remote Bay",
                           "Bus boarding/alighting adds time vs jet bridge."))

    if sh_wchr + sh_umnr + sh_meda > 5:
        risk_flags.append(("ℹ️", "Multiple Special Handling Passengers",
                           f"Total {sh_wchr + sh_umnr + sh_meda} passengers requiring special assistance."))

    if hour in (6, 7, 8, 17, 18, 19):
        risk_flags.append(("ℹ️", "Peak Hour Departure",
                           "Historically higher delay rates at this hour of day."))

    if risk_flags:
        _rfbg_light = {"🚨": "#fdecea", "⚠️": "#fef9e7", "ℹ️": "#e8f4fd"}
        _rfbg_dark  = {"🚨": "#3d0d0d", "⚠️": "#3d2e0d", "ℹ️": "#1a2a3a"}
        for icon, title, desc in risk_flags:
            color  = _rfbg_light.get(icon, "#f0f4fb") if is_light() else _rfbg_dark.get(icon, "#1a1f2e")
            border = "#e74c3c" if icon == "🚨" else ("#f39c12" if icon == "⚠️" else "#1a73e8")
            st.markdown(f"""
            <div style="background:{color};border-left:4px solid {border};border-radius:6px;
                 padding:10px 16px;margin-bottom:8px;">
                {icon} <b>{title}</b> — {desc}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No major risk factors identified. Flight conditions look favourable.")

    st.divider()
    _sev  = "red" if delay_prob > 50 else "amber" if delay_prob > 30 else "green"
    _main = risk_flags[0][1] if risk_flags else "No critical risk factors detected"
    _act  = ("Request additional ground crew or extend turnaround buffer before departure."
             if delay_prob > 50 else
             "Monitor turnaround progress closely — conditions are marginal."
             if delay_prob > 30 else
             "Conditions look favourable. Standard turnaround plan should suffice.")
    insight_card(
        problem=f"Model predicts **{pred_label}** — delay probability is **{delay_prob:.0f}%**.",
        impact=f"Primary risk driver: {_main}.",
        action=_act,
        severity=_sev,
        icon="🎯",
    )
