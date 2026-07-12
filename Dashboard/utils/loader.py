"""
utils/loader.py
Cached data and model loading utilities for the SATS dashboard.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "flights.parquet")
MODEL_PATH   = os.path.join(BASE_DIR, "data", "model.pkl")
LW_PATH      = os.path.join(os.path.dirname(BASE_DIR), "Data", "lw_2024_2025.csv")


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

QUARTER_NAMES = ["Q1 (Jan–Mar)", "Q2 (Apr–Jun)", "Q3 (Jul–Sep)", "Q4 (Oct–Dec)"]


@st.cache_data(show_spinner="Loading flight data...")
def load_data() -> pd.DataFrame:
    """Load the processed flights parquet file."""
    if not os.path.exists(PARQUET_PATH):
        st.error(
            "Processed data not found. Please run `python prepare_data.py` first."
        )
        st.stop()
    df = pd.read_parquet(PARQUET_PATH)

    # Ensure target column is present as string (for easy filtering)
    if "Target_Departure_Delay_Class" in df.columns:
        df["Target_Departure_Delay_Class"] = df["Target_Departure_Delay_Class"].astype(str)
        df["Target_Departure_Delay_Class"] = df["Target_Departure_Delay_Class"].replace("nan", np.nan)

    # Parse datetimes if they came in as objects
    for col in [
        "departure_offBlock.scheduled",
        "departure_offBlock.actual",
        "linkedFlight_arrival.inBlock.actual",
        "linkedFlight_arrival.inBlock.scheduled",
    ]:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Add date dimension columns for filtering
    if "departure_offBlock.scheduled" in df.columns:
        sched = pd.to_datetime(df["departure_offBlock.scheduled"], errors="coerce", utc=True)
        df["_dep_date"]    = sched.dt.date
        df["_dep_month"]   = sched.dt.month        # 1–12
        df["_dep_quarter"] = sched.dt.quarter      # 1–4
        df["_dep_year"]    = sched.dt.year
        df["_dep_week"]    = sched.dt.isocalendar().week.astype("Int64")  # ISO 1–53

    return df


def _multiselect_all(label, options, key):
    """Multiselect that defaults to all options selected."""
    return st.multiselect(label, options=options, default=options, key=key)


def render_date_filters(df: pd.DataFrame, page_key: str = "default") -> pd.DataFrame:
    """
    Renders sidebar filters (date, terminal, aircraft type, destination)
    and returns the filtered DataFrame.
    Pass a unique page_key per page (e.g. "overview", "when", "activity").
    """
    with st.sidebar:
        # ── Date section ──────────────────────────────────────────────────────
        st.markdown("### 📅 Date")

        # Year
        if "_dep_year" in df.columns:
            available_years = sorted(df["_dep_year"].dropna().unique().astype(int))
            sel_year_labels = _multiselect_all(
                "Year",
                options=[str(y) for y in available_years],
                key=f"slicer_year_{page_key}",
            )
            sel_years = [y for y in available_years if str(y) in sel_year_labels]
        else:
            sel_years = []

        # Quarter — restricted to selected years
        if "_dep_quarter" in df.columns:
            _base_q = df[df["_dep_year"].isin(sel_years)] if sel_years else df
            available_quarters = sorted(_base_q["_dep_quarter"].dropna().unique().astype(int))
            quarter_labels     = {q: QUARTER_NAMES[q - 1] for q in available_quarters}
            sel_quarter_labels = _multiselect_all(
                "Quarter",
                options=[quarter_labels[q] for q in available_quarters],
                key=f"slicer_quarter_{page_key}",
            )
            sel_quarters = [q for q, lbl in quarter_labels.items() if lbl in sel_quarter_labels]
        else:
            sel_quarters = []

        # Month — restricted to selected years + quarters
        if "_dep_month" in df.columns:
            _base_m = df.copy()
            if sel_years:    _base_m = _base_m[_base_m["_dep_year"].isin(sel_years)]
            if sel_quarters: _base_m = _base_m[_base_m["_dep_quarter"].isin(sel_quarters)]
            available_months = sorted(_base_m["_dep_month"].dropna().unique().astype(int))
            month_labels     = {m: MONTH_NAMES[m - 1] for m in available_months}
            sel_month_labels = _multiselect_all(
                "Month",
                options=[month_labels[m] for m in available_months],
                key=f"slicer_month_{page_key}",
            )
            sel_months = [m for m, lbl in month_labels.items() if lbl in sel_month_labels]
        else:
            sel_months = []

        # Day — drill-down, only shown when exactly one month is selected
        sel_days = []
        if "_dep_date" in df.columns and len(sel_months) == 1:
            _base_d = df.copy()
            if sel_years:    _base_d = _base_d[_base_d["_dep_year"].isin(sel_years)]
            if sel_quarters: _base_d = _base_d[_base_d["_dep_quarter"].isin(sel_quarters)]
            if sel_months:   _base_d = _base_d[_base_d["_dep_month"].isin(sel_months)]
            available_dates = sorted(_base_d["_dep_date"].dropna().unique())
            if available_dates:
                st.caption("📌 Single month selected — drill into specific days:")
                day_labels     = {d: d.strftime("%a %d") for d in available_dates}
                sel_day_labels = _multiselect_all(
                    "Day",
                    options=[day_labels[d] for d in available_dates],
                    key=f"slicer_day_{page_key}",
                )
                sel_days = [d for d, lbl in day_labels.items() if lbl in sel_day_labels]

        # ── Operations section ────────────────────────────────────────────────
        st.markdown("### ✈️ Operations")

        # Terminal
        if "origin_terminal" in df.columns:
            raw_terminals = sorted(
                df["origin_terminal"].dropna().astype(str)
                .replace({"2": "Terminal 2", "3": "Terminal 3",
                          "T2": "Terminal 2", "T3": "Terminal 3"})
                .unique().tolist()
            )
            raw_terminals = [t for t in raw_terminals if t not in ("nan", "None", "")]
            sel_terminals_label = _multiselect_all(
                "Terminal", options=raw_terminals, key=f"slicer_terminal_{page_key}"
            )
            terminal_map = {"Terminal 2": ["2", "T2", "Terminal 2"],
                            "Terminal 3": ["3", "T3", "Terminal 3"]}
            sel_terminal_raw = []
            for lbl in sel_terminals_label:
                sel_terminal_raw.extend(terminal_map.get(lbl, [lbl]))
        else:
            sel_terminal_raw = []

        # Aircraft body type
        if "aircraft_bodyType" in df.columns:
            body_opts = sorted(
                df["aircraft_bodyType"].dropna().astype(str)
                .unique().tolist()
            )
            body_opts = [b for b in body_opts if b not in ("nan", "None", "")]
            sel_body = _multiselect_all(
                "Aircraft Type", options=body_opts, key=f"slicer_body_{page_key}"
            )
        else:
            sel_body = []

        # Aircraft ICAO type
        if "aircraft_typeICAO" in df.columns:
            icao_opts = sorted(
                df["aircraft_typeICAO"].dropna().astype(str)
                .unique().tolist()
            )
            icao_opts = [i for i in icao_opts if i not in ("nan", "None", "")]
            sel_icao = _multiselect_all(
                "Aircraft Model (ICAO)", options=icao_opts, key=f"slicer_icao_{page_key}"
            )
        else:
            sel_icao = []

        # ── Destination section ────────────────────────────────────────────────
        st.markdown("### 🌏 Destination")

        if "destination_iata" in df.columns:
            dest_opts = sorted(
                df["destination_iata"].dropna().astype(str)
                .unique().tolist()
            )
            dest_opts = [d for d in dest_opts if d not in ("nan", "None", "")]
            sel_dest = _multiselect_all(
                "Destination", options=dest_opts, key=f"slicer_dest_{page_key}"
            )
        else:
            sel_dest = []

        # ── Summary caption ───────────────────────────────────────────────────
        total_before = len(df)

    # ── Apply all filters ─────────────────────────────────────────────────────
    if sel_years:
        df = df[df["_dep_year"].isin(sel_years)]
    if sel_quarters:
        df = df[df["_dep_quarter"].isin(sel_quarters)]
    if sel_months:
        df = df[df["_dep_month"].isin(sel_months)]
    if sel_days:
        df = df[df["_dep_date"].isin(sel_days)]
    if sel_terminal_raw:
        df = df[df["origin_terminal"].astype(str).isin(sel_terminal_raw)]
    if sel_body:
        df = df[df["aircraft_bodyType"].astype(str).isin(sel_body)]
    if sel_icao:
        df = df[df["aircraft_typeICAO"].astype(str).isin(sel_icao)]
    if sel_dest:
        df = df[df["destination_iata"].astype(str).isin(sel_dest)]

    with st.sidebar:
        st.caption(f"{len(df):,} of {total_before:,} flights shown.")

    return df


@st.cache_data(show_spinner="Loading lightning warning data...")
def load_lw_data() -> tuple:
    """
    Load, clean and aggregate lightning warning data.
    Returns (lw_clean, lw_daily) where:
      lw_clean  - one row per unique active LW window with lw_start / lw_end datetimes (SGT naive)
      lw_daily  - daily summary: date, LW_Count_On_Date, Total_LW_Mins_On_Date, LW_Day_Had_Warning
    """
    if not os.path.exists(LW_PATH):
        return pd.DataFrame(), pd.DataFrame()

    raw = pd.read_csv(LW_PATH)
    raw.columns = raw.columns.str.strip()

    # Drop Cancelled and deduplicate
    raw = raw[raw["Status"].isna()].copy()
    raw = raw.drop_duplicates(subset=["LWNo", "DateIssued", "LW_ValidFrom", "LW_ValidTo"])

    raw["date"] = pd.to_datetime(raw["DateIssued"], dayfirst=True).dt.date

    def _to_td(s):
        try:
            h, m, sec = str(s).strip().split(":")
            return pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(float(sec)))
        except Exception:
            return pd.NaT

    raw["td_from"] = raw["LW_ValidFrom"].apply(_to_td)
    raw["td_to"]   = raw["LW_ValidTo"].apply(_to_td)

    base = pd.to_datetime(raw["date"].astype(str))
    raw["lw_start"] = base + raw["td_from"]
    raw["lw_end"]   = base + raw["td_to"]

    raw = raw[raw["lw_end"] > raw["lw_start"]]
    raw["lw_duration_mins"] = (raw["lw_end"] - raw["lw_start"]).dt.total_seconds() / 60

    lw = raw[["LWNo", "date", "lw_start", "lw_end", "lw_duration_mins"]].reset_index(drop=True)

    lw_daily = (
        lw.groupby("date")
        .agg(LW_Count_On_Date=("LWNo", "count"),
             Total_LW_Mins_On_Date=("lw_duration_mins", "sum"))
        .reset_index()
    )
    lw_daily["LW_Day_Had_Warning"] = 1

    return lw, lw_daily


@st.cache_data(show_spinner="Merging lightning warning features...")
def merge_lw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge lightning warning features onto a flight DataFrame.

    Adds columns:
      LW_Count_On_Date                  — number of active LW windows on SGT departure date
      Total_LW_Mins_On_Date             — total minutes of LW coverage on that date
      LW_Day_Had_Warning                — 1 if any active LW that day, else 0
      LW_Active_At_Sched_Departure      — 1 if any LW window contains the scheduled departure time (SGT)
      LW_Overlap_With_Ground_Window_Mins— total overlap minutes between LW windows and ground window
      LW_Active_During_Ground_Time      — 1 if any LW overlaps with [arr_actual, dep_sched] (SGT)
      Mins_Since_Last_LW_Before_Dep     — minutes since the most-recent LW ended before scheduled departure
    """
    lw, lw_daily = load_lw_data()
    _LW_COLS = ["LW_Count_On_Date", "Total_LW_Mins_On_Date", "LW_Day_Had_Warning",
                "LW_Active_At_Sched_Departure", "LW_Overlap_With_Ground_Window_Mins",
                "LW_Active_During_Ground_Time", "Mins_Since_Last_LW_Before_Dep"]
    
    out = df.copy()

    # Drop old LW columns if they exist
    out = out.drop(columns=[c for c in _LW_COLS if c in out.columns], errors="ignore")
    # Also drop legacy columns in case they are present in parquet
    out = out.drop(columns=["LW_Active_At_Departure", "LW_In_Ground_Window", "LW_Overlap_Ground_Mins"], errors="ignore")

    if lw.empty:
        for col in _LW_COLS:
            out[col] = np.nan if col == "Mins_Since_Last_LW_Before_Dep" else 0
        return out

    # Build SGT departure datetime (flight timestamps are UTC-naive; SGT = UTC+8)
    dep_col = "departure_offBlock.scheduled"
    arr_col = "linkedFlight_arrival.inBlock.actual"

    dep_utc = pd.to_datetime(out.get(dep_col), errors="coerce")
    arr_utc = pd.to_datetime(out.get(arr_col), errors="coerce")

    if dep_utc.dt.tz is not None:
        dep_sgt = dep_utc.dt.tz_convert(None) + pd.Timedelta(hours=8)
        arr_sgt = arr_utc.dt.tz_convert(None) + pd.Timedelta(hours=8)
    else:
        dep_sgt = dep_utc + pd.Timedelta(hours=8)
        arr_sgt = arr_utc + pd.Timedelta(hours=8)

    out["_dep_sgt_date"] = dep_sgt.dt.date

    # Tier-1: daily join
    out = out.merge(
        lw_daily.rename(columns={"date": "_dep_sgt_date"}),
        on="_dep_sgt_date", how="left",
    )
    out["LW_Count_On_Date"]      = out["LW_Count_On_Date"].fillna(0).astype(int)
    out["Total_LW_Mins_On_Date"] = out["Total_LW_Mins_On_Date"].fillna(0.0)
    out["LW_Day_Had_Warning"]    = out["LW_Day_Had_Warning"].fillna(0).astype(int)

    # Tier-2: per-flight checks using LW time windows
    lw_by_date = {d: list(zip(g["lw_start"], g["lw_end"])) for d, g in lw.groupby("date")}

    active_at_dep, in_gw, overlap_gw, since_last = [], [], [], []
    for date, dep_ts, arr_ts in zip(out["_dep_sgt_date"], dep_sgt, arr_sgt):
        windows = lw_by_date.get(date, [])
        if not windows or pd.isna(dep_ts):
            active_at_dep.append(0)
            in_gw.append(0)
            overlap_gw.append(0.0)
            since_last.append(np.nan)
            continue

        aad = int(any(s <= dep_ts <= e for s, e in windows))
        active_at_dep.append(aad)

        # Overlap
        if pd.notna(arr_ts) and arr_ts < dep_ts:
            tot_overlap = 0.0
            for s, e in windows:
                ov = min(dep_ts, e) - max(arr_ts, s)
                if ov.total_seconds() > 0:
                    tot_overlap += ov.total_seconds() / 60
            in_gw.append(int(tot_overlap > 0))
            overlap_gw.append(round(tot_overlap, 1))
        else:
            in_gw.append(0)
            overlap_gw.append(0.0)

        # Mins since last warning ended
        if aad == 1:
            since_last.append(0.0)
        else:
            last_end_ts = pd.NaT
            for s, e in windows:
                if e <= dep_ts:
                    if pd.isna(last_end_ts) or e > last_end_ts:
                        last_end_ts = e
            if pd.notna(last_end_ts):
                since_last.append(round((dep_ts - last_end_ts).total_seconds() / 60, 1))
            else:
                since_last.append(np.nan)

    out["LW_Active_At_Sched_Departure"] = active_at_dep
    out["LW_Active_During_Ground_Time"] = in_gw
    out["LW_Overlap_With_Ground_Window_Mins"] = overlap_gw
    out["Mins_Since_Last_LW_Before_Dep"] = since_last
    out.drop(columns=["_dep_sgt_date"], inplace=True, errors="ignore")
    return out


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model() -> dict:
    """Load the trained XGBoost pipeline bundle."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─── Activity Stats ──────────────────────────────────────────────────────────

def get_activity_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with per-activity stats:
    activity, team, count, avg_delay_mins, late_rate, avg_actual_dur, avg_planned_dur
    """
    records = []

    # Duration-based activities
    dur_cols = [c for c in df.columns if c.endswith("_analysis_ActualDuration_mins")]
    for col in dur_cols:
        base = col.replace("_analysis_ActualDuration_mins", "")
        planned_col = f"{base}_analysis_PlannedDuration_mins"
        delay_col   = f"{base}_analysis_Delay_mins"

        parts = base.split("_")
        # base looks like milestone_aic_cleaning
        if len(parts) >= 3:
            team     = parts[1]
            activity = "_".join(parts[2:])
        else:
            team     = "unknown"
            activity = base

        act_dur  = df[col].dropna()
        plan_dur = df[planned_col].dropna() if planned_col in df.columns else pd.Series(dtype=float)
        delay    = df[delay_col].dropna()   if delay_col  in df.columns else pd.Series(dtype=float)

        count = len(act_dur)
        if count < 10:
            continue

        avg_actual  = act_dur.mean()
        avg_planned = plan_dur.mean() if not plan_dur.empty else np.nan
        avg_delay   = delay.mean()    if not delay.empty    else np.nan
        late_rate   = (delay > 0).mean() if not delay.empty else np.nan

        records.append({
            "activity":         f"{team}: {activity.replace('_', ' ').title()}",
            "team":             team,
            "raw_name":         base,
            "count":            count,
            "avg_delay_mins":   avg_delay,
            "late_rate":        late_rate,
            "avg_actual_dur":   avg_actual,
            "avg_planned_dur":  avg_planned,
        })

    # Point milestone delay columns
    point_delay_cols = [
        c for c in df.columns
        if c.endswith("_analysis_Delay_mins") and
        not any(tag in c for tag in ["ActualDuration", "PlannedDuration"])
        and "_analysis_Delay_mins" in c
    ]
    for col in point_delay_cols:
        base   = col.replace("_analysis_Delay_mins", "")
        # base looks like milestone_ramp_manAtBay
        parts  = base.split("_")
        if len(parts) >= 3:
            team     = parts[1]
            activity = "_".join(parts[2:])
        else:
            team     = "unknown"
            activity = base

        delay = df[col].dropna()
        count = len(delay)
        if count < 10:
            continue

        avg_delay = delay.mean()
        late_rate = (delay > 0).mean()
        records.append({
            "activity":        f"{team}: {activity.replace('_', ' ').title()}",
            "team":            team,
            "raw_name":        base,
            "count":           count,
            "avg_delay_mins":  avg_delay,
            "late_rate":       late_rate,
            "avg_actual_dur":  np.nan,
            "avg_planned_dur": np.nan,
        })

    stats = pd.DataFrame(records).drop_duplicates(subset=["activity"])
    return stats.sort_values("avg_delay_mins", ascending=False).reset_index(drop=True)


# ─── Cascade Correlations ────────────────────────────────────────────────────

def get_cascade_correlations(df: pd.DataFrame) -> dict:
    """
    Returns dict of {activity_name: correlation_with_departure_delay}
    using the Target_Departure_Delay_mins column.
    """
    if "Target_Departure_Delay_mins" not in df.columns:
        return {}

    target = df["Target_Departure_Delay_mins"]
    delay_cols = [
        c for c in df.columns
        if "_analysis_Delay_mins" in c and "ActualDuration" not in c and "PlannedDuration" not in c
    ]
    corr_dict = {}
    for col in delay_cols:
        col_data = df[col].dropna()
        common   = target.loc[col_data.index].dropna()
        aligned  = col_data.loc[common.index]
        if len(aligned) > 50:
            corr = aligned.corr(common)
            if not np.isnan(corr):
                name = col.replace("_analysis_Delay_mins", "").replace("milestone_", "").replace("_", " ")
                corr_dict[name] = corr

    return dict(sorted(corr_dict.items(), key=lambda x: abs(x[1]), reverse=True))
