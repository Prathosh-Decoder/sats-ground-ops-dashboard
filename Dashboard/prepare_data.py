"""
prepare_data.py
SATS Ground Operations Dashboard — Data Preparation Pipeline

Loads 4 raw data files, processes and joins them, engineers features,
trains an XGBoost model, and saves flights.parquet + model.pkl to data/.
"""

import os
import sys
import json
import ast
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PATH_OLD_MILESTONE = "/Users/prathosh/Desktop/Sats Project/Data/Old DSM Data/dsmlive_milestone_Nov2025.xlsx"
PATH_OLD_FLIGHT    = "/Users/prathosh/Desktop/Sats Project/Data/Old DSM Data/dsmlive_flight_Nov2025.xlsx"
PATH_NEW_MILESTONE = "/Users/prathosh/Desktop/Sats Project/Data/New DSM Data/dsmlive_GHAMS_01022026_07042026.csv"
PATH_NEW_FLIGHT    = "/Users/prathosh/Desktop/Sats Project/Data/New DSM Data/dsmlive_flight_01022026_07042026.csv"
PATH_NEW2_MILESTONE = "/Users/prathosh/Desktop/Sats Project/Data/dsmlive_GHAMS_19032026_30062026.csv"
PATH_NEW2_FLIGHT    = "/Users/prathosh/Desktop/Sats Project/Data/dsmlive_flight_19032026_30062026.csv"
PATH_LW            = "/Users/prathosh/Desktop/Sats Project/Data/lw_2024_2025.csv"

OUT_PARQUET = os.path.join(DATA_DIR, "flights.parquet")
OUT_MODEL   = os.path.join(DATA_DIR, "model.pkl")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_dict(val):
    """Safely parse stringified dicts/JSON into Python dicts."""
    if val is None:
        return {}
    try:
        if isinstance(val, float) and np.isnan(val):
            return {}
    except TypeError:
        pass
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val or val in ("nan", "None", "null", "NaN"):
            return {}
        try:
            return json.loads(val)
        except (ValueError, TypeError):
            pass
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
    return {}


def flatten_col(df, col, prefix):
    """Parse + json_normalize a nested column, add prefix, concat back."""
    if col not in df.columns:
        return df
    parsed = df[col].apply(parse_dict)
    first_valid = parsed.dropna().iloc[0] if not parsed.dropna().empty else None
    if not isinstance(first_valid, dict):
        return df
    expanded = pd.json_normalize(parsed)
    expanded = expanded.add_prefix(f"{prefix}_")
    return pd.concat([df.drop(columns=[col]), expanded], axis=1)


def flatten_flight(df):
    """Flatten all nested columns in a flight dataframe."""
    nested = ["identification", "transitPoints", "status", "aircraft",
              "arrival", "departure", "origin", "destination", "linkedFlight"]
    for col in nested:
        df = flatten_col(df, col, col)
    # changes is a list – just parse but don't explode
    if "changes" in df.columns:
        df["changes"] = df["changes"].apply(parse_dict)
    return df


def flatten_milestone(df, prefix="milestone"):
    """Flatten milestones and specialHandling columns."""
    for col in ["milestones", "specialHandling"]:
        df = flatten_col(df, col, col)
    # Rename 'milestones_' prefix to 'milestone_' for consistency
    df.columns = [
        c.replace("milestones_", "milestone_") if c.startswith("milestones_") else c
        for c in df.columns
    ]
    return df


def convert_datetimes(df):
    """Auto-detect and convert ISO-Z datetime strings to datetime64[UTC]."""
    text_cols = df.select_dtypes(include=["object"]).columns
    converted = 0
    for col in text_cols:
        sample = df[col].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if isinstance(first, str) and "T" in first and first.endswith("Z"):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            converted += 1
    print(f"  Converted {converted} columns to datetime.")
    return df


def clean_pts(series):
    """Extract first numeric value from PTS strings like '-5|-5' or '+10'."""
    s = series.astype(str).str.split("|").str[0]
    s = s.str.replace("+", "", regex=False)
    s = s.replace(["None", "nan", "NaN", "null", "Null", ""], np.nan)
    return pd.to_numeric(s, errors="coerce")


def get_reference_time(df, ori_series):
    """Return the correct reference datetime series based on orientation."""
    # Orientation can be pipe-compound (e.g. "T/D|RB"); take the first token so it
    # matches the recognised set — mirrors clean_pts(), which already takes the
    # first "|"-split PTS value. Without this, milestones recorded with a compound
    # orientation (cargoDoor_close, cabinDoor_close, lastPaxBoarded, …) get no
    # reference time and their delay is silently forced to 0 despite real actuals.
    ori_series = ori_series.astype(str).str.split("|").str[0].str.strip()
    is_arrival = ori_series.isin(["A/T", "A", "T"])
    is_departure = ori_series.isin(["D", "T/D"])

    ref = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    aibt_col = "linkedFlight_arrival.inBlock.actual"
    sobt_col = "departure_offBlock.scheduled"

    if aibt_col in df.columns:
        aibt = pd.to_datetime(df[aibt_col], errors="coerce", utc=True)
        ref = ref.where(~is_arrival, aibt)
    if sobt_col in df.columns:
        sobt = pd.to_datetime(df[sobt_col], errors="coerce", utc=True)
        ref = ref.where(~is_departure, sobt)
    return ref


def calculate_planned_time(df, base_col):
    """Compute planned datetime = reference + PTS for a milestone base name."""
    pts_col = f"{base_col}.PTS"
    ori_col = f"{base_col}.orientation"
    if pts_col not in df.columns or ori_col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)

    pts_min = clean_pts(df[pts_col])
    ref = get_reference_time(df, df[ori_col])
    planned = ref + pd.to_timedelta(pts_min, unit="m")
    return planned


# ─── Load ────────────────────────────────────────────────────────────────────

def load_raw():
    print("Loading raw data files...")
    old_milestone = pd.read_excel(PATH_OLD_MILESTONE)
    old_flight    = pd.read_excel(PATH_OLD_FLIGHT)
    new_milestone = pd.read_csv(PATH_NEW_MILESTONE)
    new_flight    = pd.read_csv(PATH_NEW_FLIGHT, engine="python", on_bad_lines="skip")
    new2_milestone = pd.read_csv(PATH_NEW2_MILESTONE)
    new2_flight    = pd.read_csv(PATH_NEW2_FLIGHT, engine="python", on_bad_lines="skip")
    print(f"  Old flight: {old_flight.shape} | Old milestone: {old_milestone.shape}")
    print(f"  New flight: {new_flight.shape} | New milestone: {new_milestone.shape}")
    print(f"  New2 flight: {new2_flight.shape} | New2 milestone: {new2_milestone.shape}")
    return old_flight, old_milestone, new_flight, new_milestone, new2_flight, new2_milestone


# ─── Process ─────────────────────────────────────────────────────────────────

def process_old(old_flight, old_milestone):
    print("Processing OLD data...")
    # Milestone: only 'milestones' needs parsing; specialHandling may be absent
    old_milestone["milestones"] = old_milestone["milestones"].apply(parse_dict)
    ms_expanded = pd.json_normalize(old_milestone["milestones"])
    ms_expanded = ms_expanded.add_prefix("milestone_")
    old_milestone_clean = pd.concat(
        [old_milestone.drop(columns=["milestones"]), ms_expanded], axis=1
    )
    # Flatten specialHandling if present
    if "specialHandling" in old_milestone_clean.columns:
        old_milestone_clean = flatten_col(old_milestone_clean, "specialHandling", "specialHandling")

    old_flight_clean = flatten_flight(old_flight.copy())

    merged = pd.merge(old_flight_clean, old_milestone_clean, on="id", how="inner",
                      suffixes=("_flight", "_milestone"))
    print(f"  Old merged shape: {merged.shape}")
    return merged


def process_new(new_flight, new_milestone):
    print("Processing NEW data...")
    new_milestone_clean = flatten_milestone(new_milestone.copy())
    new_flight_clean    = flatten_flight(new_flight.copy())

    # Drop Cosmos DB metadata columns
    meta_cols = [c for c in new_milestone_clean.columns if c.startswith("_")]
    new_milestone_clean = new_milestone_clean.drop(columns=meta_cols, errors="ignore")
    meta_cols = [c for c in new_flight_clean.columns if c.startswith("_")]
    new_flight_clean = new_flight_clean.drop(columns=meta_cols, errors="ignore")

    merged = pd.merge(new_flight_clean, new_milestone_clean, on="id", how="inner",
                      suffixes=("_flight", "_milestone"))

    # Fix prefix inconsistency
    merged.columns = [
        c.replace("milestones_", "milestone_") if c.startswith("milestones_") else c
        for c in merged.columns
    ]

    print(f"  New merged shape: {merged.shape}")
    return merged


# ─── Activity Analysis ───────────────────────────────────────────────────────

def compute_activity_analysis(df_dep):
    """Compute planned/actual durations and delays for all milestone activities."""
    print("Computing activity analysis...")

    all_actuals = [c for c in df_dep.columns if c.startswith("milestone_") and c.endswith(".actual")]
    start_cols  = [c for c in all_actuals if "_start" in c]
    end_cols    = [c for c in all_actuals if "_end" in c]
    point_cols  = [c for c in all_actuals if c not in start_cols and c not in end_cols]

    activities = []
    used_ends  = set()

    for start_col in start_cols:
        parts      = start_col.split(".")
        category   = parts[0]           # e.g. milestone_aic
        start_evt  = parts[1]           # e.g. cleaning_start
        core       = start_evt.replace("_start", "").replace("first", "").replace("First", "")
        full_name  = f"{category}_{core}"

        possible_ends = [e for e in end_cols if e.startswith(category + ".") and e not in used_ends]
        best_match = None
        for e in possible_ends:
            end_evt = e.split(".")[1]
            if core.lower() in end_evt.lower():
                best_match = e
                break
        if not best_match and possible_ends:
            best_match = possible_ends[0]

        if best_match:
            used_ends.add(best_match)
            activities.append({
                "full_name":  full_name,
                "start_base": start_col.replace(".actual", ""),
                "end_base":   best_match.replace(".actual", ""),
            })

    for act in activities:
        fn   = act["full_name"]
        sb   = act["start_base"]
        eb   = act["end_base"]

        planned_start = calculate_planned_time(df_dep, sb)
        planned_end   = calculate_planned_time(df_dep, eb)

        actual_start = pd.to_datetime(df_dep.get(f"{sb}.actual"), errors="coerce", utc=True)
        actual_end   = pd.to_datetime(df_dep.get(f"{eb}.actual"), errors="coerce", utc=True)

        if planned_start.dt.tz is None:
            planned_start = planned_start.dt.tz_localize("UTC")
        if planned_end.dt.tz is None:
            planned_end = planned_end.dt.tz_localize("UTC")

        planned_dur = (planned_end - planned_start).dt.total_seconds() / 60
        actual_dur  = (actual_end  - actual_start).dt.total_seconds() / 60
        planned_dur = planned_dur.abs()

        df_dep[f"{fn}_analysis_ActualDuration_mins"]  = actual_dur
        df_dep[f"{fn}_analysis_PlannedDuration_mins"] = planned_dur
        df_dep[f"{fn}_analysis_Delay_mins"] = (actual_end - planned_end).dt.total_seconds() / 60

        # Duration overrun vs the cohort-median duration for this step. Provides a
        # delay-like signal for start/end milestones that have NO planned PTS
        # target (AIC cleaning, cabin sweep, catering/hold loading, …) so the
        # cascade flowchart can colour them. NaN where the step wasn't recorded;
        # only positive overruns count. NOT used as a model feature (the feature
        # filter in train_model matches only ActualDuration/PlannedDuration/Delay).
        _med = actual_dur.median()
        df_dep[f"{fn}_analysis_DurationOverrun_mins"] = (actual_dur - _med).clip(lower=0)

    # Point milestones
    for pc in point_cols:
        base = pc.replace(".actual", "")
        planned = calculate_planned_time(df_dep, base)
        if planned.dt.tz is None:
            planned = planned.dt.tz_localize("UTC")
        actual = pd.to_datetime(df_dep.get(pc), errors="coerce", utc=True)

        late     = actual > planned
        delay_m  = (actual - planned).dt.total_seconds() / 60
        col_base = base.replace(".", "_")
        df_dep[f"{col_base}_analysis_LateFlag"]   = np.where(pd.notna(actual) & pd.notna(planned), late, np.nan)
        df_dep[f"{col_base}_analysis_Delay_mins"] = np.where(late, delay_m, 0)

    print(f"  Activity analysis done: {len(activities)} activities, {len(point_cols)} point milestones.")
    return df_dep


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df_dep):
    print("Engineering features...")

    # Target variable
    df_dep["Target_Departure_Delay_mins"] = (
        df_dep["departure_offBlock.actual"] - df_dep["departure_offBlock.scheduled"]
    ).dt.total_seconds() / 60

    bins   = [-np.inf, 0, 4, np.inf]
    labels = ["On-Time", "Acceptable", "Delayed"]
    df_dep["Target_Departure_Delay_Class"] = pd.cut(
        df_dep["Target_Departure_Delay_mins"], bins=bins, labels=labels
    )

    # Incoming delay
    df_dep["Incoming_Delay_mins"] = (
        df_dep["linkedFlight_arrival.inBlock.actual"] -
        df_dep["linkedFlight_arrival.inBlock.scheduled"]
    ).dt.total_seconds() / 60

    # Time features
    sched = pd.to_datetime(df_dep["departure_offBlock.scheduled"], errors="coerce", utc=True)
    df_dep["Hour_of_Day"]  = sched.dt.hour
    df_dep["Month"]        = sched.dt.month
    df_dep["Day_of_Week"]  = sched.dt.day_name()
    df_dep["Is_Weekend"]   = df_dep["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)

    # Ground time
    df_dep["Available_Ground_Time_mins"] = (
        df_dep["departure_offBlock.scheduled"] -
        df_dep["linkedFlight_arrival.inBlock.actual"]
    ).dt.total_seconds() / 60

    df_dep["Is_Ground_Time_Deficient"] = (
        df_dep["Available_Ground_Time_mins"] < df_dep.get("status_minGroundTime", 0)
    ).astype(int)

    df_dep["Ground_Time_Ratio"] = (
        df_dep["Available_Ground_Time_mins"] /
        df_dep.get("status_minGroundTime", pd.Series(np.nan, index=df_dep.index)).replace(0, np.nan)
    ).replace([np.inf, -np.inf], 999).fillna(-999)

    df_dep["Delay_Pressure_Score"] = (
        df_dep["Incoming_Delay_mins"].fillna(0) * df_dep["Is_Ground_Time_Deficient"].fillna(0)
    )

    print(f"  Target distribution:\n{df_dep['Target_Departure_Delay_Class'].value_counts(dropna=False)}")
    return df_dep


# ─── Lightning Warning Feature Engineering ───────────────────────────────────

LW_COLS = [
    "LW_Count_On_Date", "Total_LW_Mins_On_Date", "LW_Day_Had_Warning",
    "LW_Active_At_Sched_Departure", "LW_Overlap_With_Ground_Window_Mins",
    "LW_Active_During_Ground_Time", "Mins_Since_Last_LW_Before_Dep",
]

def merge_lw_features_train(df_dep):
    """
    Add 7 lightning warning features to df_dep for model training.
    Mirrors utils/loader.py merge_lw_features() but with no streamlit dependency.
    """
    print("Merging lightning warning features...")

    if not os.path.exists(PATH_LW):
        print(f"  LW data not found at {PATH_LW} — setting all LW features to 0.")
        for col in LW_COLS:
            df_dep[col] = 0.0
        return df_dep

    try:
        raw = pd.read_csv(PATH_LW, encoding="latin-1")
        raw.columns = raw.columns.str.strip()

        if "Status" in raw.columns:
            raw = raw[raw["Status"].isna() | (raw["Status"].str.strip().str.lower() != "cancelled")]

        raw["DateIssued"] = pd.to_datetime(raw["DateIssued"], dayfirst=True, errors="coerce")
        raw = raw.dropna(subset=["DateIssued"])

        def _parse_time(t):
            if pd.isna(t):
                return pd.NaT
            try:
                parts = str(t).strip().split(":")
                h, m = int(parts[0]), int(parts[1])
                s = int(parts[2]) if len(parts) > 2 else 0
                return pd.Timedelta(hours=h, minutes=m, seconds=s)
            except Exception:
                return pd.NaT

        raw["start_td"] = raw["TimeIssued"].apply(_parse_time)
        raw["end_td"]   = raw["TimeCancelled"].apply(_parse_time) if "TimeCancelled" in raw.columns else pd.NaT

        # LWNo >= 10000 are midnight-continuation warnings (add 1 day to end time)
        if "LWNo" in raw.columns:
            midnight = raw["LWNo"].apply(
                lambda x: str(x).strip().isdigit() and int(str(x).strip()) >= 10000
            )
            raw.loc[midnight, "end_td"] = raw.loc[midnight, "end_td"] + pd.Timedelta(days=1)

        raw = raw.dropna(subset=["start_td"])
        raw["lw_date"]  = raw["DateIssued"].dt.date
        raw["lw_start"] = raw.apply(
            lambda r: r["DateIssued"].replace(hour=0, minute=0, second=0) + r["start_td"], axis=1
        )
        raw["lw_end"] = raw.apply(
            lambda r: (r["DateIssued"].replace(hour=0, minute=0, second=0) + r["end_td"])
                      if pd.notna(r["end_td"])
                      else (r["lw_start"] + pd.Timedelta(hours=4)),
            axis=1,
        )

        # ── Tier-1: daily aggregates ──────────────────────────────────────────
        def _total_mins(grp):
            return (raw.loc[grp.index, "lw_end"] - raw.loc[grp.index, "lw_start"]) \
                       .dt.total_seconds().clip(lower=0).sum() / 60

        lw_daily = (
            raw.groupby("lw_date")
            .agg(LW_Count_On_Date=("lw_date", "count"),
                 Total_LW_Mins_On_Date=("lw_date", _total_mins))
            .reset_index()
        )
        lw_daily["LW_Day_Had_Warning"] = 1

        dep_col = "departure_offBlock.scheduled"
        arr_col = "linkedFlight_arrival.inBlock.actual"

        dep_sgt = (
            pd.to_datetime(df_dep[dep_col], errors="coerce", utc=True) + pd.Timedelta(hours=8)
        )
        df_dep["_dep_sgt_date"] = dep_sgt.dt.date

        df_dep = df_dep.merge(
            lw_daily.rename(columns={"lw_date": "_dep_sgt_date"}),
            on="_dep_sgt_date", how="left",
        )
        df_dep["LW_Count_On_Date"]      = df_dep["LW_Count_On_Date"].fillna(0).astype(float)
        df_dep["Total_LW_Mins_On_Date"] = df_dep["Total_LW_Mins_On_Date"].fillna(0).astype(float)
        df_dep["LW_Day_Had_Warning"]    = df_dep["LW_Day_Had_Warning"].fillna(0).astype(float)

        # ── Tier-2: per-flight overlap (Vectorized) ───────────────────────────
        df_dep["flight_idx"] = np.arange(len(df_dep))
        
        f_dates = df_dep[["flight_idx", "_dep_sgt_date", dep_col, arr_col]].copy()
        f_dates = f_dates.rename(columns={"_dep_sgt_date": "lw_date"})
        
        merged_windows = pd.merge(f_dates, raw[["lw_date", "lw_start", "lw_end"]], on="lw_date", how="inner")
        
        if not merged_windows.empty:
            dep_ts = pd.to_datetime(merged_windows[dep_col], errors="coerce", utc=True) + pd.Timedelta(hours=8)
            arr_ts = pd.to_datetime(merged_windows[arr_col], errors="coerce", utc=True) + pd.Timedelta(hours=8)
            
            dep_ts_naive = dep_ts.dt.tz_localize(None)
            arr_ts_naive = arr_ts.dt.tz_localize(None)
            
            lw_s = pd.to_datetime(merged_windows["lw_start"]).dt.tz_localize(None)
            lw_e = pd.to_datetime(merged_windows["lw_end"]).dt.tz_localize(None)
            
            merged_windows["is_active_dep"] = ((lw_s <= dep_ts_naive) & (dep_ts_naive <= lw_e)).astype(float)
            
            gw_ok = (arr_ts_naive < dep_ts_naive) & arr_ts_naive.notna() & dep_ts_naive.notna()
            overlap_start = np.maximum(arr_ts_naive, lw_s)
            overlap_end   = np.minimum(dep_ts_naive, lw_e)
            overlap_secs  = (overlap_end - overlap_start).dt.total_seconds().clip(lower=0)
            
            merged_windows["overlap_mins"] = np.where(gw_ok, overlap_secs / 60.0, 0.0)
            
            ended_before = lw_e <= dep_ts_naive
            ended_diff   = np.where(ended_before, (dep_ts_naive - lw_e).dt.total_seconds() / 60.0, np.nan)
            merged_windows["ended_diff"] = ended_diff
            
            agg = merged_windows.groupby("flight_idx").agg(
                LW_Active_At_Sched_Departure=("is_active_dep", "max"),
                LW_Overlap_With_Ground_Window_Mins=("overlap_mins", "sum"),
                LW_Active_During_Ground_Time=("overlap_mins", lambda x: float(x.sum() > 0)),
                Mins_Since_Last_LW_Before_Dep=("ended_diff", "min")
            )
            
            df_dep = df_dep.join(agg, on="flight_idx", how="left")
            
        # Ensure all columns exist and fill defaults
        for col in ["LW_Active_At_Sched_Departure", "LW_Overlap_With_Ground_Window_Mins", "LW_Active_During_Ground_Time"]:
            if col not in df_dep.columns:
                df_dep[col] = 0.0
            else:
                df_dep[col] = df_dep[col].fillna(0.0)
                
        if "Mins_Since_Last_LW_Before_Dep" not in df_dep.columns:
            df_dep["Mins_Since_Last_LW_Before_Dep"] = np.nan
            
        active_mask = df_dep["LW_Active_At_Sched_Departure"] == 1
        df_dep.loc[active_mask, "Mins_Since_Last_LW_Before_Dep"] = 0.0
        
        df_dep = df_dep.drop(columns=["flight_idx", "_dep_sgt_date"], errors="ignore")

        lw_n = int(df_dep["LW_Day_Had_Warning"].sum())
        gw_n = int(df_dep["LW_Active_During_Ground_Time"].sum())
        print(f"  LW features added: {lw_n:,} flights on LW days, {gw_n:,} with ground-window overlap.")

    except Exception as exc:
        print(f"  WARNING: LW merge failed ({exc}) — setting all LW features to 0.")
        for col in LW_COLS:
            if col not in df_dep.columns:
                df_dep[col] = 0.0

    return df_dep


# ─── Model Training ──────────────────────────────────────────────────────────

def train_model(df_dep):
    print("Training XGBoost model...")

    df_model = df_dep.copy()
    df_model = df_model.dropna(subset=["Target_Departure_Delay_Class"])

    # Anti-leakage purge
    leakage_direct = ["Target_Departure_Delay_mins", "status_delayTime", "status_actualGroundTime"]
    df_model = df_model.drop(columns=[c for c in leakage_direct if c in df_model.columns])

    time_leak = [c for c in df_model.columns if any(
        t in c for t in [".actual", ".scheduled", ".estimated", ".target", ".predicted", ".PTS", ".reference"]
    )]
    df_model = df_model.drop(columns=[c for c in time_leak if c in df_model.columns])
    df_model = df_model.drop(columns=["id", "lid"], errors="ignore")
    df_model = df_model.drop(columns=[c for c in df_model.columns if "_Text" in c])

    leakage_kw = ["thumbsUp", "cabinDoor_close", "PLB_retract", "paxStep_retract",
                  "cargoDoor_close", "finalReadback", "NOTOC", "loadsheetACK"]
    terminal_cols = [c for c in df_model.columns if any(kw in c for kw in leakage_kw)]
    df_model = df_model.drop(columns=[c for c in terminal_cols if c in df_model.columns])

    # Build feature set
    analysis_num  = [c for c in df_model.columns if "_analysis_ActualDuration" in c or
                     "_analysis_PlannedDuration" in c or "_analysis_Delay_mins" in c]
    special_cols  = [c for c in df_model.columns if c.startswith("specialHandling_")]
    base_numeric  = ["Available_Ground_Time_mins", "Ground_Time_Ratio", "Incoming_Delay_mins",
                     "Is_Ground_Time_Deficient", "Is_Weekend", "Hour_of_Day", "Month",
                     "status_minGroundTime", "Delay_Pressure_Score"] + LW_COLS
    base_cat      = ["identification_carrierCode", "aircraft_bodyType", "aircraft_typeICAO",
                     "origin_terminal", "destination_iata", "Day_of_Week", "status_isRemoteBay"]

    wanted_features = (
        [c for c in base_numeric if c in df_model.columns] +
        [c for c in base_cat    if c in df_model.columns] +
        analysis_num + special_cols
    )
    # Deduplicate, keep order
    seen = set()
    feature_cols = []
    for c in wanted_features:
        if c not in seen and c in df_model.columns:
            seen.add(c)
            feature_cols.append(c)

    X = df_model[feature_cols].copy()
    le = LabelEncoder()
    y  = le.fit_transform(df_model["Target_Departure_Delay_Class"])

    numeric_feats     = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    categorical_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X[categorical_feats] = X[categorical_feats].astype(str)

    print(f"  Features: {len(feature_cols)} ({len(numeric_feats)} numeric, {len(categorical_feats)} categorical)")
    print(f"  Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    num_pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-999)),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_feats),
        ("cat", cat_pipe, categorical_feats),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(random_state=42)),
        ("classifier",   xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=9,
            learning_rate=0.15,
            subsample=0.8,
            colsample_bytree=0.8,
            missing=-999,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\n  --- Model Performance ---")
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

    model_bundle = {
        "pipeline":        pipeline,
        "label_encoder":   le,
        "feature_columns": feature_cols,
        "numeric_feats":   numeric_feats,
        "categorical_feats": categorical_feats,
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"  Model saved to {OUT_MODEL}")
    return model_bundle


# ─── Save ────────────────────────────────────────────────────────────────────

def _coerce_col_to_string(series: pd.Series) -> pd.Series:
    """Convert every value in a Series to str, preserving NaN as None."""
    def _safe(v):
        if v is None:
            return None
        try:
            if isinstance(v, float) and np.isnan(v):
                return None
        except (TypeError, ValueError):
            pass
        return str(v)
    return series.apply(_safe)


def save_parquet(df_dep):
    """
    Sanitise column types before saving to parquet:
    1. tz-aware datetime64[ns, tz] → tz-naive UTC datetime64[ns]
    2. object columns with any non-str value → convert entire column to str
       (catches mixed datetime/str, mixed float/str like timedelta strings)
    """
    import pyarrow as pa

    df_save = df_dep.copy()

    # Step 1 — tz-aware datetimes
    for col in list(df_save.columns):
        if hasattr(df_save[col].dtype, "tz") and df_save[col].dtype.tz is not None:
            df_save[col] = df_save[col].dt.tz_convert("UTC").dt.tz_localize(None)

    # Step 2 — object columns: probe with pyarrow and stringify on failure
    for col in df_save.select_dtypes(include=["object"]).columns:
        try:
            pa.array(df_save[col], from_pandas=True)
        except (pa.lib.ArrowTypeError, pa.lib.ArrowInvalid):
            df_save[col] = _coerce_col_to_string(df_save[col])

    df_save.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved {len(df_save):,} rows to {OUT_PARQUET}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import gc
    print("=" * 60)
    print("  SATS Data Preparation Pipeline")
    print("=" * 60)

    # Load and process old data
    print("Loading and processing old data...")
    old_flight    = pd.read_excel(PATH_OLD_FLIGHT)
    old_milestone = pd.read_excel(PATH_OLD_MILESTONE)
    old_merged    = process_old(old_flight, old_milestone)
    del old_flight, old_milestone
    gc.collect()

    # Load and process new data
    # Load and process new data
    print("Loading and processing new data...")
    new_flight    = pd.read_csv(PATH_NEW_FLIGHT, engine="python", on_bad_lines="skip", encoding="latin-1")
    new_milestone = pd.read_csv(PATH_NEW_MILESTONE, encoding="latin-1")
    new_merged    = process_new(new_flight, new_milestone)
    del new_flight, new_milestone
    gc.collect()

    # Load and process new2 data
    print("Loading and processing new2 data...")
    new2_flight    = pd.read_csv(PATH_NEW2_FLIGHT, engine="python", on_bad_lines="skip", encoding="latin-1")
    new2_milestone = pd.read_csv(PATH_NEW2_MILESTONE, encoding="latin-1")
    new2_merged    = process_new(new2_flight, new2_milestone)
    del new2_flight, new2_milestone
    gc.collect()

    combined = pd.concat([old_merged, new_merged, new2_merged], ignore_index=True)
    del old_merged, new_merged, new2_merged
    gc.collect()
    # new_merged (Feb-Apr) and new2_merged (Mar19-Jun30) overlap for Mar19-Apr13;
    # both extracts use the same `id` convention there, so dedupe on id and keep
    # the new2 (freshest) copy of any overlapping flight.
    before = len(combined)
    combined = combined.drop_duplicates(subset="id", keep="last").reset_index(drop=True)
    print(f"Combined shape: {combined.shape} (dropped {before - len(combined):,} overlap duplicates)")

    print("Converting datetime columns...")
    combined = convert_datetimes(combined)

    print("Filtering to Departures only...")
    df_dep = combined[combined["identification_direction"] == "Departure"].copy()
    print(f"  Departure flights: {len(df_dep):,}")

    # Ensure critical timestamp columns are properly typed
    for ts_col in [
        "linkedFlight_arrival.inBlock.actual",
        "linkedFlight_arrival.inBlock.scheduled",
        "departure_offBlock.scheduled",
        "departure_offBlock.actual",
    ]:
        if ts_col in df_dep.columns:
            df_dep[ts_col] = pd.to_datetime(df_dep[ts_col], errors="coerce", utc=True)

    df_dep = compute_activity_analysis(df_dep)
    df_dep = engineer_features(df_dep)
    df_dep = merge_lw_features_train(df_dep)

    print("Saving parquet...")
    save_parquet(df_dep)

    train_model(df_dep)

    print("=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
