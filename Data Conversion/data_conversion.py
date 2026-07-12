"""
SATS Ground Operations — Data Conversion Script
================================================
Converts raw DSM flight + GHAMS milestone data into a clean,
flat file ready for Power BI (or any BI tool).

HOW TO USE
----------
1. Update the paths in the CONFIG section below.
2. Run:  python data_conversion.py
3. The output file appears in the same folder as this script.

SUPPORTED INPUT FORMATS
-----------------------
  • CSV  (.csv)   — new DSM exports
  • Excel (.xlsx) — old DSM exports (Nov 2025 and earlier)

You can add as many files as you like to each list.
The script combines them all before processing.
"""

import os
import json
import ast
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# ─── CONFIG — Edit this section, then run the script ──────────
# ═══════════════════════════════════════════════════════════════

# List all your flight files here.
# Add a new line each time you get new data.
FLIGHT_FILES = [
    r"/Users/prathosh/Desktop/Sats Project/Data/Old DSM Data/dsmlive_flight_Nov2025.xlsx",
    r"/Users/prathosh/Desktop/Sats Project/Data/New DSM Data/dsmlive_flight_01022026_07042026.csv",
    # r"/Users/prathosh/Desktop/Sats Project/Data/From March Data/dsmlive_flight_Mar-May2026.csv",
]

# List all your GHAMS/milestone files here (must cover the same dates as above).
MILESTONE_FILES = [
    r"/Users/prathosh/Desktop/Sats Project/Data/Old DSM Data/dsmlive_milestone_Nov2025.xlsx",
    r"/Users/prathosh/Desktop/Sats Project/Data/New DSM Data/dsmlive_GHAMS_01022026_07042026.csv",
]

# Lightning Warning CSV — set to "" to skip LW columns.
LW_FILE = r"/Users/prathosh/Desktop/Sats Project/Data/lw_2024_2025.csv"

# Output format: "csv" | "excel" | "both"
OUTPUT_FORMAT = "csv"

# Output file name (without extension — added automatically).
OUTPUT_NAME = "SATS_clean_data"

# ═══════════════════════════════════════════════════════════════
# ─── END OF CONFIG ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _parse_dict(val):
    """Safely parse a stringified dict / JSON blob → Python dict."""
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


def _flatten_col(df, col, prefix):
    """Expand a nested JSON column and add it back with a prefix."""
    if col not in df.columns:
        return df
    parsed = df[col].apply(_parse_dict)
    first_valid = parsed.dropna().iloc[0] if not parsed.dropna().empty else None
    if not isinstance(first_valid, dict):
        return df
    expanded = pd.json_normalize(parsed).add_prefix(f"{prefix}_")
    return pd.concat([df.drop(columns=[col]), expanded], axis=1)


def _flatten_flight(df):
    for col in ["identification", "transitPoints", "status", "aircraft",
                "arrival", "departure", "origin", "destination", "linkedFlight"]:
        df = _flatten_col(df, col, col)
    if "changes" in df.columns:
        df = df.drop(columns=["changes"])
    if "updateTime" in df.columns:
        df = df.drop(columns=["updateTime"], errors="ignore")
    return df


def _flatten_milestone(df):
    """Flatten both old (milestones) and new (already flat) milestone formats."""
    # Old format: nested 'milestones' key
    if "milestones" in df.columns:
        df["milestones"] = df["milestones"].apply(_parse_dict)
        expanded = pd.json_normalize(df["milestones"]).add_prefix("milestone_")
        df = pd.concat([df.drop(columns=["milestones"]), expanded], axis=1)
    # New format: 'specialHandling' is a separate nested column
    df = _flatten_col(df, "specialHandling", "specialHandling")
    # Normalise the prefix (milestones_ → milestone_)
    df.columns = [
        c.replace("milestones_", "milestone_") if c.startswith("milestones_") else c
        for c in df.columns
    ]
    return df


def _to_datetime_utc(df):
    """Auto-convert ISO-Z string columns to UTC datetimes."""
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if isinstance(first, str) and "T" in first and first.endswith("Z"):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _clean_pts(series):
    s = series.astype(str).str.split("|").str[0]
    s = s.str.replace("+", "", regex=False)
    s = s.replace(["None", "nan", "NaN", "null", "Null", ""], np.nan)
    return pd.to_numeric(s, errors="coerce")


def _get_ref_time(df, ori):
    is_arr = ori.isin(["A/T", "A", "T"])
    is_dep = ori.isin(["D", "T/D"])
    ref = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    aibt = "linkedFlight_arrival.inBlock.actual"
    sobt = "departure_offBlock.scheduled"
    if aibt in df.columns:
        ref = ref.where(~is_arr, pd.to_datetime(df[aibt], errors="coerce", utc=True))
    if sobt in df.columns:
        ref = ref.where(~is_dep, pd.to_datetime(df[sobt], errors="coerce", utc=True))
    return ref


def _planned_time(df, base_col):
    pts_col = f"{base_col}.PTS"
    ori_col = f"{base_col}.orientation"
    if pts_col not in df.columns or ori_col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    pts = _clean_pts(df[pts_col])
    ref = _get_ref_time(df, df[ori_col])
    return ref + pd.to_timedelta(pts, unit="m")


# ──────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────

def _load_file(path):
    """Load a CSV or Excel file into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def load_and_merge(flight_files, milestone_files):
    print("\n── Loading files ──────────────────────────────────────")

    # ── Flight data ──────────────────────────────────────────
    flight_frames = []
    for path in flight_files:
        print(f"  Flight  : {os.path.basename(path)}")
        raw = _load_file(path)
        flat = _flatten_flight(raw)
        # Drop Cosmos DB internal columns
        flat = flat.drop(columns=[c for c in flat.columns if c.startswith("_")], errors="ignore")
        flight_frames.append(flat)

    flights = pd.concat(flight_frames, ignore_index=True)

    # ── Milestone data ───────────────────────────────────────
    milestone_frames = []
    for path in milestone_files:
        print(f"  Milestone: {os.path.basename(path)}")
        raw = _load_file(path)
        flat = _flatten_milestone(raw)
        flat = flat.drop(columns=[c for c in flat.columns if c.startswith("_")], errors="ignore")
        milestone_frames.append(flat)

    milestones = pd.concat(milestone_frames, ignore_index=True)

    # ── Merge on flight ID ───────────────────────────────────
    merged = pd.merge(flights, milestones, on="id", how="inner", suffixes=("_flight", "_ms"))
    # Normalise column prefix
    merged.columns = [
        c.replace("milestones_", "milestone_") if c.startswith("milestones_") else c
        for c in merged.columns
    ]
    print(f"  Merged rows: {len(merged):,}")
    return merged


# ──────────────────────────────────────────────────────────────
# ACTIVITY ANALYSIS — planned vs actual milestone delays
# ──────────────────────────────────────────────────────────────

def compute_activity_analysis(df):
    print("\n── Computing milestone delays ─────────────────────────")

    all_actuals = [c for c in df.columns if c.startswith("milestone_") and c.endswith(".actual")]
    start_cols  = [c for c in all_actuals if "_start" in c]
    end_cols    = [c for c in all_actuals if "_end"   in c]
    point_cols  = [c for c in all_actuals if c not in start_cols and c not in end_cols]

    activities = []
    used_ends  = set()

    for sc in start_cols:
        parts    = sc.split(".")
        category = parts[0]
        core     = parts[1].replace("_start", "").replace("first", "").replace("First", "")
        fn       = f"{category}_{core}"
        ends     = [e for e in end_cols if e.startswith(category + ".") and e not in used_ends]
        best     = next((e for e in ends if core.lower() in e.split(".")[1].lower()), None)
        if best is None and ends:
            best = ends[0]
        if best:
            used_ends.add(best)
            activities.append({
                "fn":   fn,
                "s_base": sc.replace(".actual", ""),
                "e_base": best.replace(".actual", ""),
            })

    for act in activities:
        fn, sb, eb = act["fn"], act["s_base"], act["e_base"]
        ps = _planned_time(df, sb)
        pe = _planned_time(df, eb)
        as_ = pd.to_datetime(df.get(f"{sb}.actual"), errors="coerce", utc=True)
        ae  = pd.to_datetime(df.get(f"{eb}.actual"), errors="coerce", utc=True)
        if ps.dt.tz is None:
            ps = ps.dt.tz_localize("UTC")
        if pe.dt.tz is None:
            pe = pe.dt.tz_localize("UTC")
        df[f"{fn}_analysis_ActualDuration_mins"]  = (ae - as_).dt.total_seconds() / 60
        df[f"{fn}_analysis_PlannedDuration_mins"] = (pe - ps).dt.total_seconds().abs() / 60
        df[f"{fn}_analysis_Delay_mins"]           = (ae - pe).dt.total_seconds() / 60

    for pc in point_cols:
        base    = pc.replace(".actual", "")
        planned = _planned_time(df, base)
        if planned.dt.tz is None:
            planned = planned.dt.tz_localize("UTC")
        actual  = pd.to_datetime(df.get(pc), errors="coerce", utc=True)
        late    = actual > planned
        delay_m = (actual - planned).dt.total_seconds() / 60
        col_base = base.replace(".", "_")
        df[f"{col_base}_analysis_Delay_mins"] = np.where(late, delay_m, 0)

    n_act = len(activities)
    n_pt  = len(point_cols)
    print(f"  {n_act} duration activities + {n_pt} point milestones processed.")
    return df


# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def engineer_features(df):
    print("\n── Engineering features ───────────────────────────────")

    df["Departure_Delay_mins"] = (
        df["departure_offBlock.actual"] - df["departure_offBlock.scheduled"]
    ).dt.total_seconds() / 60

    bins   = [-np.inf, 0, 4, np.inf]
    labels = ["On-Time", "Acceptable", "Delayed"]
    df["Delay_Class"] = pd.cut(df["Departure_Delay_mins"], bins=bins, labels=labels).astype(str)
    df["Is_Delayed"]  = (df["Delay_Class"] == "Delayed").astype(int)

    df["Incoming_Delay_mins"] = (
        df["linkedFlight_arrival.inBlock.actual"] -
        df["linkedFlight_arrival.inBlock.scheduled"]
    ).dt.total_seconds() / 60

    sched = pd.to_datetime(df["departure_offBlock.scheduled"], errors="coerce", utc=True)
    sgt   = sched + pd.Timedelta(hours=8)

    df["Departure_Date"]         = sgt.dt.date.astype(str)
    df["Departure_Year"]         = sgt.dt.year
    df["Departure_Month_Num"]    = sgt.dt.month
    df["Departure_Month_Name"]   = sgt.dt.strftime("%b")
    df["Departure_Week"]         = sgt.dt.isocalendar().week.astype(int)
    df["Departure_Day_of_Week"]  = sgt.dt.day_name()
    df["Departure_Hour_SGT"]     = sgt.dt.hour
    df["Is_Weekend"]             = sgt.dt.dayofweek.isin([5, 6]).astype(int)

    df["Available_Ground_Time_mins"] = (
        df["departure_offBlock.scheduled"] -
        df["linkedFlight_arrival.inBlock.actual"]
    ).dt.total_seconds() / 60

    min_gt = pd.to_numeric(df.get("status_minGroundTime", pd.Series(np.nan, index=df.index)),
                           errors="coerce")
    df["Min_Ground_Time_mins"]   = min_gt
    df["Is_Ground_Time_Deficient"] = (df["Available_Ground_Time_mins"] < min_gt).astype(int)
    df["Ground_Time_Ratio"] = (
        df["Available_Ground_Time_mins"] / min_gt.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    print(f"  Delay class distribution:\n"
          f"    {df['Delay_Class'].value_counts(dropna=False).to_dict()}")
    return df


# ──────────────────────────────────────────────────────────────
# LIGHTNING WARNING
# ──────────────────────────────────────────────────────────────

LW_COLS = [
    "LW_Day_Had_Warning", "LW_Count_On_Date", "LW_Total_Duration_mins",
    "LW_Active_At_Departure", "LW_In_Ground_Window", "LW_Overlap_Ground_Mins",
]

def merge_lw(df, lw_path):
    if not lw_path or not os.path.exists(lw_path):
        print("\n── Lightning Warning: file not found — columns set to 0.")
        for c in LW_COLS:
            df[c] = 0.0
        return df

    print("\n── Merging Lightning Warning data ─────────────────────")
    try:
        raw = pd.read_csv(lw_path, encoding="latin-1")
        raw.columns = raw.columns.str.strip()

        if "Status" in raw.columns:
            raw = raw[raw["Status"].isna() | (raw["Status"].str.strip().str.lower() != "cancelled")]

        raw["DateIssued"] = pd.to_datetime(raw["DateIssued"], dayfirst=True, errors="coerce")
        raw = raw.dropna(subset=["DateIssued"])

        def _parse_td(t):
            if pd.isna(t):
                return pd.NaT
            try:
                parts = str(t).strip().split(":")
                return pd.Timedelta(hours=int(parts[0]), minutes=int(parts[1]),
                                    seconds=int(parts[2]) if len(parts) > 2 else 0)
            except Exception:
                return pd.NaT

        raw["start_td"] = raw["TimeIssued"].apply(_parse_td)
        raw["end_td"]   = raw["TimeCancelled"].apply(_parse_td) if "TimeCancelled" in raw.columns else pd.NaT

        if "LWNo" in raw.columns:
            midnight = raw["LWNo"].apply(
                lambda x: str(x).strip().isdigit() and int(str(x).strip()) >= 10000
            )
            raw.loc[midnight, "end_td"] += pd.Timedelta(days=1)

        raw = raw.dropna(subset=["start_td"])
        raw["lw_date"]  = raw["DateIssued"].dt.date
        raw["lw_start"] = raw.apply(
            lambda r: r["DateIssued"].replace(hour=0, minute=0, second=0) + r["start_td"], axis=1
        )
        raw["lw_end"] = raw.apply(
            lambda r: (r["DateIssued"].replace(hour=0, minute=0, second=0) + r["end_td"])
                      if pd.notna(r["end_td"])
                      else r["lw_start"] + pd.Timedelta(hours=4),
            axis=1,
        )

        def _total_mins(grp):
            return (raw.loc[grp.index, "lw_end"] - raw.loc[grp.index, "lw_start"]) \
                       .dt.total_seconds().clip(lower=0).sum() / 60

        lw_daily = (
            raw.groupby("lw_date")
            .agg(LW_Count_On_Date=("lw_date", "count"),
                 LW_Total_Duration_mins=("lw_date", _total_mins))
            .reset_index()
        )
        lw_daily["LW_Day_Had_Warning"] = 1

        dep_sgt = (
            pd.to_datetime(df["departure_offBlock.scheduled"], errors="coerce", utc=True)
            + pd.Timedelta(hours=8)
        )
        df["_dep_date_key"] = dep_sgt.dt.date

        df = df.merge(
            lw_daily.rename(columns={"lw_date": "_dep_date_key"}),
            on="_dep_date_key", how="left",
        )
        df["LW_Count_On_Date"]      = df["LW_Count_On_Date"].fillna(0)
        df["LW_Total_Duration_mins"]= df["LW_Total_Duration_mins"].fillna(0)
        df["LW_Day_Had_Warning"]    = df["LW_Day_Had_Warning"].fillna(0)

        # Per-flight overlap
        lw_by_date = {}
        for _, row in raw.iterrows():
            lw_by_date.setdefault(row["lw_date"], []).append((row["lw_start"], row["lw_end"]))

        dep_naive = dep_sgt.dt.tz_localize(None)
        arr_col   = "linkedFlight_arrival.inBlock.actual"
        arr_sgt   = None
        if arr_col in df.columns:
            arr_sgt = (
                pd.to_datetime(df[arr_col], errors="coerce", utc=True) + pd.Timedelta(hours=8)
            ).dt.tz_localize(None)

        active_at_dep = np.zeros(len(df), dtype=float)
        in_gw         = np.zeros(len(df), dtype=float)
        overlap_mins  = np.zeros(len(df), dtype=float)

        for i, (d_date, dep_ts) in enumerate(zip(df["_dep_date_key"], dep_naive)):
            lws = lw_by_date.get(d_date, [])
            if not lws:
                continue
            arr_ts = arr_sgt.iloc[i] if arr_sgt is not None else None
            gw_ok  = arr_ts is not None and pd.notna(arr_ts) and pd.notna(dep_ts)
            for lw_s, lw_e in lws:
                if pd.notna(dep_ts) and lw_s <= dep_ts <= lw_e:
                    active_at_dep[i] = 1
                if gw_ok:
                    ov_s = max(lw_s, arr_ts)
                    ov_e = min(lw_e, dep_ts)
                    if ov_e > ov_s:
                        in_gw[i] = 1
                        overlap_mins[i] += (ov_e - ov_s).total_seconds() / 60

        df["LW_Active_At_Departure"] = active_at_dep
        df["LW_In_Ground_Window"]    = in_gw
        df["LW_Overlap_Ground_Mins"] = overlap_mins
        df = df.drop(columns=["_dep_date_key"], errors="ignore")

        print(f"  LW days: {int(df['LW_Day_Had_Warning'].sum()):,} flights  |  "
              f"Ground-window overlap: {int(df['LW_In_Ground_Window'].sum()):,} flights")

    except Exception as exc:
        print(f"  WARNING: LW merge failed ({exc}) — columns set to 0.")
        for c in LW_COLS:
            if c not in df.columns:
                df[c] = 0.0

    return df


# ──────────────────────────────────────────────────────────────
# COLUMN SELECTION & RENAMING — Power BI friendly output
# ──────────────────────────────────────────────────────────────

# Human-readable names for key columns
RENAME_MAP = {
    # Identity
    "id":                              "Flight_ID",
    "identification_iata":             "Flight_Number",
    "identification_carrierCode":      "Carrier_Code",
    # Aircraft
    "aircraft_bodyType":               "Aircraft_Body_Type",
    "aircraft_typeICAO":               "Aircraft_ICAO_Type",
    # Routing
    "origin_terminal":                 "Terminal",
    "origin_gate":                     "Gate",
    "origin_standPosition":            "Stand_Position",
    "destination_iata":                "Destination",
    "status_isRemoteBay":              "Is_Remote_Bay",
    # Timestamps — stored in SGT (Singapore Time)
    "departure_offBlock.scheduled":    "Scheduled_Departure_SGT",
    "departure_offBlock.actual":       "Actual_Departure_SGT",
    "linkedFlight_arrival.inBlock.scheduled": "Scheduled_Inbound_Arrival_SGT",
    "linkedFlight_arrival.inBlock.actual":    "Actual_Inbound_Arrival_SGT",
}

# Milestone delay columns — strip prefix/suffix for readability
# "milestone_ramp_manAtBay_analysis_Delay_mins" → "MS_Delay_ramp_manAtBay_mins"
def _rename_milestone(col):
    col = col.replace("milestone_", "MS_")
    col = col.replace("_analysis_Delay_mins",       "_Delay_mins")
    col = col.replace("_analysis_ActualDuration_mins",  "_Duration_Actual_mins")
    col = col.replace("_analysis_PlannedDuration_mins", "_Duration_Planned_mins")
    col = col.replace(".", "_")
    return col


def _rename_special_handling(col):
    return col.replace("specialHandling_", "SH_")


def select_and_rename(df):
    print("\n── Selecting & renaming columns ───────────────────────")

    # Convert UTC datetimes → SGT naive strings (Power BI compatible)
    ts_cols = [
        "departure_offBlock.scheduled",
        "departure_offBlock.actual",
        "linkedFlight_arrival.inBlock.scheduled",
        "linkedFlight_arrival.inBlock.actual",
    ]
    for col in ts_cols:
        if col in df.columns:
            sgt = pd.to_datetime(df[col], errors="coerce", utc=True) + pd.Timedelta(hours=8)
            df[col] = sgt.dt.tz_localize(None)  # naive datetime — Power BI handles this natively

    # Core metadata columns (keep if present)
    core = [
        "id",
        "identification_iata",
        "identification_carrierCode",
        "aircraft_bodyType",
        "aircraft_typeICAO",
        "origin_terminal",
        "origin_gate",
        "origin_standPosition",
        "destination_iata",
        "status_isRemoteBay",
        "Min_Ground_Time_mins",
    ]

    # Timestamps
    timestamps = [
        "departure_offBlock.scheduled",
        "departure_offBlock.actual",
        "linkedFlight_arrival.inBlock.scheduled",
        "linkedFlight_arrival.inBlock.actual",
    ]

    # Engineered date features
    date_feats = [
        "Departure_Date",
        "Departure_Year",
        "Departure_Month_Num",
        "Departure_Month_Name",
        "Departure_Week",
        "Departure_Day_of_Week",
        "Departure_Hour_SGT",
        "Is_Weekend",
    ]

    # Delay metrics
    delay_metrics = [
        "Departure_Delay_mins",
        "Delay_Class",
        "Is_Delayed",
        "Incoming_Delay_mins",
        "Available_Ground_Time_mins",
        "Ground_Time_Ratio",
        "Is_Ground_Time_Deficient",
    ]

    # Lightning warning
    lw_cols = LW_COLS

    # Special handling
    sh_cols = [c for c in df.columns if c.startswith("specialHandling_")]

    # Milestone delays
    ms_delay_cols = sorted([
        c for c in df.columns if "_analysis_Delay_mins" in c and "Duration" not in c
    ])
    ms_dur_cols = sorted([
        c for c in df.columns
        if "_analysis_ActualDuration_mins" in c or "_analysis_PlannedDuration_mins" in c
    ])

    # Build ordered column list (keep only those that exist)
    ordered = (
        [c for c in core         if c in df.columns] +
        [c for c in timestamps   if c in df.columns] +
        [c for c in date_feats   if c in df.columns] +
        [c for c in delay_metrics if c in df.columns] +
        [c for c in lw_cols      if c in df.columns] +
        sh_cols +
        ms_delay_cols +
        ms_dur_cols
    )
    # Deduplicate while preserving order
    seen = set()
    ordered = [c for c in ordered if not (c in seen or seen.add(c))]

    df = df[ordered].copy()

    # Rename
    rename = {**RENAME_MAP}
    for col in df.columns:
        if col.startswith("milestone_"):
            rename[col] = _rename_milestone(col)
        elif col.startswith("specialHandling_"):
            rename[col] = _rename_special_handling(col)
        elif col not in rename:
            rename[col] = col  # keep as-is

    df = df.rename(columns=rename)

    # Replace any remaining dots in column names with underscores
    df.columns = [c.replace(".", "_") for c in df.columns]

    print(f"  Output columns: {len(df.columns)}")
    return df


# ──────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────

def save_output(df, fmt, name, out_dir):
    print("\n── Saving output ──────────────────────────────────────")
    saved = []

    if fmt in ("csv", "both"):
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  CSV   → {path}  ({size_mb:.1f} MB)")
        saved.append(path)

    if fmt in ("excel", "both"):
        path = os.path.join(out_dir, f"{name}.xlsx")
        if len(df) > 1_000_000:
            print(f"  Excel → SKIPPED (Excel max 1M rows; dataset has {len(df):,} rows).")
            print(f"          Use CSV output instead.")
        else:
            df.to_excel(path, index=False, engine="openpyxl")
            size_mb = os.path.getsize(path) / 1e6
            print(f"  Excel → {path}  ({size_mb:.1f} MB)")
            saved.append(path)

    return saved


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SATS Ground Operations — Data Conversion")
    print("=" * 60)

    # 1 — Load & merge
    combined = load_and_merge(FLIGHT_FILES, MILESTONE_FILES)

    # 2 — Parse datetime strings
    print("\n── Parsing datetime columns ───────────────────────────")
    combined = _to_datetime_utc(combined)

    # 3 — Filter to departure flights only
    print("\n── Filtering to Departure flights ─────────────────────")
    if "identification_direction" in combined.columns:
        df = combined[combined["identification_direction"] == "Departure"].copy()
    else:
        df = combined.copy()
    print(f"  Departure flights: {len(df):,}")

    # 4 — Ensure critical timestamp columns are UTC-aware
    for col in [
        "departure_offBlock.scheduled",
        "departure_offBlock.actual",
        "linkedFlight_arrival.inBlock.actual",
        "linkedFlight_arrival.inBlock.scheduled",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 5 — Milestone activity analysis
    df = compute_activity_analysis(df)

    # 6 — Feature engineering
    df = engineer_features(df)

    # 7 — Lightning Warning features
    df = merge_lw(df, LW_FILE)

    # 8 — Select & rename columns for Power BI
    df = select_and_rename(df)

    # 9 — Save
    save_output(df, OUTPUT_FORMAT, OUTPUT_NAME, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"  Done!  {len(df):,} flights  ·  {len(df.columns)} columns")
    print("=" * 60)

    # Quick summary of what's in the output
    print("\nQUICK SUMMARY")
    print("-" * 40)
    if "Departure_Date" in df.columns:
        print(f"  Date range  : {df['Departure_Date'].min()} → {df['Departure_Date'].max()}")
    if "Carrier_Code" in df.columns:
        print(f"  Carriers    : {df['Carrier_Code'].nunique()} unique")
    if "Destination" in df.columns:
        print(f"  Destinations: {df['Destination'].nunique()} unique")
    if "Delay_Class" in df.columns:
        dist = df["Delay_Class"].value_counts(dropna=False)
        for cls, cnt in dist.items():
            pct = cnt / len(df) * 100
            print(f"  {str(cls):12s}: {cnt:,} ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
