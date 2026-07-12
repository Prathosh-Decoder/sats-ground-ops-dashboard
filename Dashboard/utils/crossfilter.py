"""
utils/crossfilter.py
Power BI-style cross-filtering for the SATS dashboard.

Usage pattern (every page):
    from utils.crossfilter import init_xf, apply_xf, render_xf_bar, handle_selection, get_xf

    df = load_data()
    df = render_date_filters(df, page_key="...")
    init_xf()
    df = apply_xf(df)
    render_xf_bar()
    ...
    # Source chart (click-to-filter):
    evt = st.plotly_chart(fig, on_select="rerun", key="xf_carrier_src")
    handle_selection(evt, "carrier", axis="y")   # "y" for horizontal bars
"""
import streamlit as st

# Maps dimension → dataframe column
COLUMN_MAP = {
    "carrier":     "identification_carrierCode",
    "delay_class": "Target_Departure_Delay_Class",
    "day_of_week": "Day_of_Week",
    "month":       "Month",
    "hour":        "Hour_of_Day",
    "terminal":    "origin_terminal",
    "body_type":   "aircraft_bodyType",
    "destination": "destination_iata",
}

LABELS = {
    "carrier":     "Carrier",
    "delay_class": "Delay Class",
    "day_of_week": "Day",
    "month":       "Month",
    "hour":        "Hour",
    "terminal":    "Terminal",
    "body_type":   "Body Type",
    "destination": "Destination",
}

MONTH_NAME_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
MONTH_NUM_TO_NAME = {v: k for k, v in MONTH_NAME_TO_NUM.items()}

_PREFIX = "xf_"


def init_xf():
    """Initialise session-state slots for all filter dimensions (idempotent)."""
    for dim in COLUMN_MAP:
        key = f"{_PREFIX}{dim}"
        if key not in st.session_state:
            st.session_state[key] = None


def set_xf(dim: str, val):
    """Store a filter value."""
    st.session_state[f"{_PREFIX}{dim}"] = val


def get_xf(dim: str):
    """Return the current filter value, or None."""
    return st.session_state.get(f"{_PREFIX}{dim}")


def clear_xf(dim: str = None):
    """Clear one dimension (or all if dim is None)."""
    targets = [dim] if dim else list(COLUMN_MAP.keys())
    for d in targets:
        st.session_state[f"{_PREFIX}{d}"] = None


def apply_xf(df):
    """Apply all active cross-filters to df and return the filtered copy."""
    for dim, col in COLUMN_MAP.items():
        val = get_xf(dim)
        if val is None or col not in df.columns:
            continue
        if dim == "month":
            num = MONTH_NAME_TO_NUM.get(str(val))
            target = num if num is not None else _safe_int(val)
            if target is not None:
                df = df[df[col] == target]
        else:
            df = df[df[col].astype(str) == str(val)]
    return df


def _safe_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def render_xf_bar():
    """
    Render an active-filter chip strip. Each chip clears that dimension.
    Call this once per page, just after init_xf().
    """
    active = {dim: get_xf(dim) for dim in COLUMN_MAP if get_xf(dim) is not None}
    if not active:
        return

    # Display month as name
    display = {}
    for dim, val in active.items():
        if dim == "month":
            try:
                display[dim] = MONTH_NUM_TO_NAME.get(int(val), str(val))
            except (ValueError, TypeError):
                display[dim] = str(val)
        else:
            display[dim] = str(val)

    n = len(active)
    label_col, *chip_cols, clear_col = st.columns([1.6] + [2.5] * n + [1.4])
    label_col.markdown(
        "<div style='padding-top:8px;font-size:0.78rem;color:#6b7fa3;'>"
        "🔍 <b>Active filters</b></div>",
        unsafe_allow_html=True,
    )
    for col_widget, (dim, val) in zip(chip_cols, active.items()):
        chip_label = f"✕  {LABELS[dim]}: {display[dim]}"
        if col_widget.button(chip_label, key=f"xf_chip_{dim}", use_container_width=True):
            clear_xf(dim)
            st.rerun()
    if clear_col.button("Clear all", key="xf_clear_all", use_container_width=True):
        clear_xf()
        st.rerun()

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def handle_selection(evt, dim: str, axis: str = "x"):
    """
    Process a Plotly on_select event and set the filter if a point was clicked.

    axis:
        "x"     – vertical bar / scatter; value is in points[0]["x"]
        "y"     – horizontal bar; value is in points[0]["y"]
        "label" – pie / donut; value is in points[0]["label"]
    """
    if not evt or not hasattr(evt, "selection"):
        return
    pts = getattr(evt.selection, "points", [])
    if not pts:
        return

    pt = pts[0]
    if axis == "label":
        val = pt.get("label")
    elif axis == "y":
        val = pt.get("y")
    else:
        val = pt.get("x")

    if val is None:
        return

    # Normalise month names → integer
    if dim == "month":
        num = MONTH_NAME_TO_NUM.get(str(val))
        if num is not None:
            val = num
        else:
            val = _safe_int(val) or val

    current = get_xf(dim)
    # Toggle off if same value clicked again
    if str(val) == str(current or ""):
        clear_xf(dim)
        st.rerun()
    else:
        set_xf(dim, val)
        st.rerun()


def bar_colors(values, active_val, base_colorscale=None):
    """
    Return a list of colors for a bar chart.
    - If no filter is active, returns None (caller uses its own colorscale).
    - If a filter is active, highlights the matching bar and dims the rest.
    """
    if active_val is None:
        return None
    active_str = str(active_val)
    return [
        "#1a73e8" if str(v) == active_str else "rgba(80,120,200,0.18)"
        for v in values
    ]


def pie_pull(labels, active_val):
    """
    Return a pull list for a Pie trace — explodes the active slice.
    Returns None when no filter is active.
    """
    if active_val is None:
        return None
    active_str = str(active_val)
    return [0.07 if str(lbl) == active_str else 0 for lbl in labels]
