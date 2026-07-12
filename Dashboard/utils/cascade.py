"""
utils/cascade.py
Shared ground-ops cascade constants and flowchart builders.

Process model (Changi SQ turnaround, departures), 8 phases:
  0  BEFORE ARRIVAL — teams position at bay (MAB) + Load Control paperwork
  1  AIRCRAFT ARRIVES (AIBT / on-chocks reference)
  2  Tech Ramp brings aircraft to rest (ADGS → on-chocks → thumbs-up → PLB)
  3  Disembark + entry security (Security Door-2 is the CRUCIAL gate)
  4  Cargo / baggage (parallel with disembark)
  5  Loading (needs RLO at bay + ELIR)
  6  Boarding (after cabin sweep)
  7  Close-up & pushback prep
  8  PUSHBACK / DEPARTURE (off-block actual, outcome)

Nodes are either:
  ★ MEASURED   — a milestone delay column exists (coloured by delay)
  ◇ STRUCTURAL — described process step with no data (neutral box, no metric)
  + reference / outcome nodes (Aircraft Arrives, Departure)

Imported by:
  07_Cascade_Effect.py  — simulation & regression (measured nodes only)
  06_BU_Impact.py       — BU-level highlight flowchart
  09_Flight_Deep_Dive.py — per-flight actual cascade
  08_Flight_Investigation.py / 01_Flight_Monitor.py — per-flight cascade inspector
  05_Activity_Analysis.py — predecessor correlations (uses DEPS)
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.style import chart_template, chart_fc, chart_gc, is_light

# ── Team registry ─────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "ref":      "#1a5276",
    "techramp": "#34495e",   # NEW — Tech Ramp (marshaling, on-chocks, pushback)
    "ramp":     "#2980b9",
    "pax":      "#8e44ad",
    "aic":      "#27ae60",
    "cabin":    "#d4ac0d",
    "cargo":    "#cb4335",
    "baggage":  "#e67e22",   # NEW — Baggage
    "security": "#17a589",
    "loadctrl": "#ca6f1e",
    "dep":      "#27ae60",
}
TEAM_LABELS = {
    "ref":      "Reference",
    "techramp": "Tech Ramp",
    "ramp":     "Ramp",
    "pax":      "Passenger Svc",
    "aic":      "AIC Cleaning",
    "cabin":    "Cabin Service",
    "cargo":    "Cargo",
    "baggage":  "Baggage",
    "security": "Security",
    "loadctrl": "Load Control",
    "dep":      "Departure",
}

# Node display names that should never show a delay metric / cannot be simulated.
DEPARTURE_NODE = "✈️  DEPARTURE"
ARRIVES_NODE   = "Aircraft Arrives"
CRUCIAL_NODES  = {"Security: Door 2 MAB"}   # highlighted as crucial gate

# ── Phase / row layout ────────────────────────────────────────────────────────
# Each row is rendered at one y-level (top → bottom); x is spread across the row.
# A node = (display_name, team, delay_col_or_None, planned_offset_from_SOBT_or_None)
# delay_col None  → structural (no data) unless it is a ref/dep node.
# planned_offset  → minutes relative to SOBT, used only by the timeline view.
_ROWS = [
    # ── PHASE 0 · BEFORE ARRIVAL — position at bay (MAB) ──────────────────────
    [
        ("GSE: At Bay",      "techramp", None,                                      None),
        ("Ramp: MAB",        "ramp",     "milestone_ramp_manAtBay_analysis_Delay_mins",    -65),
        ("Ramp: RLO at Bay", "ramp",     "milestone_ramp_manAtBayRSM_analysis_Delay_mins", -64),
        ("AIC: MAB",         "aic",      "milestone_aic_manAtBay_analysis_Delay_mins",     -66),
        ("Cabin Svc: MAB",   "cabin",    "milestone_cabinSvc_manAtBay_analysis_Delay_mins",-66),
    ],
    [
        ("Security: Shipside MAB","security", "milestone_security_shipside_analysis_Delay_mins", -93),
        ("Baggage: MAB",         "baggage",  None,                                  None),
        ("PAX: Gate MAB",        "pax",      "milestone_pax_manAtBay_analysis_Delay_mins", -68),
    ],
    # ── PHASE 0 · BEFORE ARRIVAL — Load Control paperwork ─────────────────────
    [
        ("LC: Baggage Breakup",   "loadctrl", None,                                 None),
        ("LC: Load Plan → Bag",   "loadctrl", None,                                 None),
        ("Cargo: DLS → LC",       "cargo",    "milestone_cargo_DLS_analysis_Delay_mins",          -85),
        ("LC: ELIR → Ramp",       "loadctrl", "milestone_loadControl_ELIRnNOTOC_analysis_Delay_mins", -75),
        # NOTOC: cargo.NOTOC is never recorded (0% actual); ramp.NOTOC carries the
        # real completion time (~87% filled, ~21% late) — use that as the signal.
        ("NOTOC",                 "cargo",    "milestone_ramp_NOTOC_analysis_Delay_mins",          -80),
    ],
    # ── PHASE 1 · AIRCRAFT ARRIVES ────────────────────────────────────────────
    [
        (ARRIVES_NODE, "ref", None, None),
    ],
    # ── PHASE 2 · Tech Ramp brings aircraft to rest ───────────────────────────
    # NOTE: on-chocks == AIBT == "Aircraft Arrives" (a single event); it is the
    # reference node above, NOT a separate node here.
    [
        ("ADGS / Marshaling", "techramp", None,                                     None),
        ("Thumbs Up",         "techramp", "milestone_ramp_thumbsUp_analysis_Delay_mins", -58),
        ("Put Cones",         "ramp",     None,                                     None),
        ("PLB Activation",    "ramp",     "milestone_ramp_PLB_dock_analysis_Delay_mins", -55),
    ],
    # ── PHASE 3 + 4 · Disembark / entry security  ‖  cargo & baggage ──────────
    [
        ("Last Pax Exits",        "pax",      None,                                 None),
        ("Security: Door 2 MAB",  "security", "milestone_security_door2_analysis_Delay_mins", -45),
        ("Baggage: To Airside",   "baggage",  None,                                 None),
        ("TPO: Cargo to Airside", "cargo",    "milestone_TPO_CargoTowing_analysis_Delay_mins", -52),
        ("Security: Door-5 Cargo","security", None,                                 None),
    ],
    [
        ("AIC: Cleaning Start",   "aic",      None,                                 None),
        ("AIC: Cleaning End",     "aic",      None,                                 None),
        ("Ramp: Attach Pax-Step", "ramp",     "milestone_ramp_paxStep_dock_analysis_Delay_mins", -50),
        ("Security: Cabin Sweep", "security", "milestone_security_cabinSweep_analysis_DurationOverrun_mins", -30),
        ("Cabin Svc: Catering Unload","cabin", None,                               None),
    ],
    # ── PHASE 5 · Loading ─────────────────────────────────────────────────────
    [
        ("Ramp: RSM Loads",         "ramp",     "milestone_ramp_loading_analysis_DurationOverrun_mins", -40),
        ("Cabin Svc: Catering Load","cabin",    "milestone_cabinSvc_loading_analysis_DurationOverrun_mins", -45),
        ("Final Fuel → LC (?)",     "loadctrl", None,                              None),
        ("Above-Wing: Counter Close","loadctrl", None,                            None),
        ("LC: Loadsheet ACK",       "loadctrl", "milestone_loadControl_loadsheetACK_analysis_Delay_mins", -10),
    ],
    # ── PHASE 6 · Boarding ────────────────────────────────────────────────────
    [
        ("PAX: Gate Opens",   "pax", "milestone_pax_openGateTeam_analysis_Delay_mins",     -35),
        ("PAX: Boarding −15", "pax", "milestone_pax_boardingLoad_(15)_analysis_Delay_mins", -15),
        ("PAX: Boarding −10", "pax", "milestone_pax_boardingLoad_(10)_analysis_Delay_mins", -10),
        ("PAX: Last Pax Boarded","pax","milestone_pax_lastPaxBoarded_analysis_Delay_mins",  -7),
    ],
    # ── PHASE 7 · Close-up & pushback prep ────────────────────────────────────
    [
        ("PAX: Cabin Doors Close", "pax",      "milestone_pax_cabinDoor_close_analysis_Delay_mins", -5),
        ("Ramp: Cargo Doors Close","ramp",     "milestone_ramp_cargoDoor_close_analysis_Delay_mins", -5),
        ("Ramp: PLB Retract",      "ramp",     "milestone_ramp_PLB_retract_analysis_Delay_mins",     -3),
        ("Final Readback",         "ramp",     "milestone_ramp_finalReadback_analysis_Delay_mins",   -2),
        ("Connect Pushback",       "techramp", "milestone_techRamp_pushBackTug_arrive_analysis_Delay_mins", -25),
    ],
    [
        ("Headset Pre-Dep Checks", "techramp", None, None),
    ],
    # ── PHASE 8 · PUSHBACK / DEPARTURE ────────────────────────────────────────
    [
        (DEPARTURE_NODE, "dep", None, None),
    ],
]

# NODES / NODE_MAP / STRUCTURAL / MEASURED are built AFTER DEPS, because the
# crossing-reduction layout needs the dependency graph (see below).

# ── Dependency graph (child → parents) ────────────────────────────────────────
DEPS = {
    # Phase 0 — MAB roots
    "GSE: At Bay":             [],
    "Ramp: MAB":               [],
    "Ramp: RLO at Bay":        [],
    "AIC: MAB":                [],
    "Cabin Svc: MAB":          [],
    "Security: Shipside MAB":  [],
    "Baggage: MAB":            [],
    "PAX: Gate MAB":           [],
    # Phase 0 — Load Control paperwork
    "LC: Baggage Breakup":     [],
    "LC: Load Plan → Bag":     ["LC: Baggage Breakup"],
    "Cargo: DLS → LC":         [],
    "LC: ELIR → Ramp":         ["Cargo: DLS → LC", "LC: Load Plan → Bag"],
    "NOTOC":                   ["Cargo: DLS → LC"],
    # Phase 1
    ARRIVES_NODE:              [],
    # Phase 2 — Tech Ramp
    "ADGS / Marshaling":       [ARRIVES_NODE],
    "Thumbs Up":               ["ADGS / Marshaling"],
    "Put Cones":               ["Thumbs Up"],
    "PLB Activation":          ["Put Cones", "Ramp: MAB"],
    # Phase 3 — disembark + entry security
    "Last Pax Exits":          ["PLB Activation"],
    "Security: Door 2 MAB":    ["Last Pax Exits"],
    "AIC: Cleaning Start":     ["Security: Door 2 MAB"],
    "AIC: Cleaning End":       ["AIC: Cleaning Start"],
    "Security: Cabin Sweep":   ["AIC: Cleaning End"],
    # Cabin Service (catering): enter via Door 2 → offload old galley → uplift new
    "Cabin Svc: Catering Unload": ["Security: Door 2 MAB", "Cabin Svc: MAB"],
    "Cabin Svc: Catering Load":   ["Cabin Svc: Catering Unload"],
    # Phase 4 — cargo / baggage
    "Baggage: To Airside":     [ARRIVES_NODE, "Baggage: MAB"],
    "TPO: Cargo to Airside":   [ARRIVES_NODE],
    "Ramp: Attach Pax-Step":   ["PLB Activation"],
    "Security: Door-5 Cargo":  ["Security: Shipside MAB", "Baggage: To Airside"],
    # Phase 5 — loading
    "Ramp: RSM Loads":         ["Ramp: Attach Pax-Step", "Ramp: RLO at Bay",
                                "LC: ELIR → Ramp", "Security: Door-5 Cargo"],
    "Final Fuel → LC (?)":     [ARRIVES_NODE],
    "Above-Wing: Counter Close": ["Final Fuel → LC (?)"],
    "LC: Loadsheet ACK":       ["NOTOC", "Above-Wing: Counter Close", "Ramp: RSM Loads"],
    # Phase 6 — boarding
    "PAX: Gate Opens":         ["Security: Cabin Sweep", "Cabin Svc: Catering Load"],
    "PAX: Boarding −15":       ["PAX: Gate Opens"],
    "PAX: Boarding −10":       ["PAX: Boarding −15"],
    "PAX: Last Pax Boarded":   ["PAX: Boarding −10"],
    # Phase 7 — close-up
    "PAX: Cabin Doors Close":  ["PAX: Last Pax Boarded"],
    "Ramp: Cargo Doors Close": ["Ramp: RSM Loads", "LC: Loadsheet ACK"],
    "Ramp: PLB Retract":       ["PAX: Cabin Doors Close"],
    "Final Readback":          ["Ramp: Cargo Doors Close", "LC: Loadsheet ACK"],
    "Connect Pushback":        ["Ramp: PLB Retract", "Final Readback"],
    "Headset Pre-Dep Checks":  ["Connect Pushback"],
    # Phase 8
    DEPARTURE_NODE:            ["Headset Pre-Dep Checks"],
}


# ── Layout: BU swimlane columns (x = Business Unit, rows = its activities) ─────
# Each Business Unit is a vertical column; its activities stack top→bottom in
# process order. Aircraft Arrives sits above the columns, Pushback below them.
_BU_ORDER = ["techramp", "ramp", "cargo", "baggage", "loadctrl", "security", "aic", "cabin", "pax"]

_X_LEFT, _X_RIGHT = 0.85, 11.15
_Y_TOP, _ROW_GAP  = 12.4, 1.5
_CENTER_X = (_X_LEFT + _X_RIGHT) / 2
_COL_X = {t: _X_LEFT + i * (_X_RIGHT - _X_LEFT) / (len(_BU_ORDER) - 1)
          for i, t in enumerate(_BU_ORDER)}

# Process rank (phase row, intra-row index) used to order nodes within a column.
_RANK, _TEAM_OF = {}, {}
for _r, _row in enumerate(_ROWS):
    for _i, _spec in enumerate(_row):
        _RANK[_spec[0]] = (_r, _i)
        _TEAM_OF[_spec[0]] = _spec[1]

_COLS = {t: [] for t in _BU_ORDER}
for _name, _team in _TEAM_OF.items():
    if _team in _COLS:
        _COLS[_team].append(_name)
for _t in _COLS:
    _COLS[_t].sort(key=lambda n: _RANK[n])

_MAX_LEN = max((len(v) for v in _COLS.values()), default=1)

_NODE_X, _NODE_Y = {}, {}
for _t, _names in _COLS.items():
    for _j, _name in enumerate(_names):
        _NODE_X[_name] = _COL_X[_t]
        _NODE_Y[_name] = _Y_TOP - _j * _ROW_GAP
# Reference / outcome nodes — centred, above and below the columns.
_NODE_X[ARRIVES_NODE]   = _CENTER_X
_NODE_Y[ARRIVES_NODE]   = _Y_TOP + 1.6
_NODE_X[DEPARTURE_NODE] = _CENTER_X
_NODE_Y[DEPARTURE_NODE] = _Y_TOP - _MAX_LEN * _ROW_GAP - 0.1

# Column header labels (BU name above each column) and faint swimlane bands.
COLUMN_HEADERS = [(TEAM_LABELS.get(_t, _t), _COL_X[_t], _Y_TOP + 1.0, TEAM_COLORS.get(_t, "#888"))
                  for _t in _BU_ORDER]
# Bands stop 0.85 below the last activity row — mirroring the 0.85 padding above
# the first row — so PUSHBACK sits OUTSIDE/below the bands, like Aircraft Arrives
# sits outside/above them.
COLUMN_BANDS_Y = (_Y_TOP - (_MAX_LEN - 1) * _ROW_GAP - 0.85, _Y_TOP + 0.85)  # (bottom, top)

NODES = []
for _r, _row in enumerate(_ROWS):
    for (_name, _team, _col, _off) in _row:
        NODES.append((_name, _team, _col,
                      round(_NODE_X[_name], 3), round(_NODE_Y[_name], 3), _off))

NODE_MAP = {n[0]: n for n in NODES}

# Straight edges: {(parent, child): [(px, py), (cx, cy)]}
ROUTES = {
    (p, c): [(NODE_MAP[p][3], NODE_MAP[p][4]), (NODE_MAP[c][3], NODE_MAP[c][4])]
    for c, ps in DEPS.items() for p in ps if p in NODE_MAP and c in NODE_MAP
}

# Structural = no data column, and not a reference/outcome node.
STRUCTURAL = {n[0] for n in NODES
              if n[2] is None and n[0] not in (ARRIVES_NODE, DEPARTURE_NODE)}
MEASURED   = {n[0] for n in NODES if n[2] is not None}


# ── Label helper ──────────────────────────────────────────────────────────────
def _wrap(name: str) -> str:
    """Break a node name onto two lines so it fits inside a box."""
    if ": " in name:
        return name.replace(": ", ":<br>", 1)
    if len(name) > 15 and " " in name:
        # break near the middle on a space
        mid = len(name) // 2
        left = name.rfind(" ", 0, mid + 4)
        if left > 0:
            return name[:left] + "<br>" + name[left + 1:]
    return name


# ── Flowchart renderer ────────────────────────────────────────────────────────
def build_flowchart(nd, source, highlight_node=None, no_data_nodes=None):
    """
    Renders the ground-ops dependency flowchart.

    nd            : dict {node_name: delay_mins}  — 0 = on time
    source        : str  — node to label as origin (pass "" for actual-data view)
    highlight_node: str  — flash-highlight one node (used by animation)
    no_data_nodes : set  — measured nodes to draw as "no data" (dashed) this render
                    (e.g. per-flight missing milestones, or columns empty across the
                    dataset). Lets a node honestly show "no data" instead of a
                    misleading on-time box, and flip to a coloured box once data lands.

    Structural (no-data) nodes are always drawn neutral with a dashed border and
    never show a delay value, regardless of nd.
    """
    no_data_nodes = no_data_nodes or set()
    fig = go.Figure()
    pos = {n[0]: (n[3], n[4]) for n in NODES}
    BW, BH = 0.57, 0.42

    _light = is_light()
    _normal_fill   = "#e8edf8" if _light else "#16213e"
    _normal_text   = "#1a2340" if _light else "#7a8fb0"
    # Bright orange = no data (structural process step)
    _struct_fill   = "#fff1e0" if _light else "rgba(255,140,26,0.13)"
    _struct_bdr    = "#e67e22" if _light else "#ff8c1a"
    _struct_text   = "#a85b00" if _light else "#ffb877"
    _ref_fill      = "#d0e4ff" if _light else "#1a3a6e"
    _ref_text      = "#1a3a6e" if _light else "#cce5ff"
    _source_fill   = "#fce8e8" if _light else "#7b1818"
    _source_text   = "#8b0000" if _light else "#ffffff"
    _dep_late_fill = "#fce8e8" if _light else "#5a0d0d"
    _dep_marg_fill = "#fef3e2" if _light else "#4a3300"
    _dep_ok_fill   = "#d4f5e2" if _light else "#0d3b20"
    _dep_text      = "#1a2340" if _light else "#ffffff"
    _edge_norm     = "rgba(90,100,130,0.55)" if _light else "rgba(90,100,130,0.30)"

    # Edges — drawn as splines routed through dummy waypoints (see _layout).
    for (parent, child), waypts in ROUTES.items():
        if parent not in pos or child not in pos:
            continue
        cd = 0 if child in STRUCTURAL else nd.get(child, 0)
        pd_ = 0 if parent in STRUCTURAL else nd.get(parent, 0)
        affected = cd > 0.5 or pd_ > 0.5
        clr = "#e74c3c" if affected else _edge_norm
        wid = 2.4 if affected else 0.7
        (px, py), (cx, cy) = waypts[0], waypts[-1]
        fig.add_trace(go.Scatter(
            x=[px, cx], y=[py - BH, cy + BH], mode="lines",
            line=dict(color=clr, width=wid),
            showlegend=False, hoverinfo="skip",
        ))

    shapes, annotations = [], []

    for name, team, col, x, y, *_ in NODES:
        is_struct = name in STRUCTURAL or name in no_data_nodes
        delay = 0 if is_struct else nd.get(name, 0)
        dash = None

        if name == source and source:
            fill, border, bw = _source_fill, "#ff4444", 3
            fc = _source_text
            label = f"<b>{_wrap(name)}</b><br>⏱ +{delay:.0f} min  (SOURCE)"
        elif name == DEPARTURE_NODE:
            if delay > 4:
                fill, border = _dep_late_fill, "#e74c3c"
            elif delay > 0:
                fill, border = _dep_marg_fill, "#f39c12"
            else:
                fill, border = _dep_ok_fill, "#2ecc71"
            bw, fc = 3, _dep_text
            label = (f"<b>✈️ PUSHBACK</b><br>+{delay:.0f} min late"
                     if delay > 0.5 else "<b>✈️ PUSHBACK</b><br>✅ On Time")
        elif name == ARRIVES_NODE:
            fill, border, bw = _ref_fill, "#5dade2", 2
            fc = _ref_text
            label = "<b>✈️ Aircraft Arrives</b><br>(AIBT · On-Chocks)"
        elif is_struct:
            # No-data process step — BRIGHT ORANGE so the data gap is obvious.
            fill, border, bw = _struct_fill, _struct_bdr, 2
            fc, dash = _struct_text, "dot"
            crucial = name in CRUCIAL_NODES
            tag = "🔑 CRUCIAL · NO DATA" if crucial else "⚠ NO DATA"
            label = f"<b>{_wrap(name)}</b><br><span style='font-size:7px'>{tag}</span>"
        elif name in CRUCIAL_NODES:
            # Measured AND crucial — gold ring always, heat fill if late.
            if delay > 0.5:
                intensity = min(delay / 25.0, 1.0)
                r = int(180 + 51 * intensity); g = int(100 - 90 * intensity); b = int(40 - 35 * intensity)
                fill = f"rgba({r},{g},{b},0.88)"
            else:
                fill = _normal_fill
            border, bw, fc = "#f1c40f", 3, "#ffffff" if delay > 0.5 else _normal_text
            label = (f"<b>🔑 {_wrap(name)}</b><br>⚡ +{delay:.0f} min" if delay > 0.5
                     else f"<b>🔑 {_wrap(name)}</b><br><span style='font-size:7px'>CRUCIAL gate</span>")
        elif delay > 0.5:
            intensity = min(delay / 25.0, 1.0)
            r = int(180 + 51 * intensity)
            g = int(100 - 90 * intensity)
            b = int(40 - 35 * intensity)
            fill = f"rgba({r},{g},{b},0.85)"
            border = "#e74c3c" if delay > 12 else "#e67e22" if delay > 6 else "#f39c12"
            bw, fc = 2, "#ffffff"
            label = f"<b>{_wrap(name)}</b><br>⚡ +{delay:.0f} min"
        else:
            fill = _normal_fill
            border = TEAM_COLORS.get(team, "#555")
            bw, fc = 1, _normal_text
            label = f"<b>{_wrap(name)}</b>"

        if name == highlight_node:
            shapes.append(dict(
                type="rect",
                x0=x - BW - 0.06, y0=y - BH - 0.06,
                x1=x + BW + 0.06, y1=y + BH + 0.06,
                fillcolor="rgba(255,255,255,0.10)", opacity=1.0,
                line=dict(color="#ffffff", width=5),
                xref="x", yref="y",
            ))
            border, bw = "#ffffff", 4

        shapes.append(dict(
            type="rect",
            x0=x - BW, y0=y - BH, x1=x + BW, y1=y + BH,
            fillcolor=fill, opacity=0.9 if is_struct else 0.95,
            line=dict(color=border, width=bw, dash=dash),
            xref="x", yref="y",
        ))
        annotations.append(dict(
            x=x, y=y, text=label,
            showarrow=False,
            font=dict(size=7.6, color=fc, family="Inter, Arial"),
            xref="x", yref="y", align="center",
        ))

    title_text = (
        f"Ground Ops Flow  ·  <b style='color:#f39c12'>{source}</b>  (simulated)"
        if source else
        "Ground Ops Flow  ·  <b style='color:#4d9fff'>Actual Flight Data</b>"
    )
    # Swimlane column bands + BU header labels
    def _faint(hexc, a):
        h = hexc.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

    _band_lo, _band_hi = COLUMN_BANDS_Y
    _band_shapes = []
    for _hlabel, _hx, _hy, _hcolor in COLUMN_HEADERS:
        _band_shapes.append(dict(
            type="rect", x0=_hx - 0.62, x1=_hx + 0.62, y0=_band_lo, y1=_band_hi,
            fillcolor=_faint(_hcolor, 0.10 if _light else 0.07),
            line=dict(width=0), layer="below", xref="x", yref="y",
        ))
        annotations.append(dict(
            x=_hx, y=_hy, text=f"<b>{_hlabel}</b>", showarrow=False,
            font=dict(size=10.5, color=_hcolor, family="Inter, Arial"),
            xref="x", yref="y",
        ))
    shapes = _band_shapes + shapes   # bands sit behind nodes & edges

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        template=chart_template(),
        height=1080,
        xaxis=dict(range=[0.0, 12.0], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[pos[DEPARTURE_NODE][1] - 0.85, _Y_TOP + 2.4],
                   showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=44, b=5),
        showlegend=False,
        title=dict(text=title_text, font=dict(size=13, color=chart_fc()),
                   x=0.01, xanchor="left"),
    )
    return fig


# ── Timeline renderer ─────────────────────────────────────────────────────────
def build_timeline(nd, source):
    acts = [(n, t, off)
            for n, t, _c, _x, _y, off in NODES
            if off is not None and n not in (ARRIVES_NODE, DEPARTURE_NODE)]
    acts.sort(key=lambda a: a[2])

    names        = [a[0] for a in acts]
    teams        = [a[1] for a in acts]
    plan_times   = [a[2] for a in acts]
    delays       = [nd.get(a[0], 0) for a in acts]
    actual_times = [p + d for p, d in zip(plan_times, delays)]
    dep_shift    = max(nd.get(DEPARTURE_NODE, 0), 0)

    fig = go.Figure()

    for i, name in enumerate(names):
        d = delays[i]
        if abs(d) < 0.5:
            continue
        clr = "rgba(231,76,60,0.35)" if d > 4 else "rgba(243,156,18,0.35)"
        fig.add_trace(go.Scatter(
            x=[plan_times[i], actual_times[i]], y=[name, name],
            mode="lines", line=dict(color=clr, width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=plan_times, y=names,
        mode="markers", name="Planned",
        marker=dict(symbol="circle", size=11,
                    color="rgba(130,145,170,0.50)",
                    line=dict(color="rgba(180,195,215,0.80)", width=1.5)),
        hovertemplate="<b>%{y}</b><br>Planned: %{x} min from SOBT<extra></extra>",
    ))

    mc, ms = [], []
    for i, (name, d) in enumerate(zip(names, delays)):
        if name == source:
            mc.append("#ff4444"); ms.append(16)
        elif d > 10:
            mc.append("#e74c3c"); ms.append(14)
        elif d > 4:
            mc.append("#e67e22"); ms.append(13)
        elif d > 0.5:
            mc.append("#f39c12"); ms.append(12)
        else:
            mc.append(TEAM_COLORS.get(teams[i], "#3498db")); ms.append(11)

    hover_texts = [
        f"<b>{names[i]}</b><br>"
        f"Planned: {plan_times[i]:+d} min from SOBT<br>"
        f"Actual: {actual_times[i]:+.0f} min from SOBT<br>"
        f"Shift: {'+' if delays[i] >= 0 else ''}{delays[i]:.0f} min"
        for i in range(len(names))
    ]
    fig.add_trace(go.Scatter(
        x=actual_times, y=names,
        mode="markers+text", name="Actual",
        marker=dict(symbol="circle", size=ms, color=mc,
                    line=dict(color=chart_fc(), width=1.5)),
        text=[f" +{d:.0f}m" if d > 1.5 else "" for d in delays],
        textposition="middle right",
        textfont=dict(size=9, color="#ffcc77"),
        hovertext=hover_texts, hoverinfo="text",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="#f4a621", line_width=2)
    fig.add_annotation(x=0, y=-0.07, xref="x", yref="paper",
                       text="<b>SOBT</b>", showarrow=False,
                       font=dict(color="#f4a621", size=11),
                       xanchor="center", yanchor="top")

    if dep_shift > 0.5:
        fig.add_vline(x=dep_shift, line_dash="dot", line_color="#e74c3c", line_width=2)
        fig.add_annotation(x=dep_shift, y=1.06, xref="x", yref="paper",
                           text=f"<b>Pushback +{dep_shift:.0f} min late</b>", showarrow=False,
                           font=dict(color="#e74c3c", size=11),
                           xanchor="center", yanchor="bottom")

    x_min = min(plan_times) - 15
    x_max = max(max(actual_times), dep_shift + 2, 5) + 12
    left_margin = max(170, max((len(n) for n in names), default=20) * 7)

    fig.update_layout(
        template=chart_template(),
        height=max(520, len(acts) * 30 + 120),
        xaxis=dict(title="Minutes Relative to Scheduled Departure (SOBT = 0)",
                   range=[x_min, x_max],
                   gridcolor=chart_gc(), zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10),
                   gridcolor=chart_gc(), automargin=True),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=left_margin, r=90, t=70, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.10,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


# ── Per-flight cascade helper ─────────────────────────────────────────────────
def get_flight_cascade_nd(flight_row):
    """
    Build the nd dict from a single flight's actual milestone delays.

    Only MEASURED nodes (those with a delay column) get values; structural nodes
    stay 0 and render neutral.

    Returns
    -------
    nd          : dict {node_name: delay_mins}
    source_node : str — measured node with the largest delay (for highlighting)
    """
    nd = {n[0]: 0.0 for n in NODES}

    for name, team, col, *_ in NODES:
        if col is None:
            continue
        try:
            val = flight_row.get(col, None)
            if val is None or (hasattr(val, '__class__') and val.__class__.__name__ == 'float' and np.isnan(val)):
                val = 0.0
            nd[name] = max(0.0, float(val))
        except (TypeError, ValueError):
            nd[name] = 0.0

    # Departure node — use actual flight departure delay
    try:
        dep_delay = float(flight_row.get("Target_Departure_Delay_mins", 0) or 0)
        nd[DEPARTURE_NODE] = max(0.0, dep_delay)
    except (TypeError, ValueError):
        nd[DEPARTURE_NODE] = 0.0

    # Source = measured node with highest actual delay (for highlighting)
    candidates = [
        (name, nd[name]) for name, team, col, *_ in NODES
        if col is not None and nd[name] > 0
    ]
    source_node = max(candidates, key=lambda x: x[1])[0] if candidates else ""

    return nd, source_node


# ── "No data" — an ambiguous absence, not a stated collection failure ──────────
# SATS does not perform every service for every airline, so a milestone that is
# absent for a flight may mean the service does not apply to that airline/flight
# OR that the time was simply not captured. We deliberately do NOT distinguish the
# two — both are surfaced as "No data available".
NO_DATA_LABEL = "No data available"
NO_DATA_NOTE  = (
    "ℹ️ **“No data available”** means a milestone wasn’t recorded for the flight(s). "
    "SATS doesn’t perform every service for every airline, so this can mean the service "
    "doesn’t apply to that airline/flight **or** the time wasn’t captured — the two are "
    "not distinguished here."
)


def _is_missing(v):
    """True for None / NaN / NaT without needing a pandas import."""
    if v is None:
        return True
    try:
        return v != v          # NaN and NaT are not equal to themselves
    except Exception:
        return False


def _raw_actual_col(delay_col):
    """Best-effort map a derived delay column back to its raw '.actual' timestamp
    column (point milestones store 0 — not NaN — when absent, so the raw actual is
    the only way to tell 'absent' from 'on time')."""
    m = re.match(r"milestone_(.+?)_analysis_", delay_col or "")
    if not m:
        return None
    return "milestone_" + m.group(1).replace("_", ".", 1) + ".actual"


def flight_no_data_nodes(flight_row):
    """Set of node names with NO data for this flight — structural nodes (no column
    at all) plus measured milestones whose actual time is missing for the flight.
    See NO_DATA_NOTE: an absence is ambiguous (service may not apply, or not
    captured), so callers should label these as NO_DATA_LABEL, never 'on time'."""
    index = getattr(flight_row, "index", [])
    out = set()
    for name, team, col, *_ in NODES:
        if name in (ARRIVES_NODE, DEPARTURE_NODE):
            continue
        if col is None:                                   # structural: no column
            out.add(name)
            continue
        if "DurationOverrun" in col or "ActualDuration" in col:
            # duration metrics are NaN when the milestone is absent
            if _is_missing(flight_row.get(col, None)):
                out.add(name)
        else:
            raw = _raw_actual_col(col)                    # point milestone: check raw actual
            if raw is not None and raw in index and _is_missing(flight_row.get(raw, None)):
                out.add(name)
    return out


# ── Orphan milestones — recorded in data but not on the cascade map ────────────
# Lets NEW milestones (e.g. once SATS adds them, or for airlines with services the
# map doesn't show) surface per-flight without first hand-adding a node. As soon as
# a column carries data it shows up here; add it to _ROWS later to put it on the map.
_MS_TEAM = {
    "ramp": "ramp", "security": "security", "aic": "aic", "cabinSvc": "cabin",
    "cargo": "cargo", "TPO": "cargo", "pax": "pax", "loadControl": "loadctrl",
    "baggage": "baggage", "gse": "techramp",
}
WIRED_COLS = {n[2] for n in NODES if n[2]}


def _base_of(col):
    m = re.match(r"milestone_(.+?)_analysis_", col or "")
    return m.group(1) if m else col


# Milestone bases already represented on the map — measured (any metric) OR a
# structural node that stands for that milestone — so they are NOT orphans.
WIRED_BASES  = {_base_of(c) for c in WIRED_COLS}
STRUCT_BASES = {
    "aic_cleaning", "aic_cleaning_start", "aic_cleaning_end",
    "cabinSvc_unloading", "baggage_firstTripToBay", "baggage_lastTripToBay",
    "baggage_TripToBay",
}
MAPPED_BASES = WIRED_BASES | STRUCT_BASES


def _pretty_milestone(col):
    """('milestone_ramp_NOTOC_analysis_Delay_mins') -> ('Ramp', 'NOTOC')."""
    m = re.match(r"milestone_(.+?)_analysis_", col)
    base = m.group(1) if m else col
    team = base.split("_")[0]
    rest = base[len(team) + 1:] if "_" in base else base
    bu = TEAM_LABELS.get(_MS_TEAM.get(team, team), team)
    label = rest.replace("_", " ").title().replace("Plb", "PLB").replace("Notoc", "NOTOC")
    return bu, (label or base)


def orphan_milestone_cols(df_columns):
    """Milestone delay/overrun metric columns present in the data but not wired to
    any cascade node (deduped to one metric per milestone base)."""
    seen, out = set(), []
    for c in df_columns:
        if not c.startswith("milestone_"):
            continue
        if "_analysis_Delay_mins" not in c and "_analysis_DurationOverrun_mins" not in c:
            continue
        base = _base_of(c)
        if base in MAPPED_BASES or base in seen:
            continue
        seen.add(base)
        out.append(c)
    return out


def flight_orphan_milestones(flight_row, df_columns):
    """Per-flight orphan milestones that were actually recorded for THIS flight,
    as [(bu, label, value_mins, col)]. value 0 = recorded/on-time."""
    cols = set(getattr(flight_row, "index", []))
    out = []
    for col in orphan_milestone_cols(df_columns):
        raw_cols = _base_actual_cols(col, cols)
        if raw_cols:
            present = any(not _is_missing(flight_row.get(rc)) for rc in raw_cols)
        elif "DurationOverrun" in col or "ActualDuration" in col:
            present = not _is_missing(flight_row.get(col, None))
        else:
            present = False
        if not present:
            continue
        try:
            v = float(flight_row.get(col, 0) or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v != v:
            v = 0.0
        bu, label = _pretty_milestone(col)
        out.append((bu, label, v, col))
    return sorted(out, key=lambda x: (x[0], x[1]))


def _base_actual_cols(col, cols):
    """All raw '.actual' columns sharing a metric column's milestone base — handles
    interval milestones (_start/_end, e.g. pax.boarding) and point milestones."""
    dotted = "milestone_" + _base_of(col).replace("_", ".", 1)
    return [x for x in cols if x.startswith(dotted) and x.endswith(".actual")]


def orphan_milestone_summary(df):
    """Aggregate off-map milestones that carry data: [(bu, label, n_recorded,
    n_with_signal, col)], busiest first. Surfaces milestones present in the data
    but not on the cascade map (e.g. Pax Boarding) so they can be promoted later."""
    cols = set(df.columns)
    out = []
    for col in orphan_milestone_cols(df.columns):
        s = pd.to_numeric(df[col], errors="coerce")
        n_signal = int((s.abs() > 0.5).sum())
        raw_cols = _base_actual_cols(col, cols)
        if raw_cols:
            n_rec = int(df[raw_cols].notna().any(axis=1).sum())
        elif "DurationOverrun" in col or "ActualDuration" in col:
            n_rec = int(s.notna().sum())
        else:
            n_rec = n_signal
        if n_rec == 0 and n_signal == 0:
            continue
        bu, label = _pretty_milestone(col)
        out.append((bu, label, n_rec, n_signal, col))
    return sorted(out, key=lambda x: -x[2])


def measured_no_data_nodes(df):
    """Measured nodes whose milestone has ~no recorded data across the whole df, so
    the aggregate flowchart can draw them as 'no data' (dashed) instead of a
    misleading on-time box. Flips off automatically once the column gains data —
    pass the result to build_flowchart(..., no_data_nodes=...)."""
    cols = set(df.columns)
    out = set()
    for name, team, col, *_ in NODES:
        if not col or name in (ARRIVES_NODE, DEPARTURE_NODE):
            continue
        if "DurationOverrun" in col or "ActualDuration" in col:
            has = col in cols and bool(pd.to_numeric(df[col], errors="coerce").notna().any())
        else:
            raw = _raw_actual_col(col)
            if raw in cols:
                has = bool(df[raw].notna().any())
            else:
                # raw actual name not resolvable (e.g. synthesised start/end like
                # TPO_CargoTowing) — trust the metric: any non-zero value = has data.
                has = col in cols and bool((pd.to_numeric(df[col], errors="coerce").abs() > 0.5).any())
        if not has:
            out.add(name)
    return out
