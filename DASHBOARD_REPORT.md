# SATS Ground Operations Intelligence Dashboard
## Technical Reference & Troubleshooting Guide

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [System Architecture](#2-system-architecture)
3. [File Structure](#3-file-structure)
4. [Data Pipeline](#4-data-pipeline)
5. [Running the App](#5-running-the-app)
6. [Global Sidebar Filters](#6-global-sidebar-filters)
7. [Page-by-Page Feature Guide](#7-page-by-page-feature-guide)
   - [Home](#71-home)
   - [Flight Monitor](#72-flight-monitor)
   - [Overview](#73-overview)
   - [When Delays Happen](#74-when-delays-happen)
   - [Delay Attribution](#75-delay-attribution)
   - [Activity Analysis](#76-activity-analysis)
   - [BU Impact Analyser](#77-bu-impact-analyser)
   - [Cascade Effect](#78-cascade-effect)
   - [Flight Investigation](#79-flight-investigation)
   - [Flight Deep Dive](#710-flight-deep-dive)
   - [Delay Predictor](#711-delay-predictor)
   - [Ask the Data](#712-ask-the-data)
   - [Data Quality](#713-data-quality)
8. [ML Model](#8-ml-model)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Adding New Data](#10-adding-new-data)
11. [Known Limitations](#11-known-limitations)

---

## 1. Project Overview & Goals

### Background
SATS Ltd is the primary ground handler at Singapore Changi Airport (Terminals 2 & 3). Ground handling involves coordinating 8+ teams across ~30 activities that must all complete in sequence before a flight can depart. A delay in any one activity can cascade into a departure delay.

### Goals
| Goal | Status |
|------|--------|
| Measure on-time departure performance across all SATS-handled flights | ✅ Done |
| Identify which Business Units and activities drive delays most | ✅ Done |
| Model how one activity's lateness ripples downstream (cascade effect) | ✅ Done |
| Predict whether a given flight will be delayed before departure | ✅ Done |
| Give ground workers a case-by-case flight investigation table | ✅ Done |
| Make all insights filterable by date, terminal, airline, aircraft type | ✅ Done |

### Key Numbers
- **25,327 flights** analysed (SATS-handled departures with full milestone data)
- **Date range:** Sep 2025 – Mar 2026 (Q1, Q3, Q4)
- **Terminals:** T2 and T3, Changi Airport
- **~37–40% on-time rate** (departures within ±0 min of scheduled)
- **~41% delayed rate** (> 4 min late)
- **Avg delay (when delayed):** ~21 min (capped at 120 min to exclude cancellations/data errors)

> **Why 25,327 and not ~84,000?**
> The raw datasets contain ~141,955 departure records total, but only a subset of flights are
> SATS-handled AND have milestone/activity data. An inner join on flight ID between the flight
> information table and the milestone table yields 25,327 records. This is correct — SATS does
> not handle every Changi flight (other handlers exist).

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Raw Data (CSV / XLSX)                                       │
│  Data/Old DSM Data/  +  Data/New DSM Data/                  │
└────────────────┬────────────────────────────────────────────┘
                 │  prepare_data.py
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Dashboard/data/flights.parquet   (processed, ~25k rows)    │
│  Dashboard/data/model.pkl         (trained ML pipeline)     │
└────────────────┬────────────────────────────────────────────┘
                 │  load_data() — cached in Streamlit memory
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Streamlit App  (Dashboard/Home.py + pages/)                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  utils/loader.py   — data loading + sidebar filters   │  │
│  │  utils/style.py    — global CSS + inject_css()        │  │
│  └───────────────────────────────────────────────────────┘  │
│  Pages: 8 analysis pages + Home                             │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack
| Component | Technology |
|-----------|-----------|
| Web framework | Streamlit 1.52.1 |
| Charts | Plotly (go + express) |
| ML | scikit-learn (Ridge regression, XGBoost pipeline) |
| Data processing | pandas, numpy |
| Data storage | Parquet (Apache Arrow) |
| Styling | Custom CSS (glassmorphism, JetBrains Mono, Inter) |
| Fonts | Google Fonts — Inter + JetBrains Mono |

---

## 3. File Structure

```
Sats Project/
├── Data/
│   ├── Old DSM Data/
│   │   ├── dsmlive_flight_Nov2025.xlsx        # Flight info (Nov 2025)
│   │   └── dsmlive_milestone_Nov2025.xlsx     # Milestone/activity data (Nov 2025)
│   └── New DSM Data/
│       ├── dsmlive_flight_01022026_07042026.csv   # Flight info (Feb–Apr 2026)
│       └── dsmlive_GHAMS_01022026_07042026.csv    # Milestone/activity (Feb–Apr 2026)
├── Dashboard/
│   ├── Home.py                    # Landing page
│   ├── prepare_data.py            # Data ingestion + ML training script
│   ├── requirements.txt           # Python dependencies
│   ├── run.sh                     # Launch script
│   ├── .streamlit/
│   │   └── config.toml            # Theme + server config
│   ├── data/
│   │   ├── flights.parquet        # Processed data (generated by prepare_data.py)
│   │   └── model.pkl              # Trained ML model (generated by prepare_data.py)
│   ├── pages/
│   │   ├── 01_Flight_Monitor.py
│   │   ├── 02_Overview.py
│   │   ├── 03_When_Delays_Happen.py
│   │   ├── 04_Delay_Attribution.py
│   │   ├── 05_Activity_Analysis.py
│   │   ├── 06_BU_Impact.py
│   │   ├── 07_Cascade_Effect.py
│   │   ├── 08_Flight_Investigation.py
│   │   ├── 09_Flight_Deep_Dive.py
│   │   ├── 10_Delay_Predictor.py
│   │   ├── 11_Ask_Data.py
│   │   └── 12_Data_Quality.py
│   └── utils/
│       ├── __init__.py
│       ├── loader.py              # load_data(), render_date_filters(), load_model()
│       └── style.py               # inject_css(), kpi_card(), page_header()
└── ML Notebook/
    └── MLnotebook.ipynb           # Exploratory ML work
```

---

## 4. Data Pipeline

### Step 1 — Raw Sources
Two time periods of data, each with two tables:

| File | Contents | Rows (approx) |
|------|----------|---------------|
| `dsmlive_flight_Nov2025.xlsx` | Flight metadata, scheduled/actual times, delays | ~50k |
| `dsmlive_milestone_Nov2025.xlsx` | Ground activity milestones per flight | ~50k |
| `dsmlive_flight_01022026_07042026.csv` | Same schema, Feb–Apr 2026 | ~90k |
| `dsmlive_GHAMS_01022026_07042026.csv` | Same schema, Feb–Apr 2026 | ~90k |

### Step 2 — prepare_data.py
Run this script whenever new data arrives:
```bash
cd Dashboard
python prepare_data.py
```

What it does:
1. Loads all 4 raw files
2. Standardises column names across old/new formats
3. Computes engineered features:
   - `Target_Departure_Delay_mins` = actual offblock − scheduled offblock (minutes)
   - `Target_Departure_Delay_Class` = "On-Time" (≤0), "Acceptable" (1–4 min), "Delayed" (>4 min)
   - `Available_Ground_Time_mins` = scheduled departure − actual arrival
   - `Ground_Time_Ratio` = available / minimum required ground time
   - `Is_Ground_Time_Deficient` = 1 if available < minimum required
   - `Incoming_Delay_mins` = minutes the inbound aircraft was late
   - `Hour_of_Day`, `Day_of_Week`, `Month`, `Is_Weekend`
   - `milestone_*_analysis_Delay_mins` = actual milestone time − planned milestone time per activity
4. Inner-joins flight + milestone tables on flight ID → 25,327 rows
5. Saves `data/flights.parquet`
6. Trains the ML model and saves `data/model.pkl`

### Step 3 — loader.py
`load_data()` reads the parquet, parses datetimes, adds `_dep_date`, `_dep_month`, `_dep_quarter`, `_dep_year` columns, and caches the result in Streamlit's memory.

---

## 5. Running the App

### Prerequisites
```bash
pip install -r requirements.txt
```

Key packages: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `pyarrow`, `openpyxl`, `plotly`

### First-Time Setup
```bash
cd "/Users/prathosh/Desktop/Sats Project/Dashboard"
python prepare_data.py        # generates flights.parquet + model.pkl
streamlit run Home.py
```

### Normal Launch
```bash
cd "/Users/prathosh/Desktop/Sats Project/Dashboard"
streamlit run Home.py
```
Or use the provided script:
```bash
bash run.sh
```

The app opens at **http://localhost:8501** by default.

### Streamlit Config (`.streamlit/config.toml`)
```toml
[theme]
base = "dark"
primaryColor = "#1a73e8"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1e2130"
textColor = "#fafafa"

[server]
headless = true
```

---

## 6. Global Sidebar Filters

Every page (except Home and Delay Predictor) has these sidebar filters, applied **before** any page-level filtering:

| Filter | Column Used | Notes |
|--------|-------------|-------|
| **Quarter** | `_dep_quarter` | Q1 (Jan–Mar), Q3 (Jul–Sep), Q4 (Oct–Dec) present in current data |
| **Month** | `_dep_month` | Cascades from quarter selection |
| **Terminal** | `origin_terminal` | Terminal 2 or Terminal 3 |
| **Aircraft Type** | `aircraft_bodyType` | Narrowbody / Widebody |
| **Aircraft Model (ICAO)** | `aircraft_typeICAO` | E.g. B738, A320, B77W |
| **Destination** | `destination_iata` | IATA airport code |

A caption at the bottom of the sidebar shows **"X of 25,327 flights shown."**

Each page passes a unique `page_key` to `render_date_filters()` (e.g. `"overview"`, `"cascade"`) to prevent widget key conflicts across pages.

---

## 7. Page-by-Page Feature Guide

### 7.1 Home

**File:** `Dashboard/Home.py`

**Purpose:** Landing page with animated hero, headline KPIs, navigation tiles, and team performance overview.

**Features:**
- Animated radar-ring sweep (CSS `@keyframes`)
- Animated counter roll-up for 4 KPIs (JavaScript `requestAnimationFrame`)
- Navigation tiles — click "Open X →" to jump to any page
- Team performance mini-cards showing each BU's on-time %

**KPIs shown:**
| KPI | Formula |
|-----|---------|
| Flights Analysed | Total rows with a valid `Target_Departure_Delay_Class` |
| On-Time Rate | % with class = "On-Time" |
| Delayed Rate | % with class = "Delayed" |
| Avg Delay | Mean `Target_Departure_Delay_mins` clipped at 120 min, delayed flights only |

**Troubleshoot:**
- *KPIs show 0* → data not loaded; check `flights.parquet` exists and `prepare_data.py` was run
- *Nav tiles not clickable* → uses `st.page_link()` — ensure Streamlit ≥ 1.36
- *Sidebar missing* → hard-refresh (Cmd+Shift+R); every page forces `initial_sidebar_state="expanded"`

---

### 7.2 Flight Monitor

**File:** `pages/01_Flight_Monitor.py`

**Purpose:** Real-time flight departure monitor. Sorts flights by scheduled departure time and displays their ML-predicted delay probability.

**Key Features:**
- **Dynamic delay risk color coding** (🔴 high risk >50%, 🟡 medium risk 30-50%, 🟢 on track).
- Live operations overview panel listing the number of flights in risk categories.
- Detailed flight operations filters (by Terminal, Carrier, Aircraft Type, Remote Bay).
- **Inline Cascade Inspector**: select any flight to view its flow map or timeline directly on the page.

---

### 7.3 Overview

**File:** `pages/02_Overview.py`

**Purpose:** High-level dashboard summarizing operations KPIs, carrier rankings, and delay distributions.

**Key charts:**
- Delay class breakdown (donut/pie chart)
- Departure delay distribution histogram (with P50, Mean, P90 markers)
- Top 15 carriers by delay rate bar chart
- Delay by aircraft body type box plot
- Monthly trend line

**Troubleshoot:**
- *Charts blank* → check if sidebar filters have reduced the dataset to 0 rows.
- *Avg delay showing very high (>30 min)* → outlier capping is applied at 120 min; if still high, check for data quality issues in `prepare_data.py`.

---

### 7.4 When Delays Happen

**File:** `pages/03_When_Delays_Happen.py`

**Purpose:** Day-of-week and hour-of-day heatmaps that reveal peak congestion and delay risk times.

**Key charts:**
- Hour-of-day × Day-of-week delay rate heatmap
- Departure count heatmap (volume overlay)
- Peak hour delay rate bar chart

---

### 7.5 Delay Attribution

**File:** `pages/04_Delay_Attribution.py`

**Purpose:** Classify each departure delay into SATS-attributable vs external root causes (lightning warnings, inbound late, tight scheduling).

**Key Features:**
- **SLA Exclusion Analysis**: trace lightning warning (LW) windows and isolate weather-driven delays.
- Attribution breakdown donut chart (SATS Operation, Propagated, Tight Schedule, Weather/LW, Compound).
- Inbound Arrival Condition vs Departure Performance table (calculating delta pp vs baseline).
- SATS-attributable delay drill-down (bar chart showing which SATS activities contributed most to delays).

---

### 7.6 Activity Analysis

**File:** `pages/05_Activity_Analysis.py`

**Purpose:** Three-level drill-down from Business Unit → Activity → Detailed statistics.

**Navigation (uses `st.session_state`):**
1. **Level 1** — BU grid cards. Click any BU card (Ramp, PAX, AIC, Cabin, Cargo, Security, Load Control).
2. **Level 2** — Activity cards within that BU. Click "Explore →".
3. **Level 3** — Full analysis: delay histogram, hour-of-day bar chart, departure correlation scatter, regression summary.

**Breadcrumb** at top shows: `🔧 Activity Analysis › Ramp › Ramp: At Bay`

---

### 7.7 BU Impact Analyser

**File:** `pages/06_BU_Impact.py`

**Purpose:** Combined Business Unit performance view + Cascade flowchart. See which of your BU's activities are late and where they sit in the turnaround dependency chain.

**Layout:**
- **Top** — BU selector cards showing on-time % and progress bars.
- **Left column** — Activity list for the selected BU, sorted by % late, with an "Analyse" button per activity.
- **Right column** — Interactive cascade flowchart highlighting the selected BU's nodes.
- **Below flowchart** — Regression summary card showing departure delay minutes per minute of activity overrun.

---

### 7.8 Cascade Effect

**File:** `pages/07_Cascade_Effect.py`

**Purpose:** Simulate how a delay in one ground activity propagates downstream through the entire turnaround chain.

**How it works:**
1. Select an activity (e.g. "Ramp: At Bay") and a delay amount (slider, 0–60 min)
2. BFS propagation uses Ridge regression coefficients between pairs of activities to calculate how much each downstream activity shifts
3. The departure delay is estimated from the source activity's regression model

**Features:**
- **Flow Map tab** — Plotly flowchart with nodes coloured by delay severity; SOURCE node highlighted in red
- **Timeline View tab** — Scatter plot of planned vs predicted actual milestone times relative to SOBT.
- **Animate Cascade button** — steps through the BFS propagation order, lighting up each node in sequence.
- **Impact table** — all activities with their delay status
- **Regression bar chart** — which activities have the highest departure delay coefficient
- **Pairwise transfer table** — how strongly each direct link transfers delay
- **Indirect Cascade Links** — displays hidden correlations between activities that are not directly linked.

---

### 7.9 Flight Investigation

**File:** `pages/08_Flight_Investigation.py`

**Purpose:** Fully filterable, sortable, copyable table of every flight for case-by-case operational investigation.

**In-page filters (applied on top of sidebar filters):**
- Flight Departure Status (All / On-Time / Acceptable / Delayed)
- Min and Max Departure Delay inputs
- Activity Delay Threshold sliders (AND logic)

**Sort options:**
- Most Recent (default, using raw scheduled departure time)
- Dep Delay (min), Date, Status, Destination, Aircraft, Incoming Delay

---

### 7.10 Flight Deep Dive

**File:** `pages/09_Flight_Deep_Dive.py`

**Purpose:** Select any individual flight and see its full ground activity Gantt chart + narrative.

**Features:**
- Flight selector dropdown (up to 1,000 most recent flights, sorted by scheduled departure)
- **Summary cards** — carrier, aircraft, terminal, destination, scheduled/actual dep
- **Gantt chart** — planned bars (grey) vs actual bars (green = on time, red = late) for each ground activity
- **Delay narrative** — auto-generated text: first delayed activity, cascade effects, final departure outcome
- **Special handling requirements** — WCHR, UMNR, MEDA, MAAS, etc.
- **Milestone delay details table** — collapsible list of milestone delays per team.

---

### 7.11 Delay Predictor

**File:** `pages/10_Delay_Predictor.py`

**Purpose:** ML-powered departure delay risk predictor. Enter flight parameters and get an instant prediction.

**Input features:**
- Carrier, Aircraft Type, Aircraft Body Type, Terminal, Destination
- Departure Hour, Day of Week, Month
- Incoming Flight Delay, Available Ground Time, Minimum Required Ground Time, Remote Bay
- Special Handling counts (WCHR, UMNR, MEDA, MAAS)
- Lightning Warning conditions (active at departure, overlap duration, warnings on date)

---

### 7.12 Ask the Data

**File:** `pages/11_Ask_Data.py`

**Purpose:** Natural Language Analytics interface. Ask questions in plain English, and the AI will analyze all flight records, write python code locally, and report the specific data and text insight.

**Features:**
- **Provider selector** (Groq, Gemini, Anthropic) with custom API key saving.
- Suggested query buttons for quick exploration.
- Interactive conversation bubbles showing questions, AI insights, tables, and Plotly charts.
- View Generated Code panel for full transparency.

---

### 7.13 Data Quality

**File:** `pages/12_Data_Quality.py`

**Purpose:** Completeness observatory showing the data health and null rates for every milestone, terminal, carrier, and carrier field.

**Key Features:**
- **Overall Data Health Score** gauge chart.
- Column-by-column Milestone Delay Coverage horizontal bar chart.
- Average milestone coverage per Business Unit.
- Coverage trend line over time (monthly granularity).
- Group and Status filters to inspect specific columns.

---

## 8. ML Model

**Trained by:** `prepare_data.py`  
**Saved to:** `Dashboard/data/model.pkl`  
**Loaded by:** `utils/loader.py → load_model()`

### Model Bundle Contents
```python
{
    "pipeline":          sklearn.Pipeline,  # preprocessor + XGBoost classifier
    "label_encoder":     LabelEncoder,       # On-Time=0, Acceptable=1, Delayed=2 (approx)
    "feature_columns":   List[str],          # ordered list of all input features
    "numeric_feats":     List[str],          # features treated as numeric
    "categorical_feats": List[str],          # features treated as categorical (OHE)
}
```

### Key Features Used
- `Available_Ground_Time_mins`, `Ground_Time_Ratio`, `Is_Ground_Time_Deficient`
- `Incoming_Delay_mins`
- `Hour_of_Day`, `Day_of_Week`, `Month`, `Is_Weekend`
- `status_isRemoteBay`, `status_minGroundTime`
- `specialHandling_WCHR/UMNR/MEDA/MAAS`
- `aircraft_bodyType`, `aircraft_typeICAO`, `origin_terminal`, `destination_iata`
- `identification_carrierCode`

### Retraining
```bash
python prepare_data.py   # re-runs full pipeline, overwrites model.pkl
```

---

## 9. Troubleshooting Guide

### App won't start

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` | Package not installed | `pip install -r requirements.txt` |
| `flights.parquet not found` | prepare_data.py not run | `python prepare_data.py` |
| `model.pkl not found` | prepare_data.py not run | `python prepare_data.py` |
| Port 8501 already in use | Another Streamlit instance running | `pkill -f streamlit` then relaunch |

### Sidebar missing

1. Hard-refresh the browser: **Cmd+Shift+R** (Mac) / **Ctrl+Shift+R** (Windows)
2. Every page has `initial_sidebar_state="expanded"` — navigating to any page will reopen the sidebar
3. If the expand button (`>`) is invisible, the CSS in `utils/style.py` forces it visible via `[data-testid="stSidebarCollapsedControl"]`

### Data issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Avg delay shows 27+ min | Extreme outliers (cancellations logged as delays) | Already capped at 120 min in Overview and Home |
| All flights show "On-Time" | Target column not computed | Check `prepare_data.py` — `Target_Departure_Delay_Class` must be computed |
| Charts show NaN or empty | Column missing from parquet | Re-run `prepare_data.py`; check raw data column names haven't changed |
| Wrong flight count | Old parquet cached | Delete `data/flights.parquet`, re-run `prepare_data.py` |

### Chart-specific issues

| Page | Problem | Fix |
|------|---------|-----|
| Cascade Effect | `TypeError: only length-1 arrays` | Fixed — duplicate columns deduplicated with `list(dict.fromkeys(...))` |
| Cascade Effect | Timeline annotations overlap | Fixed — SOBT placed below x-axis, departure label above (different y positions) |
| Cascade Effect | Timeline shows 50+ on x-axis | Fixed — `x_max` now tightly clipped to `dep_shift + 12` |
| Flight Deep Dive | Gantt chart empty | Check milestone `_start` / `_end` columns in parquet |
| Delay Predictor | `Prediction failed` | Add missing columns as -999.0/Unknown — already implemented in code |
| BU Impact | `statsmodels` error | `pip install statsmodels` (needed for px.scatter trendline) |
| Activity Analysis | Drill-down stuck | Use "← Back" buttons; or refresh page to reset `st.session_state` |

### Performance issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Slow initial load | Parquet loading first time | Uses `@st.cache_data` — second load is instant |
| Slow Cascade Effect | Regression model building | Uses `@st.cache_data(show_spinner=...)` — trains once per session |
| Slow BU Impact | Same regression training | Uses `@st.cache_data` — trains once per session |
| Browser laggy on charts | Too many Plotly traces | Cascade flowchart has 60+ shapes; normal behaviour in dark mode |

---

## 10. Adding New Data

### Monthly Update Procedure

1. **Place new files in** `Data/New DSM Data/` (or update paths in `prepare_data.py`)
2. **Update file paths** in `prepare_data.py` if filenames changed
3. **Run:**
   ```bash
   cd Dashboard
   python prepare_data.py
   ```
4. **Restart Streamlit:**
   ```bash
   # Stop running instance (Ctrl+C in terminal)
   streamlit run Home.py
   ```

### Column Name Changes
If the raw data schema changes (column names renamed or added):
- Update `prepare_data.py` column mappings
- Update `ACTIVITY_COLS` in `pages/08_Flight_Investigation.py` if new milestone columns are added
- Update `NODES` in `pages/07_Cascade_Effect.py` and `pages/06_BU_Impact.py` if new milestones are tracked

### Adding a New Page
1. Create `pages/N_Page_Name.py` (N = next number)
2. Add to Home.py navigation tiles in the `pages` list
3. Call `inject_css()` at the top
4. Call `render_date_filters(df, page_key="unique_key")` for sidebar filters
5. Use `initial_sidebar_state="expanded"` in `st.set_page_config`

---

## 11. Known Limitations

| Limitation | Details |
|------------|---------|
| **UTC timestamps** | All times displayed in UTC. Changi Airport is UTC+8. Add 8 hours for local time. |
| **Data freeze** | Dashboard shows historical data only. Not connected to a live feed. Update by re-running `prepare_data.py`. |
| **SATS flights only** | ~25k flights, not all ~142k Changi departures. Non-SATS handlers' flights are excluded. |
| **Delay classification** | "On-Time" = ≤0 min, "Acceptable" = 1–4 min, "Delayed" = >4 min. This is a fixed business rule. |
| **Avg delay cap** | Avg delay metrics cap at 120 min to exclude cancellations/data errors. ~190 flights had >120 min logged delay. |
| **ML model** | Trained on Sep 2025 – Mar 2026 data. Accuracy may degrade for future periods with different patterns. Retrain regularly. |
| **Cascade animation** | Uses Python `time.sleep()` — not true async. May stutter on slow machines. Cannot use Plotly frames (they don't animate Shapes). |
| **px.scatter trendline** | Requires `statsmodels`. If not installed, BU Impact scatter will error. Run `pip install statsmodels`. |
| **Session state navigation** | Activity Analysis and BU Impact use `st.session_state` for drill-down. Refreshing the browser resets to Level 1. |
| **Flight Deep Dive dropdown** | Capped at 1,000 most recent flights for performance. Older flights must be found via Flight Investigation page. |

---

*Generated: May 2026 | SATS Ground Operations Intelligence Dashboard | Changi Airport T2 & T3*
