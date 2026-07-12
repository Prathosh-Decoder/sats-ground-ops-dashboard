"""
pages/10_Ask_Data.py
Natural Language Analytics — ask any question in plain English.

Supports three LLM providers (all free or near-free):
  • Groq       — free, console.groq.com           (llama-3.3-70b-versatile)
  • Gemini     — free 1500/day, aistudio.google.com (gemini-2.0-flash)
  • Anthropic  — pay-as-you-go, console.anthropic.com (claude-haiku-4-5)

Two-pass architecture:
  Pass 1 — LLM writes pandas/plotly code (sees schema, not raw data)
  [code runs locally on the full dataset]
  Pass 2 — aggregated result sent back → LLM writes a real text insight
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.loader import load_data, render_date_filters
from utils.style  import inject_css, chart_template, chart_fc, card_bg, card_text, card_sub, header_bg, header_border

st.set_page_config(
    page_title="Ask the Data | SATS",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
TEMPLATE = chart_template()
FC       = chart_fc()

# ── Data ──────────────────────────────────────────────────────────────────────
df = load_data()
df = render_date_filters(df, page_key="ask")

# ── Session state ─────────────────────────────────────────────────────────────
if "ask_history" not in st.session_state:
    st.session_state["ask_history"] = []

for _k in ("ask_groq_key", "ask_gemini_key", "ask_anthropic_key", "ask_provider"):
    if _k not in st.session_state:
        st.session_state[_k] = "" if "key" in _k else "Groq"

# ── Resolve keys (env / secrets / session) ────────────────────────────────────
def _pick(env_name: str, secret_name: str, session_key: str) -> str:
    if os.environ.get(env_name):
        return os.environ[env_name]
    try:
        if secret_name in st.secrets:
            return st.secrets[secret_name]
    except Exception:
        pass
    return st.session_state.get(session_key, "")

def get_keys() -> dict:
    return {
        "Groq":      _pick("GROQ_API_KEY",      "GROQ_API_KEY",      "ask_groq_key"),
        "Gemini":    _pick("GEMINI_API_KEY",     "GEMINI_API_KEY",    "ask_gemini_key"),
        "Anthropic": _pick("ANTHROPIC_API_KEY",  "ANTHROPIC_API_KEY", "ask_anthropic_key"),
    }

# ── Schema context ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_schema(_df: pd.DataFrame) -> str:
    lines = [
        f"DataFrame `df` — {len(_df):,} rows of SATS departure flight data,",
        "Singapore Changi Airport (Terminal 2 and Terminal 3).",
        "",
        "KEY COLUMNS:",
    ]
    priority = {
        "Target_Departure_Delay_Class": "str  — 'On-Time' / 'Acceptable' / 'Delayed'",
        "Target_Departure_Delay_mins":  "float — delay in minutes (negative = early)",
        "identification_carrierCode":   "str  — airline code e.g. 'SQ', 'MI', 'TR'",
        "identification_iata":          "str  — flight number",
        "aircraft_bodyType":            "str  — 'Narrowbody' or 'Widebody'",
        "aircraft_typeICAO":            "str  — ICAO model e.g. 'B738', 'A320', 'B77W'",
        "origin_terminal":              "str  — '2' or '3'",
        "destination_iata":             "str  — destination IATA code",
        "Hour_of_Day":                  "int  — UTC hour 0-23  (Singapore = UTC+8)",
        "Day_of_Week":                  "str  — 'Monday'…'Sunday'",
        "Month":                        "int  — 1-12",
        "Is_Weekend":                   "int  — 1 if Sat/Sun",
        "Incoming_Delay_mins":          "float — how late the inbound aircraft arrived",
        "Available_Ground_Time_mins":   "float — turnaround window",
        "Ground_Time_Ratio":            "float — available / minimum required ground time",
        "Is_Ground_Time_Deficient":     "int  — 1 if available time < minimum required",
        "_dep_date":                    "date  — departure date",
        "_dep_month":                   "int  — month",
        "_dep_year":                    "int  — year",
    }
    for col, desc in priority.items():
        if col in _df.columns:
            lines.append(f"  df['{col}']  →  {desc}")

    delay_cols = [c for c in _df.columns
                  if "_analysis_Delay_mins" in c and "Duration" not in c]
    if delay_cols:
        lines += [
            "",
            f"MILESTONE DELAY COLUMNS ({len(delay_cols)}, float — minutes late):",
            "  Pattern: milestone_{{team}}_{{activity}}_analysis_Delay_mins",
            "  Teams: ramp, pax, aic, cabinSvc, cargo, security, loadControl",
            "  Examples: " + ", ".join(f"'{c}'" for c in delay_cols[:5]),
        ]

    sh = [c for c in _df.columns if c.startswith("specialHandling_")]
    if sh:
        lines += ["", f"SPECIAL HANDLING ({len(sh)} cols, int — pax count per type):"]
        lines.append("  " + ", ".join(f"'{c}'" for c in sh[:6]))

    lines += [
        "",
        "FILTER PATTERNS:",
        "  Delayed:     df[df['Target_Departure_Delay_Class'] == 'Delayed']",
        "  Narrowbody:  df[df['aircraft_bodyType'] == 'Narrowbody']",
        "  Widebody:    df[df['aircraft_bodyType'] == 'Widebody']",
        "  Terminal 2:  df[df['origin_terminal'].astype(str) == '2']",
        "  Terminal 3:  df[df['origin_terminal'].astype(str) == '3']",
        "  Valid delay: df.dropna(subset=['Target_Departure_Delay_mins'])",
        "",
        "DELAY RATE CALCULATION (use this exact pattern):",
        "  rate = (grp['Target_Departure_Delay_Class'] == 'Delayed').mean() * 100",
        "",
        "NARROWBODY ICAO CODES (examples): B738, B737, A320, A321, A319, A20N, B38M",
        "WIDEBODY ICAO CODES (examples):   B77W, B773, A333, A359, B787, B748",
    ]
    return "\n".join(lines)

SCHEMA = build_schema(df)

CODE_SYSTEM = f"""{SCHEMA}

TASK: Write Python code that answers the user's question about this dataset.

Available names (DO NOT redefine):  df, pd, np, go, px

Store results in (set at least one):
  result_text  — str: a plain stat or number
  result_fig   — Plotly Figure
  result_df    — pd.DataFrame: clean summary table (≤30 rows, ≤6 cols)

CHART RULES:
  template="{TEMPLATE}", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
  height 340–480, clear title, labelled axes

CODE RULES:
  • No st.*, print(), display(), open(), or network calls
  • dropna()/fillna() before calculations
  • Check columns: if 'col' in df.columns
  • Clip delay outliers: .clip(upper=120)
  • Terminal compare: df['origin_terminal'].astype(str)
  • Delay rate: (x=='Delayed').sum()/len(x)*100
  • Always set result_df to a tidy summary even when making a chart

Return ONLY valid Python. No markdown fences."""

# ── LLM routing layer ─────────────────────────────────────────────────────────
def _strip_fences(text: str) -> str:
    text = text.strip()
    # Remove opening fence (```python or ``` or ```py)
    text = re.sub(r"^```(?:python|py)?\s*\n?", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # If the first non-empty line is not valid Python (e.g. LLM wrote prose before code),
    # scan forward until we find a line that looks like Python.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        looks_python = bool(
            stripped.startswith(("#", "import ", "from ", "try:", "with ",
                                 "if ", "for ", "def ", "class ", "result_"))
            or re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*(=|\(|\[)', stripped)
        )
        if looks_python:
            return "\n".join(lines[i:]).strip()
        if i >= 5:   # give up scanning after 5 non-Python lines
            break
    return text

def _call_groq(system: str, user: str, api_key: str, max_tokens: int) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content

def _call_gemini(system: str, user: str, api_key: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.1,
        ),
    )
    return resp.text

def _call_anthropic(system: str, user: str, api_key: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text

def llm_call(provider: str, api_key: str,
             system: str, user: str, max_tokens: int = 2000) -> tuple[str, str]:
    """Returns (response_text, error_string). One will be empty."""
    try:
        if provider == "Groq":
            return _call_groq(system, user, api_key, max_tokens), ""
        elif provider == "Gemini":
            return _call_gemini(system, user, api_key, max_tokens), ""
        elif provider == "Anthropic":
            return _call_anthropic(system, user, api_key, max_tokens), ""
        else:
            return "", f"Unknown provider: {provider}"
    except Exception as exc:
        return "", str(exc)

# ── Pass 1: generate code ─────────────────────────────────────────────────────
def generate_code(question: str, provider: str, api_key: str,
                  history: list) -> tuple[str, str]:
    hist = ""
    for item in reversed(history[-3:]):
        hist += f"\nPrevious Q: {item['question']}"
        if item.get("insight"):
            hist += f"\nPrevious answer summary: {item['insight'][:180]}"

    system = CODE_SYSTEM + (f"\n\nCONVERSATION CONTEXT:{hist}" if hist else "")
    raw, err = llm_call(provider, api_key, system, question, max_tokens=2000)
    if err:
        return "", err
    return _strip_fences(raw), ""

# ── Execute generated code ────────────────────────────────────────────────────
def run_code(code: str) -> dict:
    # Catch syntax errors before exec so the message is cleaner
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        return {"text": None, "fig": None, "df_result": None,
                "error": f"SyntaxError: {e}", "code": code}

    ns = {
        "df": df.copy(), "pd": pd, "np": np, "go": go, "px": px,
        "result_text": None, "result_fig": None, "result_df": None,
    }
    try:
        exec(code, ns)  # noqa: S102
        return {"text": ns["result_text"], "fig": ns["result_fig"],
                "df_result": ns["result_df"], "error": None, "code": code}
    except Exception as exc:
        return {"text": None, "fig": None, "df_result": None,
                "error": str(exc), "code": code}


def fix_code(bad_code: str, error: str, provider: str, api_key: str) -> tuple[str, str]:
    """Send failing code + error back to LLM and ask for a corrected version."""
    system = CODE_SYSTEM
    user = f"""The following Python code produced an error. Fix it and return ONLY valid Python.

Error: {error}

Failing code:
{bad_code}

Return ONLY the corrected Python code. No markdown fences, no explanation."""
    raw, err = llm_call(provider, api_key, system, user, max_tokens=2000)
    if err:
        return "", err
    return _strip_fences(raw), ""

# ── Pass 2: interpret actual results ─────────────────────────────────────────
def interpret_results(question: str, result: dict,
                      provider: str, api_key: str) -> str:
    if result.get("error"):
        return ""

    parts = []
    if result.get("text") not in (None, ""):
        parts.append(f"Computed value: {result['text']}")

    df_r = result.get("df_result")
    if df_r is not None:
        try:
            if not df_r.empty:
                d = df_r.head(30).copy()
                for c in d.select_dtypes("float").columns:
                    d[c] = d[c].round(2)
                parts.append(f"Summary table ({len(d)} rows):\n{d.to_string(index=False)}")
        except Exception:
            pass

    fig = result.get("fig")
    if fig is not None and not parts:
        try:
            for trace in fig.data[:3]:
                x, y = getattr(trace, "x", None), getattr(trace, "y", None)
                if x is not None and y is not None and len(x):
                    pairs = list(zip(list(x)[:20],
                                     [round(float(v), 2) if v is not None else None
                                      for v in list(y)[:20]]))
                    parts.append(f"Chart ({getattr(trace,'name','')}) data: {pairs}")
        except Exception:
            pass

    if not parts:
        return ""

    system = (
        "You are a data analyst for SATS Ground Operations, Singapore Changi Airport. "
        "Write clear, direct insights from flight operations data for non-technical staff."
    )
    user = f"""Question: "{question}"

Results from analysing {len(df):,} real departure flights:

{"".join(parts)}

Write a concise answer (3–5 sentences):
1. State the headline finding with the exact number(s)
2. Give the most operationally relevant insight or comparison
3. Note anything surprising or worth acting on
Use plain language. No mention of DataFrames, code, or tables.
Start directly — no "Based on the data..." preamble."""

    text, _ = llm_call(provider, api_key, system, user, max_tokens=450)
    return text.strip()

# ── Intent guard ─────────────────────────────────────────────────────────────
_DATA_KEYWORDS = {
    "delay", "delayed", "on-time", "ontime", "flight", "flights", "carrier",
    "airline", "aircraft", "terminal", "departure", "arrival", "rate", "chart",
    "show", "plot", "compare", "which", "what", "how", "percentage", "percent",
    "average", "avg", "mean", "top", "worst", "best", "breakdown", "hour",
    "day", "month", "week", "year", "trend", "distribution", "count", "total",
    "number", "list", "table", "heatmap", "scatter", "bar", "line", "pie",
    "ground", "turnaround", "milestone", "ramp", "pax", "cargo", "widebody",
    "narrowbody", "icao", "iata", "destination", "incoming", "outgoing",
    "sats", "changi", "singapore", "sq", "mi", "tr", "lw", "lightning",
}

_CHITCHAT_PATTERNS = re.compile(
    r"^(hi+|hello+|hey+|howdy|greetings|good\s*(morning|afternoon|evening|day)|"
    r"what('?s| is) up|how are you|how('?s| is) it going|"
    r"who are you|what (can|do) you do|what('?s| is) this|"
    r"thanks?|thank you|cheers|bye|goodbye|ok|okay|cool|nice|great|"
    r"test|testing|ping|yo|sup)\W*$",
    re.IGNORECASE,
)

def classify_intent(question: str) -> str:
    """
    Returns 'data' or 'chitchat'.
    A question is treated as data-related if it contains at least one data keyword
    OR is long enough to likely be a real question (> 5 words).
    Everything else is chitchat.
    """
    q_lower = question.lower()
    words   = re.findall(r"[a-z]+", q_lower)

    if _CHITCHAT_PATTERNS.match(question.strip()):
        return "chitchat"
    if any(w in _DATA_KEYWORDS for w in words):
        return "data"
    if len(words) > 5:
        return "data"
    return "chitchat"


_CHITCHAT_REPLY = (
    "Hi! I'm the **Ask the Data** assistant for SATS Ground Operations. "
    "I can analyse {n:,} departure flights and answer questions like:\n\n"
    "- *Which carrier has the highest delay rate?*\n"
    "- *Show delays by hour of day*\n"
    "- *Compare Terminal 2 vs Terminal 3*\n"
    "- *What activity causes the most delays?*\n\n"
    "Just type a question about the flight data and I'll run the numbers for you."
)


# ── Suggested questions ────────────────────────────────────────────────────────
SUGGESTIONS = [
    "What % of flights are delayed?",
    "Which carrier has the highest delay rate?",
    "Show delays by hour of day",
    "Compare Terminal 2 vs Terminal 3",
    "Which activity causes the most delays?",
    "Delay rate by month — line chart",
    "Widebody vs narrowbody delay rate",
    "Top 10 destinations by delay rate",
    "How does incoming delay affect departure delay?",
    "Which airline has the best on-time rate?",
    "Show a heatmap of delays by day and hour",
    "Average delay by aircraft type",
]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{header_bg()};
            border:1px solid {header_border()};border-radius:16px;
            padding:24px 32px;margin-bottom:24px;position:relative;overflow:hidden">
  <div style="position:absolute;top:0;right:0;width:280px;height:100%;
              background:radial-gradient(ellipse at right,rgba(77,159,255,0.06),transparent);
              pointer-events:none"></div>
  <h2 style="margin:0;color:{card_text()};font-size:1.6rem">💬 Ask the Data</h2>
  <p style="margin:8px 0 0;color:{card_sub()};font-size:.93rem">
    Ask any question in plain English. The AI analyses all
    <b style="color:#c5d3f0">{len(df):,} flights</b> and responds with
    specific numbers, charts, and insights.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Provider selector + key input ─────────────────────────────────────────────
keys = get_keys()

PROVIDER_INFO = {
    "Groq":      ("🟢 Free",  "console.groq.com → API Keys",           "GROQ_API_KEY"),
    "Gemini":    ("🟢 Free",  "aistudio.google.com → Get API Key",     "GEMINI_API_KEY"),
    "Anthropic": ("💳 Paid",  "console.anthropic.com → API Keys",      "ANTHROPIC_API_KEY"),
}

with st.expander("⚙️  AI Provider & API Key", expanded=not any(keys.values())):
    p_col, info_col = st.columns([1, 2])
    with p_col:
        provider = st.radio(
            "Choose provider",
            list(PROVIDER_INFO.keys()),
            index=list(PROVIDER_INFO.keys()).index(st.session_state["ask_provider"]),
            key="provider_radio",
        )
        st.session_state["ask_provider"] = provider

    badge, hint, env_name = PROVIDER_INFO[provider]
    with info_col:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:14px 18px">
          <div style="font-size:0.85rem;font-weight:700;color:#c5d3f0;margin-bottom:4px">
            {badge} &nbsp; {provider}
          </div>
          <div style="font-size:0.78rem;color:#6b7fa3;margin-bottom:10px">
            Get a free key at <b>{hint}</b>
          </div>
        """, unsafe_allow_html=True)

        session_key = f"ask_{provider.lower()}_key"
        current_key = keys[provider]
        if current_key:
            st.success(f"✅ {provider} key loaded")
        else:
            raw = st.text_input(
                f"{provider} API Key",
                type="password",
                placeholder=f"Paste your {provider} key here",
                label_visibility="collapsed",
                key=f"key_input_{provider}",
            )
            if st.button("Save Key", key=f"save_{provider}"):
                st.session_state[session_key] = raw.strip()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# Refresh keys after potential update
keys = get_keys()
provider = st.session_state["ask_provider"]
api_key  = keys[provider]

if not api_key:
    st.warning(f"Enter a **{provider}** API key above to start asking questions.")
    st.stop()

# ── Suggestions ───────────────────────────────────────────────────────────────
st.markdown(
    "<div style='font-size:0.72rem;font-weight:700;color:#6b7fa3;"
    "text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px'>"
    "Try asking</div>",
    unsafe_allow_html=True,
)
s_cols = st.columns(4)
for i, sug in enumerate(SUGGESTIONS):
    with s_cols[i % 4]:
        if st.button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state["ask_input"] = sug
            st.session_state["ask_auto"] = True
            st.rerun()

st.divider()

# ── Input row ─────────────────────────────────────────────────────────────────
i_col, b_col, c_col = st.columns([5, 0.9, 0.9])
with i_col:
    user_q = st.text_input(
        "q", label_visibility="collapsed",
        placeholder="e.g.  Which carrier has the most delays? Show a bar chart.",
        key="ask_input",
    )
with b_col:
    ask_btn = st.button("Ask →", use_container_width=True, type="primary")
with c_col:
    if st.button("Clear", use_container_width=True) and st.session_state["ask_history"]:
        st.session_state["ask_history"] = []
        st.rerun()

# ── Process ───────────────────────────────────────────────────────────────────
_auto = st.session_state.pop("ask_auto", False)
if (ask_btn or _auto) and user_q.strip():
    question = user_q.strip()

    # ── Intent guard — skip the full pipeline for greetings / chitchat ────────
    if classify_intent(question) == "chitchat":
        chitchat_entry = {
            "question": question,
            "provider": provider,
            "insight":  _CHITCHAT_REPLY.format(n=len(df)),
            "text":     None,
            "fig":      None,
            "df_result": None,
            "error":    None,
            "code":     "",
            "is_chitchat": True,
        }
        st.session_state["ask_history"].insert(0, chitchat_entry)
        st.rerun()

    with st.spinner(f"Generating code via {provider}…"):
        code, err = generate_code(question, provider, api_key,
                                  st.session_state["ask_history"])
    if err:
        st.error(f"Code generation failed: {err}")
    else:
        with st.spinner("Running analysis on full dataset…"):
            result = run_code(code)

        # Auto-retry once if the code had a syntax or runtime error
        if result["error"]:
            with st.spinner("Auto-fixing code error and retrying…"):
                fixed_code, fix_err = fix_code(result["code"], result["error"],
                                               provider, api_key)
            if not fix_err and fixed_code:
                retry = run_code(fixed_code)
                if not retry["error"]:
                    result = retry   # use the fixed result
                    result["code"] = fixed_code

        if not result["error"]:
            with st.spinner("Interpreting results…"):
                result["insight"] = interpret_results(
                    question, result, provider, api_key
                )
        else:
            result["insight"] = ""

        result["question"] = question
        result["provider"] = provider
        st.session_state["ask_history"].insert(0, result)
        st.rerun()

# ── Conversation ──────────────────────────────────────────────────────────────
if not st.session_state["ask_history"]:
    st.markdown("""
    <div style="text-align:center;padding:60px 0 40px">
      <div style="font-size:2.8rem;margin-bottom:14px">💬</div>
      <div style="font-size:1rem;font-weight:600;color:#4a6080">
        Ask anything about the flight operations data
      </div>
      <div style="font-size:0.8rem;color:#6b7fa3;margin-top:6px">
        The AI reads the actual computed numbers before answering — real data, not guesses
      </div>
    </div>
    """, unsafe_allow_html=True)

for i, entry in enumerate(st.session_state["ask_history"]):

    # Question bubble
    prov_badge = entry.get("provider", "")
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin:18px 0 8px">
      <div style="background:rgba(77,159,255,0.12);border:1px solid rgba(77,159,255,0.25);
                  border-radius:16px 16px 4px 16px;padding:12px 20px;max-width:78%">
        <div style="font-size:0.68rem;color:#4d9fff;font-weight:700;
                    text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">You</div>
        <div style="font-size:0.97rem;color:#dde8ff">{entry['question']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div style='font-size:0.68rem;color:#2ecc71;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>"
        f"Assistant &nbsp;<span style='color:#6b7fa3;font-weight:400'>"
        f"via {prov_badge}</span></div>",
        unsafe_allow_html=True,
    )

    if entry.get("error"):
        st.error(f"**Execution error:** {entry['error']}")
    else:
        insight = entry.get("insight", "")
        if insight:
            st.markdown(f"""
            <div style="background:{card_bg()};
                        border:1px solid rgba(46,204,113,0.18);border-left:3px solid #2ecc71;
                        border-radius:0 14px 14px 0;padding:18px 22px;margin-bottom:16px">
              <div style="font-size:0.68rem;font-weight:700;color:#2ecc71;text-transform:uppercase;
                          letter-spacing:1px;margin-bottom:8px">AI Insight</div>
              <div style="font-size:0.97rem;color:{card_text()};line-height:1.7">{insight}</div>
            </div>
            """, unsafe_allow_html=True)

        if entry.get("fig") is not None:
            st.plotly_chart(entry["fig"], use_container_width=True)

        df_r = entry.get("df_result")
        if df_r is not None:
            try:
                if not df_r.empty:
                    st.dataframe(df_r.reset_index(drop=True),
                                 use_container_width=True, hide_index=True)
            except Exception:
                pass

        if not insight and entry.get("text") not in (None, ""):
            st.info(str(entry["text"]))

        if (not insight and entry.get("fig") is None
                and entry.get("df_result") is None
                and entry.get("text") in (None, "")):
            st.warning("No output produced — try rephrasing.")

    with st.expander("View generated code", expanded=False):
        st.code(entry.get("code", ""), language="python")

    if i < len(st.session_state["ask_history"]) - 1:
        st.markdown(
            "<hr style='border-color:rgba(255,255,255,0.04);margin:8px 0'>",
            unsafe_allow_html=True,
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:rgba(77,159,255,0.05);border:1px solid rgba(77,159,255,0.12);
                border-radius:12px;padding:14px 16px;margin-top:16px">
      <div style="font-size:0.7rem;font-weight:700;color:#4d9fff;
                  text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">
        How it works
      </div>
      <div style="font-size:0.76rem;color:#6b7fa3;line-height:1.7">
        1. Your question → AI writes analysis code<br>
        2. Code runs on all <b style="color:#c5d3f0">{len(df):,} flights</b> locally<br>
        3. Result → AI reads the real numbers<br>
        4. AI writes a specific, data-backed answer<br><br>
        <b style="color:#8898b8">Tips:</b><br>
        • Ask for a chart type: "bar chart", "scatter"<br>
        • Follow-ups use prior context<br>
        • Sidebar filters apply to all questions
      </div>
    </div>
    """, unsafe_allow_html=True)
