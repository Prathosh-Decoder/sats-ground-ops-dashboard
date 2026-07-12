"""
utils/style.py
Global CSS for the SATS Ground Operations Dashboard.
Call inject_css() at the top of every page.
"""
import streamlit as st

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.stApp {
    background: #06091a !important;
}
/* Subtle grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(77,159,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(77,159,255,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}
.main .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px !important;
    position: relative;
    z-index: 1;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07091f 0%, #050817 100%) !important;
    border-right: 1px solid rgba(77,159,255,0.10) !important;
}
[data-testid="stSidebar"] * { color: #8898b8 !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c5d3f0 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.4px !important;
    font-weight: 700 !important;
    margin-bottom: 10px !important;
    border-bottom: 1px solid rgba(77,159,255,0.12) !important;
    padding-bottom: 8px !important;
}
/* Keep header transparent — DO NOT use display:none (hides sidebar toggle) */
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}
/* Hide only the deploy/share buttons, NOT the toolbar container itself */
[data-testid="stDecoration"]          { display: none !important; }
[data-testid="stAppDeployButton"]     { display: none !important; }
.stDeployButton                        { display: none !important; }
/* Sidebar expand button — force visible and styled */
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 99999 !important;
    background: rgba(77,159,255,0.2) !important;
    border-radius: 0 8px 8px 0 !important;
    padding: 8px !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #04060f; }
::-webkit-scrollbar-thumb { background: rgba(77,159,255,0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(77,159,255,0.5); }

/* ── KPI / stat cards ────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, rgba(14,21,48,0.9) 0%, rgba(8,12,30,0.9) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 22px 24px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.2s, box-shadow 0.3s;
    cursor: default;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, linear-gradient(90deg,#4d9fff,#00d4ff));
    border-radius: 2px 2px 0 0;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: -40%; right: -20%;
    width: 120px; height: 120px;
    background: radial-gradient(circle, var(--glow, rgba(77,159,255,0.06)) 0%, transparent 70%);
    pointer-events: none;
}
.kpi-card:hover {
    border-color: rgba(77,159,255,0.25);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(77,159,255,0.1);
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #7a90b8;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    margin-bottom: 10px;
}
.kpi-value {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--color, #4d9fff);
    line-height: 1;
    letter-spacing: -1px;
    text-shadow: 0 0 20px var(--glow-color, rgba(77,159,255,0.4));
}
.kpi-sub {
    font-size: 0.7rem;
    color: #5a7898;
    margin-top: 8px;
    font-weight: 500;
}

/* ── Impact / simulator cards ── */
.impact-card {
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    margin-top: 14px;
}
.impact-card-ok {
    background: #0b2e18;
    border: 2px solid #2ecc71;
}
.impact-card-ok .title {
    font-size: .75rem;
    color: #2ecc71 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.impact-card-ok .value {
    font-size: 2.6rem;
    font-weight: 900;
    color: #2ecc71 !important;
    margin: 8px 0;
}
.impact-card-ok .sub {
    font-size: .82rem;
    color: #7dcea0 !important;
}

.impact-card-red {
    background: #2d0808;
    border: 2px solid #e74c3c;
}
.impact-card-red .title {
    font-size: .75rem;
    color: #ff5555 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.impact-card-red .value {
    font-size: 2.6rem;
    font-weight: 900;
    color: #e74c3c !important;
    margin: 8px 0;
}
.impact-card-red .sub {
    font-size: .82rem;
    color: #f1948a !important;
}
.impact-card-red .sub-extra {
    color: #ff8a80 !important;
    font-size: .78rem;
}

.impact-card-amber {
    background: #3a2c08;
    border: 2px solid #f4a621;
}
.impact-card-amber .title {
    font-size: .75rem;
    color: #f4d03f !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.impact-card-amber .value {
    font-size: 2.6rem;
    font-weight: 900;
    color: #f4a621 !important;
    margin: 8px 0;
}
.impact-card-amber .sub {
    font-size: .82rem;
    color: #fad7a0 !important;
}

/* ── Section header ──────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #c5d3f0;
    letter-spacing: 0.2px;
    margin: 0;
}
.section-icon {
    width: 30px; height: 30px;
    background: rgba(77,159,255,0.12);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
}
.section-sub {
    font-size: 0.75rem;
    color: #5a7898;
    margin: 2px 0 18px 40px;
}

/* ── Glass panel ─────────────────────────────────────────────────────────── */
.glass {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
}

/* ── Badges / pills ──────────────────────────────────────────────────────── */
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.pill-green { background:rgba(0,255,136,0.08); color:#00d977; border:1px solid rgba(0,255,136,0.2); }
.pill-red   { background:rgba(255,71,87,0.10); color:#ff5c6c; border:1px solid rgba(255,71,87,0.25); }
.pill-amber { background:rgba(255,180,0,0.08); color:#ffc107; border:1px solid rgba(255,180,0,0.20); }
.pill-blue  { background:rgba(77,159,255,0.08); color:#4d9fff; border:1px solid rgba(77,159,255,0.20); }

/* ── Streamlit widget overrides ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,rgba(14,21,48,0.9),rgba(8,12,30,0.9)) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: #7a90b8 !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: rgba(77,159,255,0.08) !important;
    color: #4d9fff !important;
    border: 1px solid rgba(77,159,255,0.25) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(77,159,255,0.18) !important;
    border-color: rgba(77,159,255,0.5) !important;
    box-shadow: 0 0 18px rgba(77,159,255,0.25) !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: #3d5278 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 22px !important;
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #7a9fd4 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #4d9fff !important;
    background: rgba(77,159,255,0.08) !important;
    border-bottom: 2px solid #4d9fff !important;
}

/* Select / multiselect */
[data-baseweb="select"] > div {
    background: rgba(8,12,30,0.8) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
    background: #4d9fff !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(8,12,30,0.6) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

/* Dataframe */
[data-testid="stDataFrameResizable"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.04) !important; margin: 28px 0 !important; }

/* ── Glow animations ────────────────────────────────────────────────────── */
@keyframes gr  { 0%,100%{box-shadow:0 0 6px #e74c3c}  50%{box-shadow:0 0 28px #e74c3c, 0 0 60px rgba(231,76,60,0.25)} }
@keyframes gg  { 0%,100%{box-shadow:0 0 6px #00d977}  50%{box-shadow:0 0 28px #00d977, 0 0 60px rgba(0,217,119,0.25)} }
@keyframes gb  { 0%,100%{box-shadow:0 0 6px #4d9fff}  50%{box-shadow:0 0 28px #4d9fff, 0 0 60px rgba(77,159,255,0.25)} }
@keyframes glow-pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
@keyframes slide-up { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
@keyframes fade-in  { from{opacity:0} to{opacity:1} }
@keyframes radar-spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }

.glow-red   { animation: gr 1.8s ease-in-out infinite; border-radius:14px; }
.glow-ok    { animation: gg 1.8s ease-in-out infinite; border-radius:14px; }
.glow-blue  { animation: gb 1.8s ease-in-out infinite; border-radius:14px; }
.slide-up   { animation: slide-up 0.5s ease forwards; }
.fade-in    { animation: fade-in  0.6s ease forwards; }

/* ── Live dot ────────────────────────────────────────────────────────────── */
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #00d977;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 8px #00d977;
    animation: glow-pulse 1.5s ease-in-out infinite;
}

/* ── Page title ──────────────────────────────────────────────────────────── */
.page-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #dde8ff;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
}
.page-subtitle {
    font-size: 0.8rem;
    color: #2e3d5a;
    margin-bottom: 24px;
    font-weight: 500;
}
</style>
"""


_LIGHT_MODE_FIX_JS = """
<script>
(function() {
    /* Relative luminance (WCAG formula) */
    function lum(r, g, b) {
        r /= 255; g /= 255; b /= 255;
        r = r <= 0.03928 ? r/12.92 : Math.pow((r+0.055)/1.055, 2.4);
        g = g <= 0.03928 ? g/12.92 : Math.pow((g+0.055)/1.055, 2.4);
        b = b <= 0.03928 ? b/12.92 : Math.pow((b+0.055)/1.055, 2.4);
        return 0.2126*r + 0.7152*g + 0.0722*b;
    }

    /* Fallback token list — catches hex values BEFORE browser normalizes them */
    var ATTR_DARK = ['linear-gradient','#0d1117','#1a1f2e','#06091a','#0a0e1a',
                     '#070b14','#0b2e18','#16213e','#1a3a6e','#06091a',
                     'rgba(8,12,30','rgba(14,21,48'];

    function fixEl(p, el) {
        var s = el.getAttribute('style') || '';
        if (s.indexOf('background') === -1) return;

        var dark = ATTR_DARK.some(function(t){ return s.indexOf(t) !== -1; });

        /* Computed-style fallback: works when browser has already normalised colours */
        if (!dark) {
            try {
                var pwin = p.defaultView || window.parent;
                var cs   = pwin.getComputedStyle(el);
                var bgi  = cs.backgroundImage;
                /* Any gradient background */
                if (bgi && bgi !== 'none' && bgi.indexOf('gradient') !== -1) { dark = true; }
                /* Dark solid background */
                if (!dark) {
                    var m = cs.backgroundColor.match(/rgba?\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)/);
                    if (m && lum(+m[1],+m[2],+m[3]) < 0.15) { dark = true; }
                }
            } catch(e) {}
        }

        if (dark) {
            el.style.setProperty('background',       '#f4f6fb', 'important');
            el.style.setProperty('background-image', 'none',    'important');
        }
    }

    function run() {
        try {
            var p = window.parent.document;
            p.querySelectorAll('[data-testid="stMarkdownContainer"] [style]')
             .forEach(function(el){ fixEl(p, el); });
        } catch(e) {}
    }

    run();
    setTimeout(run, 300);
    setTimeout(run, 800);
    setTimeout(run, 2000);
})();
</script>
"""

_OPEN_SIDEBAR_JS = """
<script>
(function() {
    function tryOpen(attempts) {
        var p = window.parent.document;
        // If sidebar is already open (collapse button present inside sidebar), do nothing
        var closeBtn = p.querySelector('[data-testid="stSidebarCollapseButton"]');
        if (closeBtn) return;
        // Click the expand button (works even if CSS hides it)
        var expandBtn = p.querySelector('[data-testid="stSidebarCollapsedControl"] button');
        if (expandBtn) { expandBtn.click(); return; }
        // Retry up to 8 times with 250ms gaps
        if (attempts > 0) setTimeout(function(){ tryOpen(attempts - 1); }, 250);
    }
    setTimeout(function(){ tryOpen(8); }, 300);
})();
</script>
"""

LIGHT_CSS = """
<style>
/* ═══════════════════════════════ LIGHT MODE OVERRIDES ════════════════════════ */

/* ── App background ── */
.stApp { background: #f0f3fa !important; }
.stApp::before {
    background-image:
        linear-gradient(rgba(26,115,232,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(26,115,232,0.03) 1px, transparent 1px) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eaedfa 0%, #e2e6f5 100%) !important;
    border-right: 1px solid rgba(26,115,232,0.14) !important;
}
[data-testid="stSidebar"] * { color: #2c3e6b !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1a2340 !important;
    border-bottom: 1px solid rgba(26,115,232,0.15) !important;
}
[data-testid="stSidebarCollapsedControl"] { background: rgba(26,115,232,0.15) !important; }

/* ── BLANKET RULE: all custom HTML text → dark ──────────────────
   Targets every inline element inside unsafe_allow_html blocks.
   More reliable than [style*="color:#xxx"] attribute selectors.
   Accent colours are restored below with higher-specificity rules.
   ─────────────────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] b,
[data-testid="stMarkdownContainer"] small,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] a { color: #1a2340 !important; }

/* ── Restore accent colours (attribute selector has higher specificity) ── */
[data-testid="stMarkdownContainer"] [style*="color:#e74c3c"] { color: #c0392b !important; }
[data-testid="stMarkdownContainer"] [style*="color:#ff4757"] { color: #c0392b !important; }
[data-testid="stMarkdownContainer"] [style*="color:#f44336"] { color: #c0392b !important; }
[data-testid="stMarkdownContainer"] [style*="color:#2ecc71"] { color: #1a8a50 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#00d977"] { color: #1a8a50 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#f39c12"] { color: #b7770d !important; }
[data-testid="stMarkdownContainer"] [style*="color:#ffc107"] { color: #b7770d !important; }
[data-testid="stMarkdownContainer"] [style*="color:#1a73e8"] { color: #1a73e8 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#4d9fff"] { color: #1565c0 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#9c27b0"] { color: #6a1b9a !important; }
[data-testid="stMarkdownContainer"] [style*="color:#00bcd4"] { color: #00838f !important; }
[data-testid="stMarkdownContainer"] [style*="color:#795548"] { color: #5d4037 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#4caf50"] { color: #2e7d32 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#ff9800"] { color: #e65100 !important; }
[data-testid="stMarkdownContainer"] [style*="color:#e91e63"] { color: #880e4f !important; }
[data-testid="stMarkdownContainer"] [style*="color:#607d8b"] { color: #37474f !important; }
[data-testid="stMarkdownContainer"] [style*="color:#f4a621"] { color: #b7770d !important; }

/* ── Card backgrounds ──
   Broad selectors: browsers preserve function names like "linear-gradient"
   but may normalize hex colours to rgb(). Catch-all on the keyword is safest. */
[data-testid="stMarkdownContainer"] [style*="linear-gradient"],
[data-testid="stMarkdownContainer"] [style*="background:#1a1f2e"],
[data-testid="stMarkdownContainer"] [style*="background: #1a1f2e"],
[data-testid="stMarkdownContainer"] [style*="background:#0d1117"],
[data-testid="stMarkdownContainer"] [style*="background: #0d1117"],
[data-testid="stMarkdownContainer"] [style*="background:#06091a"],
[data-testid="stMarkdownContainer"] [style*="background: #06091a"],
[data-testid="stMarkdownContainer"] [style*="background:#0a0e1a"],
[data-testid="stMarkdownContainer"] [style*="background:rgba(8,12,30"],
[data-testid="stMarkdownContainer"] [style*="background: rgba(8,12,30"],
[data-testid="stMarkdownContainer"] [style*="background:rgba(14,21,48"],
[data-testid="stMarkdownContainer"] [style*="background: rgba(14,21,48"] {
    background: #f8f9fd !important;
    background-image: none !important;
}

/* ── Native Streamlit widgets ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid rgba(26,115,232,0.12) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] { color: #4a5a7a !important; }
[data-testid="stMetricValue"] { color: #1a2340 !important; }

.stApp .stMarkdown, .stApp .stMarkdown p, .stApp .stMarkdown li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #1a2340 !important; }
.stApp .stCaption,
.stApp [data-testid="stCaptionContainer"] p { color: #5a7090 !important; }

.stButton > button {
    background: rgba(26,115,232,0.07) !important;
    color: #1a73e8 !important;
    border: 1px solid rgba(26,115,232,0.25) !important;
}
.stButton > button:hover {
    background: rgba(26,115,232,0.15) !important;
    border-color: rgba(26,115,232,0.45) !important;
}
[data-testid="baseButton-primary"] {
    background: #1a73e8 !important; color: #ffffff !important; border-color: #1a73e8 !important;
}

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid rgba(0,0,0,0.10) !important; }
[data-testid="stTabs"] [role="tab"] { color: #4a5a7a !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: #1a73e8 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1a73e8 !important;
    background: rgba(26,115,232,0.07) !important;
    border-bottom: 2px solid #1a73e8 !important;
}

[data-baseweb="select"] > div {
    background: #ffffff !important; border-color: rgba(0,0,0,0.12) !important; color: #1a2340 !important;
}
.stTextInput input, .stNumberInput input {
    background: #ffffff !important; border-color: rgba(0,0,0,0.12) !important; color: #1a2340 !important;
}

[data-testid="stExpander"] {
    background: #ffffff !important; border: 1px solid rgba(26,115,232,0.12) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p { color: #1a2340 !important; }

[data-testid="stDataFrameResizable"] {
    border: 1px solid rgba(0,0,0,0.08) !important; background: #ffffff !important;
}

::-webkit-scrollbar-track { background: #eef0f8 !important; }
::-webkit-scrollbar-thumb { background: rgba(26,115,232,0.25) !important; }
hr { border-color: rgba(0,0,0,0.09) !important; }
.stAlert { background: #ffffff !important; }

/* ── Plotly SVG chart text ──
   Plotly renders tick labels, axis titles, and legend text as SVG <text>
   elements. .g-xtitle / .g-ytitle hold the axis title; .xtick / .ytick hold
   tick labels. Without this, they inherit `color: white` and vanish on light bg. */
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.js-plotly-plot .g-xtitle text,
.js-plotly-plot .g-ytitle text,
.js-plotly-plot .legendtext,
.js-plotly-plot .gtitle,
.js-plotly-plot .g-title text,
.js-plotly-plot .colorbar text,
.js-plotly-plot .trace text,
.js-plotly-plot text { fill: #2a3f5f !important; }

/* ── Custom CSS classes ── */
.kpi-card {
    background: #ffffff !important;
    border: 1px solid rgba(26,115,232,0.12) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
.kpi-card:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important; }
.kpi-label { color: #4a5a7a !important; }
.kpi-value { text-shadow: none !important; }
.kpi-sub   { color: #6b7fa3 !important; }
.section-title, .page-title { color: #1a2340 !important; }
.section-sub, .page-subtitle { color: #5a7090 !important; }
.glass {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(26,115,232,0.12) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06) !important;
}
.pill-green { background:rgba(0,180,90,0.10); color:#1a8a50 !important; border:1px solid rgba(0,180,90,0.25); }
.pill-red   { background:rgba(220,50,50,0.08); color:#c0392b !important; border:1px solid rgba(220,50,50,0.2); }
.pill-amber { background:rgba(200,140,0,0.10); color:#9a6700 !important; border:1px solid rgba(200,140,0,0.25); }
.pill-blue  { background:rgba(26,115,232,0.08); color:#1a73e8 !important; border:1px solid rgba(26,115,232,0.2); }

/* ── Impact / simulator cards light mode overrides ── */
.stApp .impact-card-ok {
    background: #eefbf3 !important;
    border-color: #2ecc71 !important;
}
.stApp .impact-card-ok .title,
.stApp .impact-card-ok .value {
    color: #1a8a50 !important;
}
.stApp .impact-card-ok .sub {
    color: #27ae60 !important;
}

.stApp .impact-card-red {
    background: #fdf2f2 !important;
    border-color: #e74c3c !important;
}
.stApp .impact-card-red .title,
.stApp .impact-card-red .value {
    color: #c0392b !important;
}
.stApp .impact-card-red .sub {
    color: #e74c3c !important;
}
.stApp .impact-card-red .sub-extra {
    color: #b03a2e !important;
}

.stApp .impact-card-amber {
    background: #fef9e7 !important;
    border-color: #f4a621 !important;
}
.stApp .impact-card-amber .title {
    color: #b7770d !important;
}
.stApp .impact-card-amber .value {
    color: #d35400 !important;
}
.stApp .impact-card-amber .sub {
    color: #a04000 !important;
}
</style>
"""


def is_light() -> bool:
    """True when the user has toggled light mode."""
    return st.session_state.get("_theme") == "light"


def chart_template() -> str:
    return "plotly_white" if is_light() else "plotly_dark"


def chart_fc() -> str:
    """Foreground/label colour for Plotly text."""
    return "#2c3e6b" if is_light() else "#c5d3f0"


def chart_gc() -> str:
    """Grid-line colour for Plotly axes."""
    return "rgba(0,0,0,0.08)" if is_light() else "rgba(255,255,255,0.04)"


# ── Python-side theme helpers for inline HTML ─────────────────────────────────

def card_bg() -> str:
    """Background for dark data/info cards."""
    return "#f0f4fb" if is_light() else "#1a1f2e"

def card_text() -> str:
    """Primary text colour on dark cards."""
    return "#1a2340" if is_light() else "#ffffff"

def card_sub() -> str:
    """Secondary/muted text colour on dark cards."""
    return "#4a5a7a" if is_light() else "#8892a4"

def header_bg() -> str:
    """Page-section gradient header → solid light card in light mode."""
    return "#e8edf8" if is_light() else "linear-gradient(135deg,#0d1117 0%,#1a1f2e 50%,#0d1117 100%)"

def header_border() -> str:
    """Border colour for page-section header boxes."""
    return "rgba(26,115,232,0.25)" if is_light() else "rgba(77,159,255,0.2)"


def inject_css():
    import streamlit.components.v1 as components  # noqa: PLC0415
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    if st.session_state.get("_theme") == "light":
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
        components.html(_LIGHT_MODE_FIX_JS, height=0)

    # Theme toggle — rendered in sidebar on every page
    with st.sidebar:
        _is_light = st.session_state.get("_theme") == "light"
        _toggled  = st.toggle("☀️  Light Mode", value=_is_light, key="_theme_toggle")
        if _toggled != _is_light:
            st.session_state["_theme"] = "light" if _toggled else "dark"
            st.rerun()
        st.divider()

    # Force sidebar open on every page load via a zero-height iframe
    components.html(_OPEN_SIDEBAR_JS, height=0)


def page_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin-bottom:24px">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
            <div style="font-size:1.4rem;background:rgba(77,159,255,0.1);
                        border-radius:10px;width:42px;height:42px;
                        display:flex;align-items:center;justify-content:center">{icon}</div>
            <div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", color: str = "#4d9fff",
             glow: str = "rgba(77,159,255,0.4)"):
    return f"""
    <div class="kpi-card" style="--accent:{color};--color:{color};
                                 --glow-color:{glow};--glow:rgba(77,159,255,0.06)">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
    </div>"""


def section_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="section-header" style="margin-top:8px">
        <div class="section-icon">{icon}</div>
        <div class="section-title">{title}</div>
    </div>
    {"" if not subtitle else f'<div class="section-sub">{subtitle}</div>'}
    """, unsafe_allow_html=True)
