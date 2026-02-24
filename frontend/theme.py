"""
OmniSearch AI — Unified Premium Theme
Dark mode with purple/violet accents, glassmorphism cards, smooth animations.
Import and call inject_theme() at the top of every page.
"""
import streamlit as st

# ── Plotly dark template (matches theme) ──
try:
    import plotly.io as pio
    pio.templates["omnisearch_dark"] = pio.templates["plotly_dark"]
    pio.templates["omnisearch_dark"].layout.update(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8EAED",
        colorway=["#6C63FF", "#8B83FF", "#2DD4A0", "#F0B429", "#EF4444",
                  "#38BDF8", "#A78BFA", "#FB923C", "#34D399", "#F472B6"],
    )
    pio.templates.default = "omnisearch_dark"
except ImportError:
    pass  # Plotly not installed — skip

# Color tokens
ACCENT = "#6C63FF"
ACCENT_LIGHT = "#8B83FF"
ACCENT_DARK = "#5A52E0"
BG_PRIMARY = "#0E1117"
BG_CARD = "#161B22"
BG_CARD_HOVER = "#1C2230"
TEXT_PRIMARY = "#E8EAED"
TEXT_SECONDARY = "#9AA0A6"
SUCCESS = "#2DD4A0"
WARNING = "#F0B429"
ERROR = "#EF4444"
BORDER = "rgba(108, 99, 255, 0.15)"
GLOW = "rgba(108, 99, 255, 0.35)"

GLOBAL_CSS = """
<style>
/* ===== IMPORTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== GLOBAL ===== */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background: #0E1117 !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid rgba(108, 99, 255, 0.12);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #E8EAED !important;
}

/* ===== PAGE HEADER ===== */
.page-header {
    background: linear-gradient(135deg, #1A1040 0%, #2D1B69 40%, #1E1145 100%);
    padding: 2.5rem 3rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    border: 1px solid rgba(108, 99, 255, 0.2);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.12);
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(108, 99, 255, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.page-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.page-header p {
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
    opacity: 0.8;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* ===== GLASS CARDS ===== */
.glass-card {
    background: rgba(22, 27, 34, 0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(108, 99, 255, 0.12);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.75rem 0;
    transition: all 0.25s ease;
}
.glass-card:hover {
    border-color: rgba(108, 99, 255, 0.3);
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.08);
    transform: translateY(-1px);
}

/* ===== STEP CARDS (pipeline steps) ===== */
.step-card {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.08) 0%, rgba(108, 99, 255, 0.02) 100%);
    border: 1px solid rgba(108, 99, 255, 0.18);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.step-card:hover {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.15) 0%, rgba(108, 99, 255, 0.05) 100%);
    border-color: rgba(108, 99, 255, 0.4);
    box-shadow: 0 8px 24px rgba(108, 99, 255, 0.15);
    transform: translateY(-3px);
}
.step-card .step-icon {
    font-size: 1.8rem;
    display: block;
    margin-bottom: 0.4rem;
}
.step-card .step-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #E8EAED;
    margin: 0;
}
.step-card .step-desc {
    font-size: 0.8rem;
    color: #9AA0A6;
    margin: 0.25rem 0 0 0;
}

/* ===== INFO CARDS ===== */
.info-card {
    background: #161B22;
    border: 1px solid rgba(108, 99, 255, 0.12);
    border-left: 3px solid #6C63FF;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.info-card h4 {
    color: #E8EAED;
    margin: 0 0 0.75rem 0;
    font-weight: 700;
    font-size: 1.1rem;
}
.info-card p, .info-card li {
    color: #9AA0A6;
    line-height: 1.6;
    margin: 0.25rem 0;
}
.info-card strong {
    color: #B8B3FF;
}

/* ===== SUCCESS / ERROR CARDS ===== */
.success-card {
    background: linear-gradient(135deg, rgba(45, 212, 160, 0.1) 0%, rgba(45, 212, 160, 0.03) 100%);
    border: 1px solid rgba(45, 212, 160, 0.25);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #E8EAED;
}
.success-card .success-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.success-card h3 { color: #2DD4A0; margin: 0.5rem 0; font-weight: 700; }
.success-card p { color: #9AA0A6; }

.error-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.03) 100%);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #E8EAED;
}
.error-card h3 { color: #EF4444; }

/* ===== METRICS ===== */
div[data-testid="stMetric"] {
    background: #161B22 !important;
    border: 1px solid rgba(108, 99, 255, 0.12);
    border-radius: 10px;
    padding: 1rem 1.25rem !important;
}
div[data-testid="stMetric"] label {
    color: #9AA0A6 !important;
    font-weight: 600;
    font-size: 0.85rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #E8EAED !important;
    font-weight: 700;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #5A52E0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 8px rgba(108, 99, 255, 0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8B83FF 0%, #6C63FF 100%) !important;
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.35) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ===== DOWNLOAD BUTTON ===== */
.stDownloadButton > button {
    background: linear-gradient(135deg, #2DD4A0 0%, #22B885 100%) !important;
    color: #0E1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(45, 212, 160, 0.25) !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 6px 20px rgba(45, 212, 160, 0.35) !important;
}

/* ===== INPUTS ===== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #1A1D23 !important;
    border: 1px solid rgba(108, 99, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #E8EAED !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.15) !important;
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #1A1D23 !important;
    border: 1px solid rgba(108, 99, 255, 0.15) !important;
    border-radius: 8px !important;
}

/* ===== DATAFRAMES & TABLES ===== */
.stDataFrame {
    border: 1px solid rgba(108, 99, 255, 0.12) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
/* Force dark theme on styled pandas tables */
.stDataFrame table,
.stDataFrame thead,
.stDataFrame tbody,
.stDataFrame tr {
    background-color: #161B22 !important;
}
.stDataFrame th {
    background-color: #1A1D23 !important;
    color: #B8B3FF !important;
    font-weight: 700 !important;
    border-bottom: 2px solid rgba(108, 99, 255, 0.2) !important;
    padding: 0.6rem 0.8rem !important;
    font-size: 0.85rem !important;
}
.stDataFrame td {
    color: #E8EAED !important;
    border-bottom: 1px solid rgba(108, 99, 255, 0.08) !important;
    padding: 0.5rem 0.8rem !important;
    font-size: 0.84rem !important;
}
.stDataFrame tr:hover td {
    background-color: rgba(108, 99, 255, 0.06) !important;
}
/* Styled dataframe (from df.style) */
table.dataframe {
    background-color: #161B22 !important;
    color: #E8EAED !important;
    border-collapse: collapse;
    width: 100%;
}
table.dataframe th {
    background-color: #1A1D23 !important;
    color: #B8B3FF !important;
    font-weight: 700 !important;
}
table.dataframe td {
    color: #E8EAED !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161B22;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    color: #9AA0A6;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(108, 99, 255, 0.15) !important;
    color: #B8B3FF !important;
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    background: #161B22 !important;
    border: 1px solid rgba(108, 99, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #E8EAED !important;
    font-weight: 600;
}

/* ===== FILE UPLOADER ===== */
.stFileUploader > div > div {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(108, 99, 255, 0.04) !important;
    border: 2px dashed rgba(108, 99, 255, 0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(108, 99, 255, 0.5) !important;
    background: rgba(108, 99, 255, 0.08) !important;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6C63FF, #8B83FF) !important;
}

/* ===== DIVIDER ===== */
hr {
    border-color: rgba(108, 99, 255, 0.1) !important;
    margin: 1.5rem 0;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0E1117; }
::-webkit-scrollbar-thumb { background: rgba(108, 99, 255, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(108, 99, 255, 0.5); }

/* ===== ALERTS ===== */
.stAlert {
    border-radius: 10px !important;
}

/* ===== CAPTION / FOOTER ===== */
.stCaption {
    text-align: center;
    color: #6B7280 !important;
}

/* ===== SIDEBAR NAV LINKS ===== */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    color: #9AA0A6 !important;
    border-radius: 8px;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
    background: rgba(108, 99, 255, 0.1) !important;
    color: #E8EAED !important;
}

/* ===== UPLOAD ZONE (custom) ===== */
.upload-zone {
    background: rgba(108, 99, 255, 0.04);
    padding: 3rem;
    border-radius: 16px;
    border: 2px dashed rgba(108, 99, 255, 0.25);
    text-align: center;
    margin: 1.5rem 0;
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: rgba(108, 99, 255, 0.5);
    background: rgba(108, 99, 255, 0.08);
    box-shadow: 0 8px 30px rgba(108, 99, 255, 0.1);
}
.upload-zone .upload-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.upload-zone .upload-text { font-size: 1.1rem; color: #E8EAED; font-weight: 600; margin: 0.5rem 0; }
.upload-zone .upload-subtext { font-size: 0.85rem; color: #9AA0A6; }

/* ===== NEXT STEPS ===== */
.next-steps {
    background: rgba(108, 99, 255, 0.06);
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: 12px;
    padding: 2rem;
    margin: 1.5rem 0;
}
.next-steps h3 { color: #E8EAED; margin-top: 0; font-weight: 700; }
.next-steps .step-item {
    background: rgba(108, 99, 255, 0.06);
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid rgba(108, 99, 255, 0.3);
    color: #9AA0A6;
}
.next-steps .step-item strong { color: #B8B3FF; }

/* ===== DATASET INFO ===== */
.dataset-info {
    background: #161B22;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(108, 99, 255, 0.12);
    border-left: 3px solid #6C63FF;
    margin: 1rem 0;
}
.dataset-info .info-title { color: #B8B3FF; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.75rem; }
.dataset-info .info-metric {
    display: inline-block;
    background: rgba(108, 99, 255, 0.08);
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    margin: 0.3rem;
    font-weight: 600;
    color: #E8EAED;
    font-size: 0.85rem;
    border: 1px solid rgba(108, 99, 255, 0.12);
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.page-header, .glass-card, .step-card, .info-card {
    animation: fadeIn 0.4s ease-out;
}
</style>
"""


def inject_theme():
    """Inject the global theme CSS into the current page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str = ""):
    """Render a consistent page header."""
    sub_html = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="page-header">
        <h1>{icon} {title}</h1>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def page_footer():
    """Render a consistent footer."""
    st.markdown("---")
    st.caption("© 2025 OmniSearch AI • Industrial ML Platform • Built with FastAPI & Streamlit")
