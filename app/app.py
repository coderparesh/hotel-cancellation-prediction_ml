"""
HotelGuard AI — Predict. Prevent. Optimize.

A production-quality hotel cancellation prediction system built for
revenue-management teams. Elegant hospitality-focused interface with
real ML predictions and interactive analytics.

Refactored following the 40 Frontend Engineering Skills:
- Reusable components (Skill 4)
- Loading / Error / Empty states (Skills 16-18)
- Form validation (Skill 5)
- Chart consistency (Skill 23)
- Responsive design (Skill 15)
- Accessibility (Skill 24)
- Filter enhancements (Skill 20)
- Data tables (Skill 6)
- Dashboard widgets (Skill 22)
- Batch prediction (Skills 2, 21)
- Booking comparison (Skill 2)
- Navigation enhancement (Skill 13)
- Consistency across pages (Skill 35)
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    MODEL_PATH, MODEL_METADATA_PATH, RAW_DATA_FILE,
    ORDINAL_MONTH_MAP,
)

# ═══════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HotelGuard AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# Design Tokens  (Skill 14 — reuse existing theme)
# ═══════════════════════════════════════════════════════════════════
CHARCOAL = "#202522"
IVORY = "#F7F4ED"
CHAMPAGNE = "#C5A46D"
FOREST = "#315C4B"
SAGE = "#DCE5DD"
BEIGE = "#E8E0D2"
CORAL = "#B85C5C"
AMBER = "#C08A45"
SUCCESS = "#4F7C63"

# Chart colors — muted, professional
CHART_PALETTE = [FOREST, CHAMPAGNE, CORAL, AMBER, SUCCESS, "#7A8B7E", BEIGE]

# ═══════════════════════════════════════════════════════════════════
# Cached Loaders  (Skill 26 — data fetching)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    import joblib
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_metadata():
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset():
    if RAW_DATA_FILE.exists():
        return pd.read_csv(RAW_DATA_FILE)
    return None


@st.cache_data
def get_hero_b64():
    p = Path(__file__).parent / "assets" / "hero_bg.jpg"
    if p.exists():
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


model = load_model()
metadata = load_metadata()
df_raw = load_dataset()
hero_b64 = get_hero_b64()

# Reverse month map for display
MONTH_NAMES = list(ORDINAL_MONTH_MAP.keys())


# ═══════════════════════════════════════════════════════════════════
# Reusable Components  (Skills 4, 33 — DRY, focused, composable)
# ═══════════════════════════════════════════════════════════════════

def render_section_header(title: str, subtitle: str = ""):
    """Reusable section heading — consistent typography across all pages."""
    sub_html = f'<div class="hg-section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="hg-section" style="margin-top:1.5rem;">
        <div class="hg-section-title">{title}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(value: str, label: str, trend: str = "", font_size: str = "2rem"):
    """Reusable KPI metric card with optional trend indicator (Skill 22)."""
    trend_html = ""
    if trend:
        is_up = trend.startswith("+") or trend.startswith("↑")
        color = SUCCESS if is_up else CORAL
        trend_html = f'<div style="font-size:0.78rem; color:{color}; margin-top:2px;">{trend}</div>'
    st.markdown(f"""
    <div class="hg-kpi" role="status" aria-label="{label}: {value}">
        <div class="hg-kpi-value" style="font-size:{font_size};">{value}</div>
        <div class="hg-kpi-label">{label}</div>
        {trend_html}
    </div>
    """, unsafe_allow_html=True)


def render_info_card(content: str, border_left_color: str = ""):
    """Reusable info/recommendation card."""
    border = f"border-left:3px solid {border_left_color};" if border_left_color else ""
    st.markdown(f"""
    <div class="hg-card" style="padding:1rem 1.5rem; {border}">
        <p style="margin:0; font-size:0.92rem;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_divider():
    """Reusable visual separator."""
    st.markdown('<div class="hg-divider"></div>', unsafe_allow_html=True)


def render_loading_state(message: str = "Loading data…"):
    """Skeleton-like loading placeholder (Skill 16)."""
    st.markdown(f"""
    <div class="hg-card" style="text-align:center; padding:3rem;" role="status" aria-live="polite">
        <div class="hg-loading-pulse" style="margin-bottom:1rem;">
            <div style="width:60px;height:60px;border-radius:50%;
                border:3px solid {BEIGE};border-top-color:{FOREST};
                animation:hg-spin 0.8s linear infinite;margin:0 auto;"></div>
        </div>
        <div style="font-size:0.95rem; color:#6B7280;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_error_state(title: str, message: str, show_retry: bool = False):
    """Structured error display — never show stack traces to users (Skill 17)."""
    retry_html = ""
    if show_retry:
        retry_html = """
        <p style="color:#9CA3AF; font-size:0.82rem; margin-top:1rem; margin-bottom:0;">
            Try refreshing the page or check your configuration.
        </p>"""
    st.markdown(f"""
    <div class="hg-card" style="text-align:center; padding:2.5rem; border-left:3px solid {CORAL};"
         role="alert" aria-live="assertive">
        <div style="font-size:1.6rem; margin-bottom:0.5rem;">⚠</div>
        <div style="font-size:1.05rem; font-weight:600; color:{CHARCOAL}; margin-bottom:0.5rem;">
            {title}
        </div>
        <p style="color:#6B7280; margin:0;">{message}</p>
        {retry_html}
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(title: str, description: str, cta_text: str = ""):
    """Informative empty state — explains what, why, and what to do next (Skill 18)."""
    cta_html = ""
    if cta_text:
        cta_html = f"""
        <div style="margin-top:1rem;">
            <span class="hg-btn hg-btn-primary" style="font-size:0.85rem; padding:8px 20px;">
                {cta_text}
            </span>
        </div>"""
    st.markdown(f"""
    <div class="hg-card" style="text-align:center; padding:3rem;" role="status">
        <div style="font-size:1.6rem; margin-bottom:0.5rem;">📋</div>
        <div style="font-size:1.05rem; font-weight:600; color:{CHARCOAL}; margin-bottom:0.5rem;">
            {title}
        </div>
        <p style="color:#6B7280; margin:0; max-width:400px; margin-left:auto; margin-right:auto;">
            {description}
        </p>
        {cta_html}
    </div>
    """, unsafe_allow_html=True)


def render_metric_row(name: str, value: str):
    """Single metric key-value row — used in feature importance, details."""
    st.markdown(f"""
    <div class="hg-metric-row">
        <span class="hg-metric-name">{name}</span>
        <span class="hg-metric-val">{value}</span>
    </div>
    """, unsafe_allow_html=True)


def render_flow_steps(steps: list, container_label: str = ""):
    """Reusable vertical flow/pipeline visualization."""
    label_html = (
        f'<div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase; '
        f'color:#9CA3AF; margin-bottom:1rem;">{container_label}</div>'
        if container_label else ""
    )
    steps_html = "".join([
        f'<div class="hg-flow-step"><div class="hg-flow-dot"></div><div class="hg-flow-text">{step}</div></div>'
        + ('<div class="hg-flow-line"></div>' if i < len(steps) - 1 else '')
        for i, step in enumerate(steps)
    ])
    st.markdown(
        f'<div class="hg-card" style="padding:1.5rem;">{label_html}{steps_html}</div>',
        unsafe_allow_html=True,
    )


def render_result_card(cancel_pct: float):
    """Prediction result display with risk classification."""
    if cancel_pct < 30:
        risk_level, risk_class, badge_class = "LOW RISK", "hg-result-low", "hg-badge-low"
    elif cancel_pct < 60:
        risk_level, risk_class, badge_class = "MEDIUM RISK", "hg-result-med", "hg-badge-med"
    else:
        risk_level, risk_class, badge_class = "HIGH RISK", "hg-result-high", "hg-badge-high"

    st.markdown(f"""
    <div class="hg-result {risk_class}" role="status" aria-live="polite"
         aria-label="Cancellation risk: {risk_level}, {cancel_pct:.1f} percent">
        <div class="hg-result-label">Cancellation Assessment</div>
        <div class="hg-result-badge {badge_class}">{risk_level}</div>
        <div class="hg-result-pct">{cancel_pct:.1f}%</div>
        <div class="hg-result-desc">Estimated Cancellation Probability</div>
    </div>
    """, unsafe_allow_html=True)
    return risk_level, cancel_pct


def engineer_features(row: dict) -> dict:
    """Extract feature engineering into reusable function (Skill 33 — avoid duplication).
    Used by both single and batch prediction."""
    total_stay = row.get("stays_in_weekend_nights", 0) + row.get("stays_in_week_nights", 0)
    total_guests = row.get("adults", 0) + row.get("children", 0) + row.get("babies", 0)
    is_local = 1 if str(row.get("country", "")).upper() == "PRT" else 0
    adr_per_person = row.get("adr", 0) / total_guests if total_guests > 0 else 0.0
    is_same_room = 1 if row.get("reserved_room_type") == row.get("assigned_room_type") else 0
    booking_changes_flag = 1 if row.get("booking_changes", 0) > 0 else 0
    is_family = 1 if (row.get("children", 0) > 0 or row.get("babies", 0) > 0) else 0
    arrival_month = row.get("arrival_date_month", 1)
    arrival_month_sin = np.sin(2 * np.pi * arrival_month / 12)
    arrival_month_cos = np.cos(2 * np.pi * arrival_month / 12)
    total_previous = row.get("previous_cancellations", 0) + row.get("previous_bookings_not_canceled", 0)
    cancellation_ratio = row.get("previous_cancellations", 0) / total_previous if total_previous > 0 else 0.0

    return {
        "total_stay": total_stay,
        "total_guests": total_guests,
        "is_local": is_local,
        "adr_per_person": adr_per_person,
        "is_same_room": is_same_room,
        "booking_changes_flag": booking_changes_flag,
        "is_family": is_family,
        "arrival_month_sin": arrival_month_sin,
        "arrival_month_cos": arrival_month_cos,
        "total_previous": total_previous,
        "cancellation_ratio": cancellation_ratio,
    }


# ═══════════════════════════════════════════════════════════════════
# Centralized CSS Theme  (Skills 14, 15, 24 — theme, responsive, a11y)
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Reset ── */
    *, html, body, [class*="css"] {{
        font-family: 'Inter', Arial, sans-serif !important;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 0 !important; max-width: 1200px; }}

    /* ── App background ── */
    .stApp {{
        background-color: {IVORY};
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {CHARCOAL} !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {IVORY} !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: {BEIGE} !important;
        font-weight: 400 !important;
        font-size: 0.95rem !important;
    }}
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
        font-size: 0.95rem !important;
    }}

    /* ── Typography — force dark text on light background ── */
    h1, h2, h3, h4, h5, h6 {{
        color: {CHARCOAL} !important;
        font-weight: 600;
    }}
    h2 {{ font-size: 1.7rem; }}
    h3 {{ font-size: 1.25rem; }}
    p, li, span {{ color: #4A4A4A; line-height: 1.7; }}

    /* Ensure all main-area markdown text is dark */
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h4,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h5,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] h6 {{
        color: {CHARCOAL} !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p {{
        color: #4A4A4A !important;
    }}

    /* ── Hero ── */
    .hg-hero {{
        position: relative;
        border-radius: 8px;
        overflow: hidden;
        margin: -1rem -1rem 2.5rem -1rem;
        min-height: 520px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-size: cover;
        background-position: center;
    }}
    .hg-hero-overlay {{
        position: absolute;
        inset: 0;
        background: linear-gradient(
            135deg,
            rgba(32, 37, 34, 0.78) 0%,
            rgba(49, 92, 75, 0.55) 100%
        );
    }}
    .hg-hero-content {{
        position: relative;
        z-index: 2;
        text-align: center;
        padding: 3rem 2rem;
        max-width: 700px;
    }}
    .hg-hero-label {{
        font-size: 0.8rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: {CHAMPAGNE};
        margin-bottom: 0.75rem;
        font-weight: 600;
    }}
    .hg-hero-title {{
        font-size: 3.2rem;
        font-weight: 600;
        color: {IVORY};
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }}
    .hg-hero-tagline {{
        font-size: 1.15rem;
        color: {CHAMPAGNE};
        font-weight: 500;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }}
    .hg-hero-desc {{
        font-size: 1rem;
        color: rgba(247, 244, 237, 0.8);
        line-height: 1.7;
        margin-bottom: 2rem;
    }}
    .hg-hero-btns {{
        display: flex;
        gap: 0.75rem;
        justify-content: center;
        flex-wrap: wrap;
    }}
    .hg-btn {{
        padding: 12px 28px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
        text-decoration: none;
        cursor: pointer;
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
        display: inline-block;
        border: none;
    }}
    .hg-btn-primary {{
        background: {FOREST};
        color: {IVORY};
    }}
    .hg-btn-primary:hover {{
        background: #3d7a5f;
    }}
    .hg-btn-secondary {{
        background: transparent;
        color: {IVORY};
        border: 1.5px solid rgba(247, 244, 237, 0.4);
    }}
    .hg-btn-secondary:hover {{
        border-color: {CHAMPAGNE};
        color: {CHAMPAGNE};
    }}

    /* ── Section Heading ── */
    .hg-section {{
        margin: 2.5rem 0 1.5rem;
    }}
    .hg-section-title {{
        font-size: 1.7rem;
        font-weight: 600;
        color: {CHARCOAL};
        margin-bottom: 0.3rem;
    }}
    .hg-section-sub {{
        font-size: 0.95rem;
        color: #6B7280;
        max-width: 600px;
    }}

    /* ── KPI Card ── */
    .hg-kpi {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1.5rem 1.25rem;
        text-align: center;
        transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .hg-kpi:hover {{
        border-color: {CHAMPAGNE};
        box-shadow: 0 2px 8px rgba(197, 164, 109, 0.1);
    }}
    .hg-kpi-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {CHARCOAL};
        margin-bottom: 0.15rem;
    }}
    .hg-kpi-label {{
        font-size: 0.85rem;
        color: #6B7280;
        font-weight: 400;
    }}

    /* ── Info Card ── */
    .hg-card {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1.75rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }}
    .hg-card:hover {{
        border-color: #D1D5DB;
    }}

    /* ── Prediction Result ── */
    .hg-result {{
        border-radius: 6px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }}
    .hg-result-low {{
        background: rgba(79, 124, 99, 0.08);
        border: 1px solid rgba(79, 124, 99, 0.25);
    }}
    .hg-result-med {{
        background: rgba(192, 138, 69, 0.08);
        border: 1px solid rgba(192, 138, 69, 0.25);
    }}
    .hg-result-high {{
        background: rgba(184, 92, 92, 0.08);
        border: 1px solid rgba(184, 92, 92, 0.25);
    }}
    .hg-result-label {{
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6B7280;
        margin-bottom: 0.5rem;
    }}
    .hg-result-badge {{
        display: inline-block;
        padding: 4px 16px;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }}
    .hg-badge-low {{ background: {SUCCESS}; color: white; }}
    .hg-badge-med {{ background: {AMBER}; color: white; }}
    .hg-badge-high {{ background: {CORAL}; color: white; }}
    .hg-result-pct {{
        font-size: 3rem;
        font-weight: 700;
        color: {CHARCOAL};
        margin-bottom: 0.25rem;
    }}
    .hg-result-desc {{
        font-size: 0.9rem;
        color: #6B7280;
    }}

    /* ── Workflow ── */
    .hg-flow-step {{
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.7rem 0;
    }}
    .hg-flow-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {FOREST};
        flex-shrink: 0;
    }}
    .hg-flow-line {{
        width: 1px;
        height: 18px;
        background: #D1D5DB;
        margin-left: 3.5px;
    }}
    .hg-flow-text {{
        font-size: 0.93rem;
        color: #4A4A4A;
    }}

    /* ── Tech Badge ── */
    .hg-tech {{
        display: inline-block;
        padding: 5px 14px;
        border-radius: 3px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px;
        background: rgba(49, 92, 75, 0.08);
        border: 1px solid rgba(49, 92, 75, 0.2);
        color: {FOREST};
    }}

    /* ── Metric row ── */
    .hg-metric-row {{
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid #F3F4F6;
    }}
    .hg-metric-name {{ color: #6B7280; font-size: 0.9rem; }}
    .hg-metric-val {{ color: {CHARCOAL}; font-weight: 600; font-size: 0.9rem; }}

    /* ── Divider ── */
    .hg-divider {{
        height: 1px;
        background: #E5E7EB;
        margin: 2rem 0;
    }}

    /* ── Footer ── */
    .hg-footer {{
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid #E5E7EB;
    }}
    .hg-footer-brand {{
        font-weight: 600;
        color: {CHARCOAL};
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }}
    .hg-footer-sub {{
        color: #9CA3AF;
        font-size: 0.8rem;
    }}
    .hg-footer-note {{
        color: #D1D5DB;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }}

    /* ── Form styling ── */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {{
        background: white !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 4px !important;
        color: {CHARCOAL} !important;
    }}
    .stSelectbox > div > div {{
        background: white !important;
        border-color: #D1D5DB !important;
        border-radius: 4px !important;
    }}
    label {{
        color: #4A4A4A !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }}

    /* ── Button override ── */
    .stButton > button {{
        background: {FOREST} !important;
        color: {IVORY} !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: background 0.2s, box-shadow 0.2s !important;
    }}
    .stButton > button:hover {{
        background: #3d7a5f !important;
        box-shadow: 0 2px 8px rgba(49, 92, 75, 0.2) !important;
    }}
    .stButton > button:focus-visible {{
        outline: 2px solid {CHAMPAGNE} !important;
        outline-offset: 2px !important;
    }}

    /* ── Expander ── */
    .streamlit-expanderHeader {{
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }}

    /* ── Plotly ── */
    .js-plotly-plot .plotly .bg {{ fill: transparent !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid #E5E7EB;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #6B7280;
        font-weight: 500;
        border-radius: 0;
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        color: {CHARCOAL} !important;
        border-bottom: 2px solid {FOREST} !important;
    }}

    /* ── Loading spinner animation ── */
    @keyframes hg-spin {{
        to {{ transform: rotate(360deg); }}
    }}

    /* ── Filter badge ── */
    .hg-filter-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: {FOREST};
        color: white;
        margin-left: 0.5rem;
        vertical-align: middle;
    }}

    /* ── Comparison card ── */
    .hg-compare-card {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1.25rem;
        text-align: center;
    }}
    .hg-compare-label {{
        font-size: 0.7rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #9CA3AF;
        margin-bottom: 0.5rem;
    }}
    .hg-compare-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {CHARCOAL};
    }}

    /* ── Validation message ── */
    .hg-validation {{
        font-size: 0.8rem;
        color: {CORAL};
        margin-top: 2px;
    }}

    /* ── Responsive Design (Skill 15) ── */
    @media (max-width: 768px) {{
        .hg-hero {{
            min-height: 380px;
        }}
        .hg-hero-title {{
            font-size: 2.2rem;
        }}
        .hg-hero-desc {{
            font-size: 0.9rem;
        }}
        .hg-kpi-value {{
            font-size: 1.5rem !important;
        }}
        .hg-result-pct {{
            font-size: 2.2rem;
        }}
    }}

    /* ── Accessibility: focus-visible (Skill 24) ── */
    a:focus-visible, button:focus-visible, [tabindex]:focus-visible {{
        outline: 2px solid {CHAMPAGNE};
        outline-offset: 2px;
    }}

    /* ── Reduced motion (Skill 24) ── */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Plotly helpers  (Skill 23 — consistent charts)
# ═══════════════════════════════════════════════════════════════════
def style_fig(fig, height=400):
    """Apply consistent Plotly styling across all charts."""
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="#4A4A4A", size=12),
        margin=dict(l=30, r=20, t=40, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=height,
        xaxis=dict(gridcolor="#F3F4F6"),
        yaxis=dict(gridcolor="#F3F4F6"),
    )
    return fig


def render_chart_with_states(
    df_data, chart_fn, empty_title="No data available",
    empty_desc="Adjust your filters or add more data to see this chart."
):
    """Wrap chart rendering with loading/empty/error states (Skills 16, 17, 18, 23)."""
    if df_data is None or (hasattr(df_data, "empty") and df_data.empty) or len(df_data) == 0:
        render_empty_state(empty_title, empty_desc)
        return
    try:
        chart_fn(df_data)
    except Exception as e:
        render_error_state("Chart Error", f"Unable to render this chart: {e}", show_retry=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR  (Skill 13 — navigation with icons + dataset info)
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 1.5rem 0 0.5rem; text-align:center;">
        <div style="font-size:1.15rem; font-weight:600; color:{IVORY}; letter-spacing:1px;">
            HOTELGUARD AI
        </div>
        <div style="font-size:0.75rem; color:{CHAMPAGNE}; letter-spacing:1.5px; margin-top:4px;">
            Predict · Prevent · Optimize
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px; background:rgba(247,244,237,0.12); margin:1rem 0;"></div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔮  Prediction", "📊  Analytics", "💡  AI Insights", "⚙️  Model Info"],
        label_visibility="collapsed",
    )

    st.markdown(f'<div style="height:1px; background:rgba(247,244,237,0.12); margin:1rem 0;"></div>',
                unsafe_allow_html=True)

    # Model status
    st.markdown(f"""
    <div style="padding:0.5rem 0;">
        <div style="font-size:0.7rem; letter-spacing:1.5px; color:#9CA3AF; text-transform:uppercase;
            margin-bottom:0.5rem;">Model Status</div>
    </div>
    """, unsafe_allow_html=True)

    if model is not None:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:8px;height:8px;border-radius:50%;background:{SUCCESS};display:inline-block;" aria-hidden="true"></span>
            <span style="font-size:0.85rem; color:{IVORY};">Model Online</span>
        </div>
        <div style="font-size:0.78rem; color:#9CA3AF; margin-top:4px;">
            XGBClassifier v1.0
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:8px;height:8px;border-radius:50%;background:{CORAL};display:inline-block;" aria-hidden="true"></span>
            <span style="font-size:0.85rem; color:{IVORY};">Model Offline</span>
        </div>
        <div style="font-size:0.78rem; color:#9CA3AF; margin-top:4px;">
            Connect a trained model
        </div>
        """, unsafe_allow_html=True)

    # Dataset info (Skill 13 — contextual sidebar information)
    st.markdown(f'<div style="height:1px; background:rgba(247,244,237,0.12); margin:1rem 0;"></div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:0.5rem 0;">
        <div style="font-size:0.7rem; letter-spacing:1.5px; color:#9CA3AF; text-transform:uppercase;
            margin-bottom:0.5rem;">Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:8px;height:8px;border-radius:50%;background:{SUCCESS};display:inline-block;" aria-hidden="true"></span>
            <span style="font-size:0.85rem; color:{IVORY};">Connected</span>
        </div>
        <div style="font-size:0.78rem; color:#9CA3AF; margin-top:4px;">
            {len(df_raw):,} bookings · {df_raw.shape[1]} features
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:8px;height:8px;border-radius:50%;background:{CORAL};display:inline-block;" aria-hidden="true"></span>
            <span style="font-size:0.85rem; color:{IVORY};">Not Connected</span>
        </div>
        <div style="font-size:0.78rem; color:#9CA3AF; margin-top:4px;">
            Add hotel_bookings.csv
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # Hero
    bg_style = ""
    if hero_b64:
        bg_style = f"background-image:url('data:image/jpeg;base64,{hero_b64}');"

    st.markdown(f"""
    <div class="hg-hero" style="{bg_style}" role="banner" aria-label="HotelGuard AI hero section">
        <div class="hg-hero-overlay"></div>
        <div class="hg-hero-content">
            <div class="hg-hero-label">HotelGuard AI</div>
            <div class="hg-hero-title">Predict. Prevent.<br>Optimize.</div>
            <div class="hg-hero-desc">
                Intelligent hotel cancellation prediction for smarter
                revenue management.
            </div>
            <div class="hg-hero-btns">
                <span class="hg-btn hg-btn-primary">Predict Cancellation</span>
                <span class="hg-btn hg-btn-secondary">Explore Analytics</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Why HotelGuard AI?
    render_section_header("Why HotelGuard AI?")

    st.markdown("""
    <div class="hg-card">
        <p style="margin-bottom:0.75rem; font-size:1.02rem; line-height:1.6;">
            Hotel cancellations create uncertainty in room allocation, revenue forecasting,
            and operational planning. Even a few percentage points of unpredicted cancellations
            can significantly impact quarterly revenue.
        </p>
        <p style="margin-bottom:0; font-size:1.02rem; line-height:1.6;">
            <strong>HotelGuard AI</strong> uses machine learning to estimate cancellation risk before arrival,
            helping hotel revenue and operations teams make proactive allocation and overbooking decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    # KPIs  (Skill 22 — dashboard widgets with trends)
    if df_raw is not None:
        total_bookings = len(df_raw)
        cancel_rate = df_raw["is_canceled"].mean()
        n_features = df_raw.shape[1] - 1  # exclude target
        avg_adr_home = df_raw["adr"].mean()

        # Compute trend hints from data splits
        city_count = len(df_raw[df_raw["hotel"] == "City Hotel"])
        resort_count = len(df_raw[df_raw["hotel"] == "Resort Hotel"])

        kpi_values = [
            (f"{total_bookings:,}", "Bookings Analyzed",
             f"City: {city_count:,} · Resort: {resort_count:,}"),
            (f"{cancel_rate:.0%}", "Cancellation Rate",
             f"↑ Higher for City Hotels"),
            (f"{n_features}", "Predictive Features",
             "8 engineered features"),
            (f"€{avg_adr_home:.0f}", "Average Daily Rate",
             "Across all bookings"),
        ]
    else:
        kpi_values = [
            ("—", "Bookings Analyzed", ""),
            ("—", "Cancellation Rate", ""),
            ("—", "Predictive Features", ""),
            ("—", "Average Daily Rate", ""),
        ]

    cols = st.columns(4)
    for col, (val, label, trend) in zip(cols, kpi_values):
        with col:
            render_kpi_card(val, label, trend)

    render_divider()

    # Quick Stats  (Skill 22 — additional dashboard context)
    if df_raw is not None:
        render_section_header("Quick Insights", "Key patterns from the booking dataset.")

        qi1, qi2, qi3 = st.columns(3)
        with qi1:
            top_country = df_raw["country"].value_counts().head(1)
            if not top_country.empty:
                render_info_card(
                    f"<strong>Top Origin:</strong> {top_country.index[0]} "
                    f"({top_country.values[0]:,} bookings)",
                    border_left_color=FOREST
                )
        with qi2:
            repeat_pct = df_raw["is_repeated_guest"].mean()
            render_info_card(
                f"<strong>Repeat Guests:</strong> {repeat_pct:.1%} of bookings "
                f"are from returning guests",
                border_left_color=CHAMPAGNE
            )
        with qi3:
            avg_lead = df_raw["lead_time"].mean()
            render_info_card(
                f"<strong>Avg Lead Time:</strong> {avg_lead:.0f} days "
                f"between booking and arrival",
                border_left_color=AMBER
            )


# ═══════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮  Prediction":

    render_section_header(
        "Cancellation Risk Prediction",
        "Enter booking details to estimate the probability of cancellation."
    )

    if model is None:
        render_error_state(
            "Model Not Connected",
            "Connect your trained model to enable cancellation prediction. "
            "Ensure <code>models/model.pkl</code> exists in the project root.",
            show_retry=True
        )
        st.stop()

    # Tabs for single vs batch prediction  (Skills 2, 21)
    pred_tab, batch_tab = st.tabs(["Single Prediction", "Batch Prediction"])

    # ── SINGLE PREDICTION ──
    with pred_tab:
        with st.form("prediction_form"):

            # Hotel & Arrival
            st.markdown("#### Hotel & Arrival")
            ha1, ha2, ha3 = st.columns(3)
            with ha1:
                hotel = st.selectbox("Hotel *", ["City Hotel", "Resort Hotel"])
                arrival_date_year = st.selectbox("Arrival Year *", [2015, 2016, 2017], index=1)
            with ha2:
                arrival_date_month_name = st.selectbox("Arrival Month *", MONTH_NAMES, index=6)
                arrival_date_month = ORDINAL_MONTH_MAP[arrival_date_month_name]
                arrival_date_week_number = st.number_input("Week Number", 1, 53, 27)
            with ha3:
                arrival_date_day_of_month = st.number_input("Day of Month", 1, 31, 15)

            render_divider()

            # Guest Information
            st.markdown("#### Guest Information")
            gi1, gi2, gi3 = st.columns(3)
            with gi1:
                adults = st.number_input("Adults *", 0, 10, 2)
                children = st.number_input("Children", 0, 10, 0)
            with gi2:
                babies = st.number_input("Babies", 0, 5, 0)
                customer_type = st.selectbox("Customer Type",
                                             ["Transient", "Contract", "Transient-Party", "Group"])
            with gi3:
                country = st.text_input("Country Code *", "PRT",
                                        help="ISO 3166-1 alpha-3 code, e.g. PRT, GBR, USA")
                is_repeated_guest = st.selectbox("Repeated Guest", [0, 1])

            render_divider()

            # Stay Details
            st.markdown("#### Stay Details")
            sd1, sd2, sd3 = st.columns(3)
            with sd1:
                stays_in_weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
                stays_in_week_nights = st.number_input("Week Nights", 0, 20, 3)
            with sd2:
                meal = st.selectbox("Meal Plan", ["BB", "HB", "SC", "FB", "Undefined"])
            with sd3:
                reserved_room_type = st.selectbox("Reserved Room", list("ABCDEFGHILKP"))
                assigned_room_type = st.selectbox("Assigned Room", list("ABCDEFGHILKP"))

            render_divider()

            # Booking Information
            st.markdown("#### Booking Information")
            bi1, bi2, bi3 = st.columns(3)
            with bi1:
                lead_time = st.number_input("Lead Time *", 0, 800, 100,
                                            help="Days between booking and arrival")
                booking_changes = st.number_input("Booking Changes", 0, 25, 0)
            with bi2:
                previous_cancellations = st.number_input("Previous Cancellations", 0, 30, 0)
                previous_bookings_not_canceled = st.number_input("Previous Non-Canceled", 0, 80, 0)
            with bi3:
                days_in_waiting_list = st.number_input("Days in Waiting List", 0, 400, 0)

            render_divider()

            # Channel & Revenue
            st.markdown("#### Channel & Revenue")
            cr1, cr2, cr3 = st.columns(3)
            with cr1:
                market_segment = st.selectbox("Market Segment",
                                              ["Online TA", "Offline TA/TO", "Direct", "Corporate",
                                               "Groups", "Complementary", "Aviation", "Undefined"])
                distribution_channel = st.selectbox("Distribution Channel",
                                                     ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"])
            with cr2:
                deposit_type = st.selectbox("Deposit Type",
                                             ["No Deposit", "Non Refund", "Refundable"])
                adr = st.number_input("Average Daily Rate (€) *", 0.0, 5000.0, 100.0, step=5.0)
            with cr3:
                agent = st.number_input("Agent ID", 0, 600, 0,
                                        help="0 = no agent")
                required_car_parking_spaces = st.number_input("Parking Spaces", 0, 8, 0)
                total_of_special_requests = st.number_input("Special Requests", 0, 5, 0)

            st.markdown("")
            submitted = st.form_submit_button("Predict Cancellation Risk", use_container_width=True)

        # ── Handle prediction ──
        if submitted:
            # Validation  (Skill 5 — field validation)
            validation_errors = []
            if adults == 0 and children == 0 and babies == 0:
                validation_errors.append("At least one guest (adult, child, or baby) is required.")
            if len(country) != 3:
                validation_errors.append("Country code must be exactly 3 characters (ISO 3166-1 alpha-3).")
            if adr < 0:
                validation_errors.append("Average Daily Rate cannot be negative.")

            if validation_errors:
                for err in validation_errors:
                    st.markdown(f'<div class="hg-validation">⚠ {err}</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Analyzing booking details…"):
                    # Build raw row
                    raw_row = {
                        "hotel": hotel, "lead_time": lead_time,
                        "arrival_date_year": arrival_date_year,
                        "arrival_date_month": arrival_date_month,
                        "arrival_date_week_number": arrival_date_week_number,
                        "arrival_date_day_of_month": arrival_date_day_of_month,
                        "stays_in_weekend_nights": stays_in_weekend_nights,
                        "stays_in_week_nights": stays_in_week_nights,
                        "adults": adults, "children": children, "babies": babies,
                        "meal": meal, "country": country,
                        "market_segment": market_segment,
                        "distribution_channel": distribution_channel,
                        "is_repeated_guest": is_repeated_guest,
                        "previous_cancellations": previous_cancellations,
                        "previous_bookings_not_canceled": previous_bookings_not_canceled,
                        "reserved_room_type": reserved_room_type,
                        "assigned_room_type": assigned_room_type,
                        "booking_changes": booking_changes,
                        "deposit_type": deposit_type, "agent": agent,
                        "days_in_waiting_list": days_in_waiting_list,
                        "customer_type": customer_type, "adr": adr,
                        "required_car_parking_spaces": required_car_parking_spaces,
                        "total_of_special_requests": total_of_special_requests,
                    }
                    # Engineer features  (Skill 33 — reuse extracted function)
                    eng = engineer_features(raw_row)
                    raw_row.update(eng)

                    input_data = pd.DataFrame([raw_row])

                try:
                    prediction = model.predict(input_data)[0]
                    proba = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_data)[0]

                    cancel_pct = proba[1] * 100 if proba is not None else (80 if prediction == 1 else 20)

                    # Result card
                    risk_level, _ = render_result_card(cancel_pct)

                    # Feature importance / explanation
                    st.markdown("#### What influenced this prediction?")
                    if hasattr(model, "feature_importances_"):
                        importances = model.feature_importances_
                        feat_names = input_data.columns.tolist()
                        imp_df = pd.DataFrame({
                            "Feature": feat_names,
                            "Importance": importances[:len(feat_names)]
                        }).sort_values("Importance", ascending=False).head(8)

                        for _, row in imp_df.iterrows():
                            impact = ("High impact" if row["Importance"] > 0.08
                                      else "Moderate impact" if row["Importance"] > 0.03
                                      else "Low impact")
                            render_metric_row(
                                row['Feature'].replace('_', ' ').title(),
                                impact
                            )
                    elif hasattr(model, "named_steps"):
                        # Pipeline — try to get feature importance from final estimator
                        estimator = model.named_steps.get("classifier") or model.named_steps.get("model")
                        if estimator and hasattr(estimator, "feature_importances_"):
                            importances = estimator.feature_importances_
                            st.caption(f"Top features from the model's {len(importances)} internal features.")
                        else:
                            render_empty_state(
                                "Feature Explainability Unavailable",
                                "This model type does not expose feature importances. "
                                "Consider using SHAP for more detailed explanations."
                            )
                    else:
                        render_empty_state(
                            "Feature Explainability Unavailable",
                            "Feature-level explainability is not available for this model type."
                        )

                    # Business recommendation
                    st.markdown("#### Recommended Action")
                    if cancel_pct >= 60:
                        rec = ("Consider proactive confirmation and monitor room allocation carefully "
                               "for this booking. A pre-arrival follow-up may help reduce cancellation risk.")
                    elif cancel_pct >= 30:
                        rec = ("Monitor the booking and consider a confirmation reminder closer to the "
                               "arrival date. No immediate action is required.")
                    else:
                        rec = ("Booking appears relatively stable based on the model prediction. "
                               "Standard procedures apply.")

                    st.markdown(f"""
                    <div class="hg-card">
                        <p style="margin:0;">{rec}</p>
                        <p style="color:#9CA3AF; font-size:0.8rem; margin-top:0.75rem; margin-bottom:0;">
                            These are decision-support recommendations, not guaranteed outcomes.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("View input summary"):
                        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

                except Exception as e:
                    render_error_state(
                        "Prediction Failed",
                        f"Unable to generate prediction. Please verify that the model was trained "
                        f"with the same feature set.",
                        show_retry=True
                    )
                    with st.expander("Technical details"):
                        st.code(str(e))

    # ── BATCH PREDICTION  (Skills 2, 21 — file upload) ──
    with batch_tab:
        render_section_header(
            "Batch Prediction",
            "Upload a CSV file with booking data to predict cancellation risk for multiple bookings."
        )

        st.markdown("""
        <div class="hg-card" style="padding:1.25rem;">
            <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                color:#9CA3AF; margin-bottom:0.75rem;">Requirements</div>
            <p style="margin:0; font-size:0.88rem;">
                • CSV file with the same columns as the prediction form<br>
                • Maximum file size: 10 MB<br>
                • Supported format: <code>.csv</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload booking data",
            type=["csv"],
            help="Upload a CSV file containing booking data for batch prediction.",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Validate file size  (Skill 21)
            if uploaded_file.size > 10 * 1024 * 1024:
                render_error_state(
                    "File Too Large",
                    "The uploaded file exceeds the 10 MB limit. Please split your data into smaller files."
                )
            else:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(batch_df):,} bookings from {uploaded_file.name}")

                    with st.expander(f"Preview data ({min(5, len(batch_df))} rows)"):
                        st.dataframe(batch_df.head(), use_container_width=True)

                    if st.button("Run Batch Prediction", use_container_width=True, key="batch_predict"):
                        with st.spinner(f"Processing {len(batch_df):,} bookings…"):
                            progress = st.progress(0)
                            results = []
                            errors = 0

                            for idx, row in batch_df.iterrows():
                                try:
                                    row_dict = row.to_dict()
                                    # Check if month is string and convert
                                    if "arrival_date_month" in row_dict:
                                        month_val = row_dict["arrival_date_month"]
                                        if isinstance(month_val, str) and month_val in ORDINAL_MONTH_MAP:
                                            row_dict["arrival_date_month"] = ORDINAL_MONTH_MAP[month_val]
                                    eng = engineer_features(row_dict)
                                    row_dict.update(eng)
                                    input_row = pd.DataFrame([row_dict])
                                    pred = model.predict(input_row)[0]
                                    prob = model.predict_proba(input_row)[0][1] if hasattr(model, "predict_proba") else (0.8 if pred == 1 else 0.2)
                                    risk = "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low"
                                    results.append({
                                        "booking_index": idx,
                                        "cancel_probability": round(prob * 100, 1),
                                        "risk_level": risk,
                                        "prediction": int(pred)
                                    })
                                except Exception:
                                    errors += 1
                                    results.append({
                                        "booking_index": idx,
                                        "cancel_probability": None,
                                        "risk_level": "Error",
                                        "prediction": None
                                    })
                                progress.progress((idx + 1) / len(batch_df))

                            result_df = pd.DataFrame(results)
                            progress.empty()

                        # Results summary
                        st.markdown("#### Batch Results")
                        valid_results = result_df[result_df["risk_level"] != "Error"]

                        if not valid_results.empty:
                            br1, br2, br3, br4 = st.columns(4)
                            with br1:
                                render_kpi_card(f"{len(valid_results):,}", "Processed", font_size="1.5rem")
                            with br2:
                                high_risk = len(valid_results[valid_results["risk_level"] == "High"])
                                render_kpi_card(f"{high_risk}", "High Risk", font_size="1.5rem")
                            with br3:
                                med_risk = len(valid_results[valid_results["risk_level"] == "Medium"])
                                render_kpi_card(f"{med_risk}", "Medium Risk", font_size="1.5rem")
                            with br4:
                                low_risk = len(valid_results[valid_results["risk_level"] == "Low"])
                                render_kpi_card(f"{low_risk}", "Low Risk", font_size="1.5rem")

                        if errors > 0:
                            st.warning(f"{errors} bookings could not be processed due to data issues.")

                        st.dataframe(result_df, use_container_width=True)

                        # Download  (Skill 21)
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "Download Results (CSV)",
                            csv_buffer.getvalue(),
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                except Exception as e:
                    render_error_state(
                        "Invalid File",
                        "Unable to read the uploaded CSV. Please check the file format and try again.",
                        show_retry=True
                    )
                    with st.expander("Technical details"):
                        st.code(str(e))
        else:
            render_empty_state(
                "No file uploaded",
                "Upload a CSV file with booking data to run batch predictions. "
                "The file should contain the same columns as the single prediction form.",
                cta_text="Upload CSV"
            )


# ═══════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════
elif page == "📊  Analytics":

    render_section_header(
        "Booking & Cancellation Analytics",
        "Explore historical cancellation patterns across hotel, booking, customer, and revenue dimensions."
    )

    if df_raw is None:
        render_error_state(
            "Dataset Not Connected",
            "Ensure <code>hotel_bookings.csv</code> is in the project root directory. "
            "The analytics dashboard requires the full booking dataset to generate insights.",
            show_retry=True
        )
        st.stop()

    # Tabs for charts vs data explorer  (Skill 6)
    chart_tab, explorer_tab = st.tabs(["Charts & KPIs", "Data Explorer"])

    with chart_tab:
        df = df_raw.copy()

        # ── Filters  (Skill 20 — filter builder) ──
        with st.container():
            f1, f2, f3, f4, f5 = st.columns(5)
            with f1:
                hotel_filter = st.selectbox("Hotel", ["All"] + sorted(df["hotel"].unique().tolist()),
                                            key="af_hotel")
            with f2:
                year_filter = st.selectbox("Arrival Year",
                                           ["All"] + sorted(df["arrival_date_year"].unique().tolist()),
                                           key="af_year")
            with f3:
                seg_filter = st.selectbox("Market Segment",
                                          ["All"] + sorted(df["market_segment"].unique().tolist()),
                                          key="af_seg")
            with f4:
                cust_filter = st.selectbox("Customer Type",
                                           ["All"] + sorted(df["customer_type"].unique().tolist()),
                                           key="af_cust")
            with f5:
                dep_filter = st.selectbox("Deposit Type",
                                          ["All"] + sorted(df["deposit_type"].unique().tolist()),
                                          key="af_dep")

        # Apply filters
        if hotel_filter != "All":
            df = df[df["hotel"] == hotel_filter]
        if year_filter != "All":
            df = df[df["arrival_date_year"] == year_filter]
        if seg_filter != "All":
            df = df[df["market_segment"] == seg_filter]
        if cust_filter != "All":
            df = df[df["customer_type"] == cust_filter]
        if dep_filter != "All":
            df = df[df["deposit_type"] == dep_filter]

        # Filter status + reset  (Skill 20)
        active_filters = sum(1 for f in [hotel_filter, year_filter, seg_filter, cust_filter, dep_filter] if f != "All")
        filter_status_col, reset_col = st.columns([4, 1])
        with filter_status_col:
            if active_filters > 0:
                st.markdown(
                    f'<div style="font-size:0.85rem; color:#6B7280; margin:0.5rem 0;">'
                    f'Showing <strong>{len(df):,}</strong> of {len(df_raw):,} bookings '
                    f'<span class="hg-filter-badge">{active_filters} filter{"s" if active_filters > 1 else ""}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="font-size:0.85rem; color:#6B7280; margin:0.5rem 0;">'
                    f'Showing all <strong>{len(df):,}</strong> bookings</div>',
                    unsafe_allow_html=True
                )
        with reset_col:
            if active_filters > 0:
                if st.button("Reset Filters", key="reset_filters"):
                    st.rerun()

        render_divider()

        # ── Empty state for filtered results  (Skill 18) ──
        if len(df) == 0:
            render_empty_state(
                "No bookings match your filters",
                "Try adjusting or removing some filters to see booking data. "
                "Click 'Reset Filters' to see all bookings."
            )
            st.stop()

        # ── KPIs ──
        total = len(df)
        cancelled = df["is_canceled"].sum()
        cancel_rate = df["is_canceled"].mean() if total > 0 else 0
        avg_lead = df["lead_time"].mean() if total > 0 else 0
        avg_adr = df["adr"].mean() if total > 0 else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        for col, (v, l) in zip([k1, k2, k3, k4, k5], [
            (f"{total:,}", "Total Bookings"),
            (f"{cancelled:,}", "Cancelled"),
            (f"{cancel_rate:.1%}", "Cancellation Rate"),
            (f"{avg_lead:.0f} days", "Avg Lead Time"),
            (f"€{avg_adr:.0f}", "Avg ADR"),
        ]):
            with col:
                render_kpi_card(v, l, font_size="1.5rem")

        render_divider()

        # ── Charts  (Skill 23 — consistent charts with states) ──
        c1, c2 = st.columns(2)

        with c1:
            def chart_hotel_cancel(data):
                hotel_data = data.groupby("hotel")["is_canceled"].agg(["mean", "count"]).reset_index()
                hotel_data.columns = ["Hotel", "Rate", "Count"]
                fig = px.bar(hotel_data, x="Hotel", y="Rate",
                             text=hotel_data["Rate"].apply(lambda x: f"{x:.1%}"),
                             color_discrete_sequence=[FOREST],
                             custom_data=["Count"])
                fig.update_traces(textposition="outside",
                                  hovertemplate="<b>%{x}</b><br>Rate: %{y:.1%}<br>Bookings: %{customdata[0]:,}<extra></extra>")
                fig.update_layout(title="Cancellation by Hotel", yaxis_title="Cancellation Rate",
                                  xaxis_title="", yaxis_tickformat=".0%")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

            render_chart_with_states(df, chart_hotel_cancel,
                                     "No hotel data", "Add booking data to see hotel cancellation rates.")

        with c2:
            def chart_segment_cancel(data):
                seg_data = data.groupby("market_segment")["is_canceled"].mean().reset_index()
                seg_data.columns = ["Segment", "Rate"]
                seg_data = seg_data.sort_values("Rate", ascending=True)
                fig = px.bar(seg_data, x="Rate", y="Segment", orientation="h",
                             text=seg_data["Rate"].apply(lambda x: f"{x:.1%}"),
                             color_discrete_sequence=[CHAMPAGNE])
                fig.update_traces(textposition="outside",
                                  hovertemplate="<b>%{y}</b><br>Rate: %{x:.1%}<extra></extra>")
                fig.update_layout(title="Cancellation by Market Segment",
                                  xaxis_title="Cancellation Rate", yaxis_title="", xaxis_tickformat=".0%")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

            render_chart_with_states(df, chart_segment_cancel,
                                     "No segment data", "Add booking data to see market segment analysis.")

        c3, c4 = st.columns(2)

        with c3:
            def chart_deposit_cancel(data):
                dep_data = data.groupby("deposit_type")["is_canceled"].mean().reset_index()
                dep_data.columns = ["Deposit", "Rate"]
                fig = px.bar(dep_data, x="Deposit", y="Rate",
                             text=dep_data["Rate"].apply(lambda x: f"{x:.1%}"),
                             color_discrete_sequence=[CORAL])
                fig.update_traces(textposition="outside",
                                  hovertemplate="<b>%{x}</b><br>Rate: %{y:.1%}<extra></extra>")
                fig.update_layout(title="Cancellation by Deposit Type",
                                  yaxis_title="Cancellation Rate", xaxis_title="", yaxis_tickformat=".0%")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

            render_chart_with_states(df, chart_deposit_cancel,
                                     "No deposit data", "Add data to see deposit type analysis.")

        with c4:
            def chart_leadtime_cancel(data):
                lt_bins = pd.cut(data["lead_time"], bins=[0, 30, 90, 180, 365, 800],
                                 labels=["0–30", "31–90", "91–180", "181–365", "365+"])
                lt_data = data.groupby(lt_bins, observed=True)["is_canceled"].mean().reset_index()
                lt_data.columns = ["Lead Time (days)", "Rate"]
                fig = px.bar(lt_data, x="Lead Time (days)", y="Rate",
                             text=lt_data["Rate"].apply(lambda x: f"{x:.1%}"),
                             color_discrete_sequence=[AMBER])
                fig.update_traces(textposition="outside",
                                  hovertemplate="<b>%{x} days</b><br>Rate: %{y:.1%}<extra></extra>")
                fig.update_layout(title="Cancellation vs Lead Time",
                                  yaxis_title="Cancellation Rate", xaxis_title="", yaxis_tickformat=".0%")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

            render_chart_with_states(df, chart_leadtime_cancel,
                                     "No lead time data", "Add data to see lead time analysis.")

        c5, c6 = st.columns(2)

        with c5:
            def chart_customer_cancel(data):
                ct_data = data.groupby("customer_type")["is_canceled"].mean().reset_index()
                ct_data.columns = ["Customer", "Rate"]
                fig = px.bar(ct_data, x="Customer", y="Rate",
                             text=ct_data["Rate"].apply(lambda x: f"{x:.1%}"),
                             color_discrete_sequence=[SUCCESS])
                fig.update_traces(textposition="outside",
                                  hovertemplate="<b>%{x}</b><br>Rate: %{y:.1%}<extra></extra>")
                fig.update_layout(title="Cancellation by Customer Type",
                                  yaxis_title="Cancellation Rate", xaxis_title="", yaxis_tickformat=".0%")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

            render_chart_with_states(df, chart_customer_cancel,
                                     "No customer data", "Add data to see customer type analysis.")

        with c6:
            def chart_monthly_trend(data):
                month_order = list(ORDINAL_MONTH_MAP.keys())
                monthly = data.groupby("arrival_date_month")["is_canceled"].mean().reset_index()
                if monthly["arrival_date_month"].dtype in [np.int64, np.int32, np.float64]:
                    inv_map = {v: k for k, v in ORDINAL_MONTH_MAP.items()}
                    monthly["Month"] = monthly["arrival_date_month"].map(inv_map)
                else:
                    monthly["Month"] = monthly["arrival_date_month"]
                monthly["Rate"] = monthly["is_canceled"]
                monthly["Month"] = pd.Categorical(monthly["Month"], categories=month_order, ordered=True)
                monthly = monthly.sort_values("Month").dropna(subset=["Month"])
                fig = px.line(monthly, x="Month", y="Rate", markers=True,
                              color_discrete_sequence=[FOREST])
                fig.update_traces(hovertemplate="<b>%{x}</b><br>Rate: %{y:.1%}<extra></extra>")
                fig.update_layout(title="Monthly Cancellation Trend",
                                  yaxis_title="Cancellation Rate", xaxis_title="", yaxis_tickformat=".0%")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

            render_chart_with_states(df, chart_monthly_trend,
                                     "No monthly data", "Add data to see monthly cancellation trends.")

    # ── DATA EXPLORER  (Skill 6 — table builder) ──
    with explorer_tab:
        render_section_header(
            "Data Explorer",
            "Browse and search the raw booking dataset."
        )

        if df_raw is not None:
            explorer_df = df_raw.copy()

            # Search and filter controls
            search_col, rows_col = st.columns([3, 1])
            with search_col:
                search_query = st.text_input(
                    "Search bookings",
                    placeholder="Search by country, hotel, market segment…",
                    key="explorer_search",
                    label_visibility="collapsed"
                )
            with rows_col:
                rows_per_page = st.selectbox("Rows per page", [25, 50, 100], key="rows_per_page")

            # Apply search
            if search_query:
                mask = explorer_df.astype(str).apply(
                    lambda col: col.str.contains(search_query, case=False, na=False)
                ).any(axis=1)
                explorer_df = explorer_df[mask]

            st.markdown(
                f'<div style="font-size:0.85rem; color:#6B7280; margin-bottom:0.5rem;">'
                f'Showing {min(rows_per_page, len(explorer_df)):,} of {len(explorer_df):,} bookings</div>',
                unsafe_allow_html=True
            )

            if len(explorer_df) == 0:
                render_empty_state(
                    "No matching bookings",
                    "Your search didn't match any bookings. Try a different search term."
                )
            else:
                # Pagination  (Skill 6)
                total_pages = max(1, (len(explorer_df) - 1) // rows_per_page + 1)
                current_page = st.number_input("Page", 1, total_pages, 1, key="explorer_page",
                                               label_visibility="collapsed")
                start_idx = (current_page - 1) * rows_per_page
                end_idx = start_idx + rows_per_page

                st.dataframe(
                    explorer_df.iloc[start_idx:end_idx],
                    use_container_width=True,
                    height=500
                )

                st.markdown(
                    f'<div style="text-align:center; font-size:0.82rem; color:#9CA3AF; margin-top:0.5rem;">'
                    f'Page {current_page} of {total_pages}</div>',
                    unsafe_allow_html=True
                )
        else:
            render_empty_state(
                "No dataset available",
                "Connect the booking dataset to explore the data. "
                "Place hotel_bookings.csv in the project root."
            )


# ═══════════════════════════════════════════════════════════════════
# PAGE: AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif page == "💡  AI Insights":

    render_section_header(
        "AI Business Insights",
        "Model-driven insights translated into actionable business recommendations."
    )

    # Tabs for insights vs comparison
    insights_tab, compare_tab = st.tabs(["Insights & Patterns", "Booking Comparison"])

    with insights_tab:
        # Cancellation Drivers
        st.markdown("#### Cancellation Drivers")

        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_names = [c for c in (df_raw.columns.tolist() if df_raw is not None else []) if c != "is_canceled"]
            if len(feat_names) >= len(importances):
                feat_names = feat_names[:len(importances)]
            elif len(importances) > len(feat_names):
                feat_names = [f"feature_{i}" for i in range(len(importances))]

            imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=True).tail(12)

            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[FOREST])
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
            )
            fig.update_layout(title="", xaxis_title="Relative Importance", yaxis_title="")
            style_fig(fig, height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            render_empty_state(
                "Feature Importance Unavailable",
                "Feature importance data will be available once a tree-based model "
                "is connected (Random Forest, XGBoost, LightGBM, etc.). "
                "Train your model and save it to models/model.pkl."
            )

        render_divider()

        # Booking Risk Patterns
        st.markdown("#### Booking Risk Patterns")

        if df_raw is not None:
            patterns = []
            # Deposit analysis
            nr_subset = df_raw[df_raw["deposit_type"] == "Non Refund"]
            if len(nr_subset) > 0:
                nr_rate = nr_subset["is_canceled"].mean()
                if nr_rate > 0.8:
                    patterns.append(
                        f"<strong>Deposit Risk:</strong> Non-refundable deposits show a {nr_rate:.0%} cancellation rate — "
                        "significantly higher than other deposit types."
                    )
            # Lead time
            long_lead = df_raw[df_raw["lead_time"] > 200]["is_canceled"].mean()
            short_lead = df_raw[df_raw["lead_time"] <= 30]["is_canceled"].mean()
            if long_lead > short_lead * 1.5:
                patterns.append(
                    f"<strong>Lead Time Effect:</strong> Bookings with lead time > 200 days cancel at {long_lead:.0%}, "
                    f"compared to {short_lead:.0%} for bookings within 30 days."
                )
            # Hotel type
            city_rate = df_raw[df_raw["hotel"] == "City Hotel"]["is_canceled"].mean()
            resort_rate = df_raw[df_raw["hotel"] == "Resort Hotel"]["is_canceled"].mean()
            patterns.append(
                f"<strong>Hotel Type:</strong> City Hotel cancellations ({city_rate:.0%}) are higher than "
                f"Resort Hotel ({resort_rate:.0%})."
            )
            # Repeated guests
            rep_rate = df_raw[df_raw["is_repeated_guest"] == 1]["is_canceled"].mean()
            new_rate = df_raw[df_raw["is_repeated_guest"] == 0]["is_canceled"].mean()
            patterns.append(
                f"<strong>Guest Loyalty:</strong> Repeated guests cancel at {rep_rate:.0%} vs {new_rate:.0%} for new guests."
            )
            # Special requests
            sr_zero = df_raw[df_raw["total_of_special_requests"] == 0]["is_canceled"].mean()
            sr_pos = df_raw[df_raw["total_of_special_requests"] > 0]["is_canceled"].mean()
            patterns.append(
                f"<strong>Special Requests:</strong> Guests with no special requests cancel at {sr_zero:.0%} "
                f"vs {sr_pos:.0%} for those with requests — engagement signals lower risk."
            )

            for p in patterns:
                render_info_card(p, border_left_color=FOREST)
        else:
            render_empty_state(
                "No Data Available",
                "Connect the booking dataset to discover risk patterns. "
                "Place hotel_bookings.csv in the project root.",
            )

        render_divider()

        # Revenue Management Recommendations
        st.markdown("#### Revenue Management Recommendations")

        recs = [
            ("🎯", "Monitor high-risk bookings with long lead times and consider proactive confirmation strategies."),
            ("📊", "Use cancellation probability when planning room inventory to balance overbooking risk."),
            ("📧", "Consider proactive communication for high-risk bookings approaching their arrival date."),
            ("💳", "Evaluate deposit policies for segments with elevated cancellation rates."),
            ("🏆", "Repeated guests show lower cancellation risk — loyalty programs may further reduce attrition."),
        ]
        for icon, r in recs:
            render_info_card(f"{icon}  {r}", border_left_color=CHAMPAGNE)

        st.caption("Recommendations are derived from dataset analysis and model outputs. "
                   "They are decision-support suggestions, not guaranteed outcomes.")

    # ── BOOKING COMPARISON  (Skill 2 — new feature) ──
    with compare_tab:
        render_section_header(
            "Booking Comparison",
            "Compare two booking profiles side-by-side to understand what drives cancellation risk."
        )

        if model is None:
            render_error_state(
                "Model Required",
                "Connect a trained model to enable booking comparison.",
                show_retry=True
            )
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("##### Booking A")
                hotel_a = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"], key="cmp_hotel_a")
                lead_a = st.number_input("Lead Time", 0, 800, 50, key="cmp_lead_a")
                deposit_a = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"], key="cmp_dep_a")
                adults_a = st.number_input("Adults", 0, 10, 2, key="cmp_adults_a")
                children_a = st.number_input("Children", 0, 10, 0, key="cmp_children_a")
                country_a = st.text_input("Country", "PRT", key="cmp_country_a")
                repeated_a = st.selectbox("Repeated Guest", [0, 1], key="cmp_rep_a")
                special_a = st.number_input("Special Requests", 0, 5, 1, key="cmp_special_a")

            with col_b:
                st.markdown("##### Booking B")
                hotel_b = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"], key="cmp_hotel_b", index=1)
                lead_b = st.number_input("Lead Time", 0, 800, 300, key="cmp_lead_b")
                deposit_b = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"], key="cmp_dep_b", index=1)
                adults_b = st.number_input("Adults", 0, 10, 1, key="cmp_adults_b")
                children_b = st.number_input("Children", 0, 10, 0, key="cmp_children_b")
                country_b = st.text_input("Country", "GBR", key="cmp_country_b")
                repeated_b = st.selectbox("Repeated Guest", [0, 1], key="cmp_rep_b")
                special_b = st.number_input("Special Requests", 0, 5, 0, key="cmp_special_b")

            if st.button("Compare Bookings", use_container_width=True, key="compare_btn"):
                with st.spinner("Comparing bookings…"):
                    # Build minimal rows with defaults for missing fields
                    defaults = {
                        "arrival_date_year": 2017, "arrival_date_month": 7,
                        "arrival_date_week_number": 27, "arrival_date_day_of_month": 15,
                        "stays_in_weekend_nights": 1, "stays_in_week_nights": 3,
                        "babies": 0, "meal": "BB", "market_segment": "Online TA",
                        "distribution_channel": "TA/TO",
                        "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
                        "reserved_room_type": "A", "assigned_room_type": "A",
                        "booking_changes": 0, "agent": 0,
                        "days_in_waiting_list": 0, "customer_type": "Transient",
                        "adr": 100.0, "required_car_parking_spaces": 0,
                    }

                    row_a = {**defaults, "hotel": hotel_a, "lead_time": lead_a,
                             "deposit_type": deposit_a, "adults": adults_a,
                             "children": children_a, "country": country_a,
                             "is_repeated_guest": repeated_a,
                             "total_of_special_requests": special_a}
                    row_b = {**defaults, "hotel": hotel_b, "lead_time": lead_b,
                             "deposit_type": deposit_b, "adults": adults_b,
                             "children": children_b, "country": country_b,
                             "is_repeated_guest": repeated_b,
                             "total_of_special_requests": special_b}

                    for row in [row_a, row_b]:
                        eng = engineer_features(row)
                        row.update(eng)

                    try:
                        df_a = pd.DataFrame([row_a])
                        df_b = pd.DataFrame([row_b])
                        prob_a = model.predict_proba(df_a)[0][1] * 100 if hasattr(model, "predict_proba") else 50
                        prob_b = model.predict_proba(df_b)[0][1] * 100 if hasattr(model, "predict_proba") else 50

                        render_divider()
                        st.markdown("#### Comparison Results")

                        res1, res2 = st.columns(2)
                        with res1:
                            st.markdown(f"""
                            <div class="hg-compare-card">
                                <div class="hg-compare-label">Booking A</div>
                                <div class="hg-compare-value" style="color:{SUCCESS if prob_a < 30 else AMBER if prob_a < 60 else CORAL};">
                                    {prob_a:.1f}%
                                </div>
                                <div style="font-size:0.82rem; color:#6B7280; margin-top:4px;">
                                    {"Low" if prob_a < 30 else "Medium" if prob_a < 60 else "High"} Risk
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with res2:
                            st.markdown(f"""
                            <div class="hg-compare-card">
                                <div class="hg-compare-label">Booking B</div>
                                <div class="hg-compare-value" style="color:{SUCCESS if prob_b < 30 else AMBER if prob_b < 60 else CORAL};">
                                    {prob_b:.1f}%
                                </div>
                                <div style="font-size:0.82rem; color:#6B7280; margin-top:4px;">
                                    {"Low" if prob_b < 30 else "Medium" if prob_b < 60 else "High"} Risk
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Key differences
                        diff = abs(prob_a - prob_b)
                        higher = "Booking A" if prob_a > prob_b else "Booking B"
                        st.markdown(f"""
                        <div class="hg-card" style="margin-top:1rem;">
                            <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                                color:#9CA3AF; margin-bottom:0.75rem;">Analysis</div>
                            <p style="margin:0; font-size:0.92rem;">
                                <strong>{higher}</strong> has a <strong>{diff:.1f}%</strong> higher cancellation risk.
                                Key differences: lead time ({lead_a} vs {lead_b} days),
                                deposit type ({deposit_a} vs {deposit_b}),
                                and guest loyalty ({"repeat" if repeated_a else "new"} vs {"repeat" if repeated_b else "new"}).
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        render_error_state(
                            "Comparison Failed",
                            "Unable to compare bookings. Check model compatibility.",
                            show_retry=True
                        )
                        with st.expander("Technical details"):
                            st.code(str(e))


# ═══════════════════════════════════════════════════════════════════
# PAGE: MODEL INFORMATION
# ═══════════════════════════════════════════════════════════════════
elif page == "⚙️  Model Info":

    render_section_header(
        "Model Information",
        "Technical details of the machine learning pipeline."
    )

    mi1, mi2 = st.columns([1, 1], gap="large")

    with mi1:
        # Problem  (Skill 7 — display important metadata)
        st.markdown("""
        <div class="hg-card">
            <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                color:#9CA3AF; margin-bottom:0.75rem;">Problem</div>
            <p style="margin:0;">Binary classification — predict whether a hotel booking
            will be cancelled (<code>1</code>) or not (<code>0</code>).</p>
        </div>
        """, unsafe_allow_html=True)

        # Dataset
        if df_raw is not None:
            n_rows, n_cols = df_raw.shape
            ds_text = f"~{n_rows:,} bookings with {n_cols} features (numerical + categorical)."
        else:
            ds_text = "Dataset not connected."

        st.markdown(f"""
        <div class="hg-card">
            <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                color:#9CA3AF; margin-bottom:0.75rem;">Dataset</div>
            <p style="margin:0;">{ds_text}</p>
        </div>
        """, unsafe_allow_html=True)

        # Model  (Skill 7 — entity name, version, date)
        if metadata:
            model_type = metadata.get("model_type", "Unknown")
            model_version = metadata.get("version", "—")
            training_date = metadata.get("training_date", "—")
            if training_date != "—":
                training_date = training_date[:10]  # YYYY-MM-DD
        elif model is not None:
            model_type = type(model).__name__
            model_version = "—"
            training_date = "—"
        else:
            model_type = "Not connected"
            model_version = "—"
            training_date = "—"

        st.markdown(f"""
        <div class="hg-card">
            <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                color:#9CA3AF; margin-bottom:0.75rem;">Best Model</div>
            <p style="margin:0; font-weight:600; font-size:1.05rem;">{model_type}</p>
        </div>
        """, unsafe_allow_html=True)

        # Version & Training Date
        st.markdown(f"""
        <div class="hg-card">
            <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                color:#9CA3AF; margin-bottom:0.75rem;">Model Details</div>
        """, unsafe_allow_html=True)
        render_metric_row("Version", model_version)
        render_metric_row("Training Date", training_date)
        render_metric_row("Random Seed", str(metadata.get("random_seed", "—")) if metadata else "—")
        st.markdown('</div>', unsafe_allow_html=True)

    with mi2:
        # Pipeline  (Skill 7 — clear entity hierarchy)
        render_flow_steps(
            ["Raw Dataset", "Data Cleaning", "Feature Engineering",
             "Train / Test Split", "Preprocessing Pipeline",
             "Model Training", "Hyperparameter Tuning",
             "Model Evaluation", "Saved Model", "Streamlit Prediction"],
            container_label="ML Pipeline"
        )

    render_divider()

    # Metrics  (Skill 7 — prominent metrics display)
    st.markdown("#### Evaluation Metrics")

    if metadata and metadata.get("metrics"):
        metrics = metadata["metrics"]
        m_cols = st.columns(len(metrics))
        for col, (k, v) in zip(m_cols, metrics.items()):
            with col:
                display_val = f"{v:.4f}" if v is not None else "—"
                render_kpi_card(display_val, k.replace('_', ' ').title(), font_size="1.5rem")
    else:
        render_empty_state(
            "Metrics Not Available",
            "Model metrics will appear here once a trained model with metadata is connected. "
            "Run notebook 05 to train the final model and save metrics."
        )

    render_divider()

    # Feature Groups  (Skill 7 — show related records)
    if metadata and metadata.get("features"):
        st.markdown("#### Feature Groups")
        feat_info = metadata["features"]

        fg1, fg2, fg3 = st.columns(3)
        with fg1:
            st.markdown(f"""
            <div class="hg-card" style="padding:1.25rem;">
                <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                    color:#9CA3AF; margin-bottom:0.75rem;">Numeric ({len(feat_info.get('numeric', []))})</div>
            """, unsafe_allow_html=True)
            for f in feat_info.get("numeric", [])[:10]:
                st.markdown(f'<span class="hg-tech">{f}</span>', unsafe_allow_html=True)
            remaining = len(feat_info.get("numeric", [])) - 10
            if remaining > 0:
                st.markdown(f'<div style="font-size:0.8rem; color:#9CA3AF; margin-top:8px;">+{remaining} more</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with fg2:
            st.markdown(f"""
            <div class="hg-card" style="padding:1.25rem;">
                <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                    color:#9CA3AF; margin-bottom:0.75rem;">Categorical Low-Card ({len(feat_info.get('categorical_low', []))})</div>
            """, unsafe_allow_html=True)
            for f in feat_info.get("categorical_low", []):
                st.markdown(f'<span class="hg-tech">{f}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with fg3:
            st.markdown(f"""
            <div class="hg-card" style="padding:1.25rem;">
                <div style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                    color:#9CA3AF; margin-bottom:0.75rem;">Categorical High-Card ({len(feat_info.get('categorical_high', []))})</div>
            """, unsafe_allow_html=True)
            for f in feat_info.get("categorical_high", []):
                st.markdown(f'<span class="hg-tech">{f}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    render_divider()

    # Technologies
    st.markdown("#### Technologies")
    techs = ["Python", "Scikit-Learn", "XGBoost", "LightGBM", "Pandas",
             "NumPy", "Plotly", "Streamlit", "MLflow", "Docker"]
    tech_html = "".join(f'<span class="hg-tech">{t}</span>' for t in techs)
    st.markdown(f'<div style="margin-bottom:1rem;">{tech_html}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# FOOTER (all pages)  (Skill 35 — consistency)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hg-footer" role="contentinfo">
    <div class="hg-footer-brand">HotelGuard AI</div>
    <div class="hg-footer-sub">
        Machine Learning · Hospitality Analytics · Revenue Intelligence
    </div>
    <div class="hg-footer-note">
        Built for educational and demonstration purposes.
    </div>
</div>
""", unsafe_allow_html=True)
