"""
UI Styling & Components
==========================
Design tokens and reusable HTML/CSS components for the HR Attrition app.
Keeps app.py focused on logic; this file owns the look.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#F7F8FA",
    "surface": "#FFFFFF",
    "navy": "#16233F",
    "navy_light": "#28395C",
    "accent": "#2D6E7E",
    "accent_light": "#E4EEF0",
    "text": "#1A1D23",
    "text_muted": "#6B7280",
    "border": "#E5E7EB",
    "low": "#2F9E44",
    "low_bg": "#E9F7EC",
    "medium": "#DE9A00",
    "medium_bg": "#FDF3DC",
    "high": "#D64550",
    "high_bg": "#FBE7E9",
}

RISK_COLOR_SCALE = [COLORS["low"], COLORS["medium"], COLORS["high"]]


def risk_tier(prob: float) -> str:
    if prob >= 0.5:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


def risk_label(prob: float) -> str:
    return {"high": "High risk", "medium": "Medium risk", "low": "Low risk"}[risk_tier(prob)]


def inject_base_styles():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {COLORS['text']};
        }}

        h1, h2, h3 {{
            font-family: 'Space Grotesk', sans-serif !important;
            color: {COLORS['navy']} !important;
            letter-spacing: -0.01em;
        }}

        .stApp {{
            background-color: {COLORS['bg']};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COLORS['navy']};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E8ECF3 !important;
        }}
        section[data-testid="stSidebar"] input {{
            color: {COLORS['text']} !important;
        }}

        /* Metric numbers in mono, dashboard-style */
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {COLORS['navy']} !important;
        }}

        /* Card container */
        .card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.25rem;
        }}

        .step-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: {COLORS['accent']};
            color: white;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.6rem;
        }}

        .risk-badge {{
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .risk-high {{ background: {COLORS['high_bg']}; color: {COLORS['high']}; }}
        .risk-medium {{ background: {COLORS['medium_bg']}; color: {COLORS['medium']}; }}
        .risk-low {{ background: {COLORS['low_bg']}; color: {COLORS['low']}; }}

        .hero {{
            background: linear-gradient(135deg, {COLORS['navy']} 0%, {COLORS['navy_light']} 100%);
            border-radius: 12px;
            padding: 2rem 2.25rem;
            margin-bottom: 1.5rem;
            color: #EDF1F7;
        }}
        .hero h1 {{ color: white !important; margin-bottom: 0.3rem; }}
        .hero p {{ color: #C3CCDC; margin: 0; font-size: 0.95rem; }}

        .step-nav-item {{
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            margin-bottom: 0.25rem;
            font-size: 0.9rem;
        }}
        .step-nav-active {{
            background: rgba(255,255,255,0.12);
            font-weight: 600;
        }}
        .step-nav-done {{ opacity: 0.65; }}
        .step-nav-pending {{ opacity: 0.4; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Workforce Risk Predictor</h1>
            <p>Upload employee data → predict attrition risk → get an AI-generated retention strategy, grounded in your numbers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_nav(current_step: int):
    """Sidebar step tracker. current_step is 1-4."""
    steps = ["Upload data", "Run predictions", "Company context", "Get recommendations"]
    st.sidebar.markdown("### Workflow")
    for i, label in enumerate(steps, start=1):
        if i < current_step:
            cls, mark = "step-nav-done", "✓"
        elif i == current_step:
            cls, mark = "step-nav-active", "→"
        else:
            cls, mark = "step-nav-pending", "○"
        st.sidebar.markdown(
            f'<div class="step-nav-item {cls}">{mark} &nbsp; {i}. {label}</div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")


def risk_gauge_svg(score_pct: float, size: int = 220) -> str:
    """
    Renders a radial gauge (0-100) as inline SVG — the app's signature visual.
    score_pct: 0-100, the overall workforce risk score.
    """
    import math

    tier = "low" if score_pct < 30 else ("medium" if score_pct < 50 else "high")
    color = COLORS[tier]

    cx, cy, r = size / 2, size / 2, size / 2 - 18
    start_angle, end_angle = 135, 405  # 270-degree arc

    def polar(cx, cy, r, angle_deg):
        angle_rad = math.radians(angle_deg)
        return cx + r * math.cos(angle_rad), cy + r * math.sin(angle_rad)

    def arc_path(cx, cy, r, a0, a1):
        x0, y0 = polar(cx, cy, r, a0)
        x1, y1 = polar(cx, cy, r, a1)
        large_arc = 1 if (a1 - a0) > 180 else 0
        return f"M {x0:.2f} {y0:.2f} A {r} {r} 0 {large_arc} 1 {x1:.2f} {y1:.2f}"

    track_path = arc_path(cx, cy, r, start_angle, end_angle)
    value_angle = start_angle + (end_angle - start_angle) * min(score_pct, 100) / 100
    value_path = arc_path(cx, cy, r, start_angle, value_angle)

    svg = f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
        <path d="{track_path}" fill="none" stroke="#E5E7EB" stroke-width="16" stroke-linecap="round"/>
        <path d="{value_path}" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"/>
        <text x="{cx}" y="{cy - 4}" text-anchor="middle" font-family="IBM Plex Mono, monospace"
              font-size="34" font-weight="600" fill="{COLORS['navy']}">{score_pct:.0f}%</text>
        <text x="{cx}" y="{cy + 22}" text-anchor="middle" font-family="Inter, sans-serif"
              font-size="12" fill="{COLORS['text_muted']}">workforce risk score</text>
    </svg>
    """
    return svg
