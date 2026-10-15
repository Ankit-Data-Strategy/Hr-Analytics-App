"""
Workforce Risk Predictor & Consulting Assistant
==================================================
Upload a raw HR dataset -> clean it -> predict attrition risk ->
answer a few company-context questions -> get an AI-generated,
data-backed consulting report.

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from clean_data import clean_hr_data, flag_outliers
from predict import predict_attrition, summarize_predictions
from prompt_builder import build_recommendation_prompt
from gemini_api import generate_recommendations
from ui_style import (
    COLORS, inject_base_styles, render_hero, render_step_nav,
    risk_gauge_svg, risk_tier, risk_label,
)

st.set_page_config(page_title="Workforce Risk Predictor", page_icon="📊", layout="wide")
inject_base_styles()

# Determine current step for the sidebar tracker
current_step = 1
if "result_df" in st.session_state:
    current_step = 2
if "company_context" in st.session_state:
    current_step = 3
if "report" in st.session_state:
    current_step = 4

render_step_nav(current_step)

st.sidebar.markdown("### API Key")
api_key = st.sidebar.text_input(
    "Google Gemini API Key", type="password",
    help="Get a free key at aistudio.google.com/apikey — used only for this session.",
    label_visibility="collapsed",
)
st.sidebar.caption("Never saved to disk — sent directly to the Gemini API.")

render_hero()

# ---------------------------------------------------------------------------
# Step 1 — Upload
# ---------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<span class="step-badge">1</span>**Upload employee data**', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Raw HR CSV — same column format as the training data (Title, BusinessUnit, "
    "DepartmentType, EngagementScore, SatisfactionScore, StartDate, DOB, etc.)",
    type=["csv"], label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.caption(f"Loaded **{len(raw_df)}** rows, **{len(raw_df.columns)}** columns.")
with st.expander("Preview raw data"):
    st.dataframe(raw_df.head(20), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Step 2 — Clean + Predict
# ---------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<span class="step-badge">2</span>**Run predictions**', unsafe_allow_html=True)

id_cols_available = [c for c in ["FirstName", "LastName", "Title", "Employee ID"] if c in raw_df.columns]
show_names = st.checkbox(
    "Show employee names in results (names are never sent to Gemini either way)",
    value=True if id_cols_available else False,
)

if st.button("Run attrition predictions", type="primary"):
    try:
        with st.spinner("Cleaning data and running the model..."):
            id_cols = id_cols_available if show_names else []
            result_df = predict_attrition(raw_df, id_columns=id_cols)
            summary = summarize_predictions(result_df)
            outliers = flag_outliers(clean_hr_data(raw_df, has_target=False))
        st.session_state["result_df"] = result_df
        st.session_state["summary"] = summary
        st.session_state["outliers"] = outliers
        st.rerun()
    except ValueError as e:
        st.error(f"Couldn't process this file: {e}")
        st.stop()

st.markdown('</div>', unsafe_allow_html=True)

if "result_df" not in st.session_state:
    st.stop()

result_df = st.session_state["result_df"]
summary = st.session_state["summary"]

# --- Signature visual: risk gauge + key metrics ---
st.markdown('<div class="card">', unsafe_allow_html=True)
gcol, mcol = st.columns([1, 2])
with gcol:
    st.markdown(
        f'<div style="display:flex; justify-content:center;">{risk_gauge_svg(summary["avg_attrition_probability"] * 100)}</div>',
        unsafe_allow_html=True,
    )
with mcol:
    m1, m2 = st.columns(2)
    m1.metric("Total employees", summary["n_total_employees"])
    m2.metric("Predicted to leave", summary["n_predicted_leave"], f"{summary['pct_predicted_leave']}%")
    m3, m4 = st.columns(2)
    m3.metric("High risk (≥50%)", summary["n_high_risk_50pct+"])
    m4.metric("Medium risk (30-50%)", summary["n_medium_risk_30_50pct"])
st.markdown('</div>', unsafe_allow_html=True)

# --- Results table with risk badges ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("**Prediction results**")

display_df = result_df.copy()
display_df["risk_level"] = display_df["attrition_probability"].apply(risk_label)
cols_order = [c for c in display_df.columns if c not in ("FirstName", "LastName")] if not show_names else display_df.columns.tolist()

st.dataframe(
    display_df[cols_order],
    use_container_width=True,
    height=400,
    column_config={
        "attrition_probability": st.column_config.ProgressColumn(
            "Attrition probability", min_value=0, max_value=1, format="%.2f",
        ),
        "risk_level": st.column_config.TextColumn("Risk level"),
    },
)

csv_out = result_df.to_csv(index=False).encode("utf-8")
st.download_button("Download full results as CSV", csv_out, "attrition_predictions.csv", "text/csv")
st.markdown('</div>', unsafe_allow_html=True)

# --- Department risk chart, color-coded by tier ---
if "avg_risk_by_department" in summary:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Average risk by department**")
    dept_items = sorted(summary["avg_risk_by_department"].items(), key=lambda x: x[1])
    depts = [d[0] for d in dept_items]
    vals = [d[1] for d in dept_items]
    bar_colors = [COLORS[risk_tier(v)] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=depts, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.0%}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        height=max(220, 40 * len(depts)),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(tickformat=".0%", range=[0, max(vals) * 1.3]),
        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Data quality notes (outlier flags — informational only, nothing removed)"):
    st.dataframe(st.session_state["outliers"], use_container_width=True)

# ---------------------------------------------------------------------------
# Step 3 — Company context
# ---------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<span class="step-badge">3</span>**Add company context**', unsafe_allow_html=True)
st.caption("This helps Gemini tailor recommendations to your actual situation rather than giving generic advice.")

with st.form("company_context_form"):
    c1, c2 = st.columns(2)
    with c1:
        industry = st.text_input("Industry / business type", placeholder="e.g. Telecom infrastructure")
        recent_policy_changes = st.text_area(
            "Recent policy changes",
            placeholder="e.g. Return-to-office mandate started in June; pay freeze this year",
        )
        known_pain_points = st.text_area(
            "Known pain points / anecdotal issues",
            placeholder="e.g. High manager turnover in Production; complaints about scheduling",
        )
    with c2:
        budget_constraints = st.text_area(
            "Budget constraints",
            placeholder="e.g. Limited budget for raises; can fund small perks/programs only",
        )
        priority_focus = st.text_area(
            "Priority focus area",
            placeholder="e.g. Reduce attrition specifically in the Production department",
        )
        additional_notes = st.text_area("Anything else worth knowing?", placeholder="Optional")

    submitted = st.form_submit_button("Save context")
    if submitted:
        st.session_state["company_context"] = {
            "industry": industry,
            "recent_policy_changes": recent_policy_changes,
            "known_pain_points": known_pain_points,
            "budget_constraints": budget_constraints,
            "priority_focus": priority_focus,
            "additional_notes": additional_notes,
        }
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Step 4 — Generate recommendations
# ---------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<span class="step-badge">4</span>**Generate consulting recommendations**', unsafe_allow_html=True)

if "company_context" not in st.session_state:
    st.info("Fill out and save the company context form above first.")
else:
    if st.button("Generate AI recommendations", type="primary"):
        if not api_key:
            st.error("Enter your Google Gemini API key in the sidebar first.")
        else:
            with st.spinner("Asking Gemini for a data-backed recommendation report..."):
                try:
                    prompt = build_recommendation_prompt(result_df, summary, st.session_state["company_context"])
                    report = generate_recommendations(prompt, api_key=api_key)
                    st.session_state["report"] = report
                    st.rerun()
                except Exception as e:
                    st.error(f"Something went wrong calling the Gemini API: {e}")

    if "report" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["report"])
        st.download_button(
            "Download report as Markdown",
            st.session_state["report"].encode("utf-8"),
            "workforce_risk_report.md", "text/markdown",
        )
st.markdown('</div>', unsafe_allow_html=True)
