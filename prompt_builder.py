"""
HR Attrition — Prompt Builder
================================
Combines model predictions + summary stats + user-provided company context
into a single, well-structured prompt for Gemini to generate consulting-style
recommendations.
"""

import json
import pandas as pd


def build_recommendation_prompt(
    result_df: pd.DataFrame,
    summary: dict,
    company_context: dict,
) -> str:
    """
    Build the prompt sent to Gemini for the final consulting narrative.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of predict.predict_attrition() — per-employee predictions.
    summary : dict
        Output of predict.summarize_predictions() — aggregate stats.
    company_context : dict
        Free-form answers the user gave in the Streamlit form, e.g.:
        {
            "industry": "Telecom",
            "recent_policy_changes": "Return-to-office mandate started in June",
            "known_pain_points": "Manager turnover in Production dept",
            "budget_constraints": "Limited budget for raises this year",
            "priority_focus": "Reduce attrition in Production department",
        }

    Returns
    -------
    str
        The full prompt text.
    """
    # Top 10 highest-risk employees (anonymized-ish — no names needed for the
    # model itself, but if id columns like FirstName were attached, keep them
    # out of the LLM prompt unless the user explicitly wants named recs).
    display_cols = [
        c for c in result_df.columns
        if c not in ('FirstName', 'LastName', 'ADEmail', 'Employee ID')
    ]
    top_risk = result_df[display_cols].head(10)

    context_lines = "\n".join(f"- **{k.replace('_', ' ').title()}**: {v}" for k, v in company_context.items() if v)

    prompt = f"""You are an experienced HR/People Analytics consultant. You have just received the output of a predictive attrition model run on a batch of employees, along with some company context. Your job is to turn this into a clear, data-backed, consulting-style set of insights and recommendations for HR leadership.

## Model Output Summary
```json
{json.dumps(summary, indent=2, default=str)}
```

## Top 10 Highest-Risk Employees (by predicted attrition probability)
```
{top_risk.to_string(index=False)}
```

## Company Context (provided by the user)
{context_lines if context_lines else "(No additional context provided.)"}

## Your Task
Write a consulting-style report with the following sections:

1. **Executive Summary** (3-5 sentences): What's the headline finding? How serious is the attrition risk right now?
2. **Key Risk Drivers**: Based on the data (department breakdown, engagement/satisfaction scores, tenure patterns, etc.), what factors seem most associated with predicted attrition in this batch? Be specific and cite the numbers.
3. **Department/Segment Focus**: Which groups need the most urgent attention, and why?
4. **Recommended Actions**: 4-6 concrete, prioritized recommendations. For each, note whether it's a quick win or a longer-term structural change, and connect it to the company context provided (e.g. if there's a budget constraint, favor low-cost interventions; if a policy change was mentioned, discuss whether it may be contributing).
5. **Caveats**: Briefly note the model's limitations (e.g. it's a baseline logistic regression, sample size, no causal claims) so leadership doesn't over-index on the predictions as certainty.

Keep the tone professional, direct, and grounded in the actual numbers provided — avoid generic HR platitudes. Where you make a claim, tie it back to a specific number from the data above.
"""
    return prompt
