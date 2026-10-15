"""
HR Attrition — Prediction Module
===================================
Takes a raw (uncleaned) HR upload with no known outcome yet, cleans it,
and scores it with the trained model.
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clean_data import clean_hr_data, FEATURE_COLUMNS

MODEL_PATH = Path(__file__).parent.parent / 'models' / 'attrition_model.pkl'

_artifact_cache = None


def load_artifact():
    global _artifact_cache
    if _artifact_cache is None:
        _artifact_cache = joblib.load(MODEL_PATH)
    return _artifact_cache


def predict_attrition(raw_df: pd.DataFrame, id_columns: list = None) -> pd.DataFrame:
    """
    Clean a raw HR upload and predict attrition risk for each employee.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw HR export (same format as Messy_HR_Dataset_Capstone.csv). Does NOT
        need an EmployeeStatus/outcome column — this is for employees whose
        future is being predicted.
    id_columns : list[str], optional
        Columns to carry through for display purposes (e.g. ['FirstName',
        'LastName', 'Title']) even though they aren't model features. If None,
        no extra identifying columns are attached.

    Returns
    -------
    pd.DataFrame
        The cleaned feature rows plus 'attrition_probability' (0-1) and
        'predicted_attrition' ('Yes'/'No') columns.
    """
    artifact = load_artifact()
    pipeline = artifact['pipeline']
    threshold = artifact['threshold']

    # Keep a copy of any id columns present, aligned by original row order
    id_data = None
    if id_columns:
        present = [c for c in id_columns if c in raw_df.columns]
        if present:
            id_data = raw_df[present].reset_index(drop=True)

    cleaned = clean_hr_data(raw_df, has_target=False)

    X = cleaned[FEATURE_COLUMNS]
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold)

    result = cleaned.copy()
    result['attrition_probability'] = proba.round(4)
    result['predicted_attrition'] = ['Yes' if p else 'No' for p in pred]

    if id_data is not None and len(id_data) == len(result):
        for c in id_data.columns:
            if c in result.columns:
                # Already present as a model feature (e.g. 'Title') — skip,
                # it's already visible in the output.
                continue
            result.insert(0, c, id_data[c])

    return result.sort_values('attrition_probability', ascending=False).reset_index(drop=True)


def summarize_predictions(result_df: pd.DataFrame) -> dict:
    """Aggregate stats for the prompt builder / dashboard display."""
    n_total = len(result_df)
    n_at_risk = int((result_df['predicted_attrition'] == 'Yes').sum())
    high_risk = result_df[result_df['attrition_probability'] >= 0.5]
    med_risk = result_df[(result_df['attrition_probability'] >= 0.3) & (result_df['attrition_probability'] < 0.5)]

    summary = {
        'n_total_employees': n_total,
        'n_predicted_leave': n_at_risk,
        'pct_predicted_leave': round(n_at_risk / n_total * 100, 1) if n_total else 0,
        'avg_attrition_probability': round(float(result_df['attrition_probability'].mean()), 3),
        'n_high_risk_50pct+': int(len(high_risk)),
        'n_medium_risk_30_50pct': int(len(med_risk)),
    }

    # Risk breakdown by department, if present
    if 'DepartmentType' in result_df.columns:
        dept_risk = (
            result_df.groupby('DepartmentType')['attrition_probability']
            .mean()
            .round(3)
            .sort_values(ascending=False)
            .to_dict()
        )
        summary['avg_risk_by_department'] = dept_risk

    if 'EngagementScore' in result_df.columns:
        summary['avg_engagement_score'] = round(float(result_df['EngagementScore'].mean()), 2)
    if 'SatisfactionScore' in result_df.columns:
        summary['avg_satisfaction_score'] = round(float(result_df['SatisfactionScore'].mean()), 2)
    if 'WorkLifeBalanceScore' in result_df.columns:
        summary['avg_worklife_balance_score'] = round(float(result_df['WorkLifeBalanceScore'].mean()), 2)

    return summary
