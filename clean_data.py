"""
HR Attrition — Data Cleaning Module
=====================================
Refactored from Data_Cleaning.ipynb into a reusable function so it can be
called from the Streamlit app on any freshly uploaded raw HR export.

Usage:
    from clean_data import clean_hr_data
    df_clean = clean_hr_data(df_raw)
"""

import re
import numpy as np
import pandas as pd

# Columns that get renamed to be Orange-safe / consistent (no spaces, parens, hyphens)
RENAME_MAP = {
    'Performance Score': 'PerformanceScore',
    'Current Employee Rating': 'CurrentEmployeeRating',
    'Engagement Score': 'EngagementScore',
    'Satisfaction Score': 'SatisfactionScore',
    'Work-Life Balance Score': 'WorkLifeBalanceScore',
    'Training Program Name': 'TrainingProgramName',
    'Training Type': 'TrainingType',
    'Training Outcome': 'TrainingOutcome',
    'Training Duration(Days)': 'TrainingDurationDays',
    'Training Cost': 'TrainingCost',
}

# Statuses that count as attrition (voluntary/involuntary termination)
ATTRITION_STATUSES = ['Voluntarily Terminated', 'Terminated for Cause']

# Columns dropped as IDs / PII / leakage / high-cardinality / raw-dates-already-derived
DROP_COLS = [
    'Unnamed: 0', 'FirstName', 'LastName', 'Supervisor', 'ADEmail', 'Employee ID',
    'ExitDate', 'TerminationType', 'TerminationDescription', 'EmployeeStatus',
    'DOB', 'StartDate', 'Survey Date', 'Training Date',
    'LocationCode', 'Location', 'Trainer', 'JobFunctionDescription',
]

# The 22 feature columns the model was trained on (order doesn't matter for
# the sklearn pipeline, but this is the exact set Orange's Select Columns used)
FEATURE_COLUMNS = [
    'Title', 'BusinessUnit', 'EmployeeType', 'PayZone', 'EmployeeClassificationType',
    'DepartmentType', 'Division', 'State', 'GenderCode', 'RaceDesc', 'MaritalDesc',
    'PerformanceScore', 'CurrentEmployeeRating', 'EngagementScore', 'SatisfactionScore',
    'WorkLifeBalanceScore', 'TrainingProgramName', 'TrainingType', 'TrainingOutcome',
    'TrainingDurationDays', 'TrainingCost', 'Age', 'TenureDays',
]

TARGET_COLUMN = 'Attrition'

NUMERIC_OUTLIER_COLS = [
    'CurrentEmployeeRating', 'EngagementScore', 'SatisfactionScore',
    'WorkLifeBalanceScore', 'TrainingDurationDays', 'TrainingCost',
    'Age', 'TenureDays',
]


def clean_hr_data(df: pd.DataFrame, has_target: bool = True) -> pd.DataFrame:
    """
    Clean a raw HR export into model-ready shape.

    Parameters
    ----------
    df : pd.DataFrame
        Raw HR data, expected to look like the messy export this was built for
        (same column names as Messy_HR_Dataset_Capstone.csv).
    has_target : bool
        If True, builds the Attrition target from EmployeeStatus (use this for
        training/backtesting data that already has an outcome).
        If False, skips target creation (use this for new employees whose
        outcome hasn't happened yet — this is what the app will use for
        predictions on freshly uploaded batches).

    Returns
    -------
    pd.DataFrame
        Cleaned data containing FEATURE_COLUMNS (+ Attrition if has_target=True).
    """
    df = df.copy()

    # --- Step 1: drop exact duplicate rows ---
    df = df.drop_duplicates().reset_index(drop=True)

    # --- Step 2: Orange-safe column renames ---
    df = df.rename(columns=RENAME_MAP)

    # --- Step 3: target variable (only if EmployeeStatus is present) ---
    if has_target:
        if 'EmployeeStatus' not in df.columns:
            raise ValueError(
                "has_target=True but no 'EmployeeStatus' column found — "
                "pass has_target=False for data with no known outcome yet."
            )
        df['EmployeeStatus'] = df['EmployeeStatus'].astype(str).str.strip()
        df[TARGET_COLUMN] = np.where(
            df['EmployeeStatus'].isin(ATTRITION_STATUSES), 'Yes', 'No'
        )

    # --- Step 4: strip whitespace from all text columns ---
    text_cols = df.select_dtypes(include='object').columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()

    # --- Step 5: parse dates -> Age & TenureDays ---
    for c in ['StartDate', 'ExitDate', 'DOB']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format='mixed', dayfirst=True, errors='coerce')

    ref_candidates = [c for c in ['StartDate', 'ExitDate'] if c in df.columns]
    if ref_candidates:
        ref_date = df[ref_candidates].max().max()
    else:
        ref_date = pd.Timestamp.today()

    if 'DOB' in df.columns:
        df['Age'] = (ref_date - df['DOB']).dt.days / 365.25
    if 'StartDate' in df.columns:
        exit_filled = df['ExitDate'].fillna(ref_date) if 'ExitDate' in df.columns else ref_date
        df['TenureDays'] = (exit_filled - df['StartDate']).dt.days

    # --- Step 6: drop ID / leakage / high-cardinality / raw-date columns ---
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # --- Step 7: keep only what the model needs (+ target if present) ---
    keep_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing required columns after cleaning: {missing}. "
            "Check that the upload matches the expected HR export format."
        )

    final_cols = keep_cols + ([TARGET_COLUMN] if has_target and TARGET_COLUMN in df.columns else [])
    df = df[final_cols]

    # --- Step 8: sanity-check for remaining missing values in feature columns ---
    na_counts = df[keep_cols].isna().sum()
    if na_counts.sum() > 0:
        # Fill numeric with median, categorical with mode — keeps the pipeline
        # robust to small amounts of missingness in fresh uploads rather than
        # failing outright.
        for c in keep_cols:
            if na_counts[c] > 0:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
                else:
                    mode = df[c].mode(dropna=True)
                    df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else 'Unknown')

    return df.reset_index(drop=True)


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Informational IQR-based outlier flags — mirrors notebook Step 5. Returns a summary table."""
    rows = []
    for c in NUMERIC_OUTLIER_COLS:
        if c not in df.columns:
            continue
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_flagged = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({'column': c, 'lower_bound': round(lo, 1), 'upper_bound': round(hi, 1), 'flagged': n_flagged})
    return pd.DataFrame(rows)
