"""
Train the HR Attrition model — a scikit-learn replica of the Orange workflow:

    Logistic Regression (C=1.0, L2 penalty)
        -> Calibrated Learner (Sigmoid/Platt calibration)
        -> F1-optimized decision threshold

Saves a single artifact (models/attrition_model.pkl) containing:
    - the fitted sklearn Pipeline (preprocessing + calibrated classifier)
    - the F1-optimal probability threshold
    - the feature column list (for input validation)
    - basic evaluation metrics (for reference / sanity-checking)
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clean_data import FEATURE_COLUMNS, TARGET_COLUMN

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
)

CATEGORICAL_COLS = [
    'Title', 'BusinessUnit', 'EmployeeType', 'PayZone', 'EmployeeClassificationType',
    'DepartmentType', 'Division', 'State', 'GenderCode', 'RaceDesc', 'MaritalDesc',
    'PerformanceScore', 'TrainingProgramName', 'TrainingType', 'TrainingOutcome',
]
NUMERIC_COLS = [
    'CurrentEmployeeRating', 'EngagementScore', 'SatisfactionScore',
    'WorkLifeBalanceScore', 'TrainingDurationDays', 'TrainingCost', 'Age', 'TenureDays',
]

assert set(CATEGORICAL_COLS + NUMERIC_COLS) == set(FEATURE_COLUMNS), "Column list mismatch!"


def build_pipeline() -> Pipeline:
    """Preprocessing (one-hot for categoricals, passthrough numerics) + calibrated LR."""
    preprocess = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
            # Orange continuizes/normalizes numeric inputs internally before fitting;
            # StandardScaler replicates that so the solver converges and no single
            # large-magnitude feature (e.g. TrainingCost) dominates the coefficients.
            ('num', StandardScaler(), NUMERIC_COLS),
        ]
    )
    base_lr = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, class_weight=None)
    # Orange's Calibrated Learner default = Sigmoid (Platt) calibration
    calibrated = CalibratedClassifierCV(base_lr, method='sigmoid', cv=5)
    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('model', calibrated),
    ])
    return pipeline


def find_best_f1_threshold(y_true, y_proba) -> float:
    """Scan thresholds and return the one maximizing F1 — mirrors Orange's
    Calibrated Learner 'Optimize threshold (F1 score)' behavior."""
    thresholds = np.linspace(0.01, 0.99, 197)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def main():
    data_path = Path(__file__).parent.parent / 'data' / 'Cleaned_data.csv'
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS]
    y = (df[TARGET_COLUMN] == 'Yes').astype(int)  # 1 = attrition, 0 = stayed

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Find optimal F1 threshold using cross-validated probabilities on the training set
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    threshold = find_best_f1_threshold(y_train, train_proba)

    # Evaluate on held-out test set
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)

    metrics = {
        'threshold': threshold,
        'accuracy': float((test_preds == y_test).mean()),
        'precision': float(precision_score(y_test, test_preds, zero_division=0)),
        'recall': float(recall_score(y_test, test_preds, zero_division=0)),
        'f1': float(f1_score(y_test, test_preds, zero_division=0)),
        'auc': float(roc_auc_score(y_test, test_proba)),
        'confusion_matrix': confusion_matrix(y_test, test_preds).tolist(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'attrition_rate': float(y.mean()),
    }

    print("=== Test Set Performance ===")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v}")
    print("  confusion_matrix [[TN, FP], [FN, TP]]:", metrics['confusion_matrix'])
    print("\n" + classification_report(y_test, test_preds, target_names=['No', 'Yes']))

    # Refit on ALL data for the final production model (standard practice once validated)
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    final_proba = final_pipeline.predict_proba(X)[:, 1]
    final_threshold = find_best_f1_threshold(y, final_proba)

    artifact = {
        'pipeline': final_pipeline,
        'threshold': final_threshold,
        'feature_columns': FEATURE_COLUMNS,
        'categorical_cols': CATEGORICAL_COLS,
        'numeric_cols': NUMERIC_COLS,
        'holdout_metrics': metrics,  # kept for reference/display in the app
    }

    out_path = Path(__file__).parent.parent / 'models' / 'attrition_model.pkl'
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"\nSaved model to {out_path}")
    print(f"Final threshold (fit on full data): {final_threshold:.3f}")


if __name__ == '__main__':
    main()
