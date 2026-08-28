# =============================================================================
# train_xgboost.py
#
# PURPOSE:
#   Train an XGBoost classifier on the same labeled dataset used for
#   Random Forest, then compare the two models side by side.
#
# HOW TO RUN (from the backend/ folder with venv active):
#   python app/ml/train_xgboost.py
#
# INPUT:
#   backend/data/processed/labeled_dataset.csv
#
# OUTPUT:
#   backend/models/xgboost_forecast_5s.pkl ← saved trained model (predicts 5s-ahead state)
#   backend/models/comparison_report.json  ← side-by-side metrics file (baseline + RF + XGB)
#
# WHAT IS XGBOOST?
#   XGBoost (Extreme Gradient Boosting) builds trees SEQUENTIALLY.
#   Each new tree focuses on the mistakes made by all previous trees —
#   this is called "boosting". It is often more accurate than Random Forest
#   on tabular data because it keeps correcting its own errors.
#
#   KEY DIFFERENCE FROM RANDOM FOREST:
#   - Random Forest: many trees built IN PARALLEL on random data subsets.
#                    Trees are independent. Final answer = majority vote.
#   - XGBoost:       trees built ONE AFTER ANOTHER. Each tree fixes the
#                    errors of the last. Final answer = weighted sum.
# =============================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

# Enable pandas Copy-on-Write mode. Without this, pandas <3.0 defaults to
# eagerly DEEP-COPYING the entire DataFrame inside routine calls like
# rename()/drop()/set_index() (see pandas.core.generic._rename's internal
# `self.copy(deep=copy and not using_copy_on_write())`). On a small sample
# dataset that's invisible; on a real O-RAN recording with tens of millions
# of rows, EVERY such call briefly allocates a full extra multi-GB copy and
# can crash with numpy.core._exceptions._ArrayMemoryError. Copy-on-Write
# makes these operations cheap (share memory until an actual write happens)
# without changing any pipeline behaviour.
pd.set_option("mode.copy_on_write", True)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from forecast_config import (
    NON_FEATURE_COLS, CURRENT_LABEL_COL, FUTURE_LABEL_COL,
    FORECAST_HORIZON_SECONDS,
)
from forecast_utils import (
    naive_baseline_metrics, transition_report, upsert_comparison_report,
    downcast_numeric_dtypes,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

LABELED_DATA_PATH      = os.path.join("data", "processed", "labeled_dataset.csv")
# Both renamed to the "_forecast_5s" convention — these models now target
# degradation_risk_future, not the same-instant label (section 28).
RF_MODEL_PATH          = os.path.join("models", "random_forest_forecast_5s.pkl")
XGB_MODEL_SAVE_PATH    = os.path.join("models", "xgboost_forecast_5s.pkl")
COMPARISON_REPORT_PATH = os.path.join("models", "comparison_report.json")

# XGBoost hyperparameters
N_ESTIMATORS   = 100    # number of boosting rounds (trees)
MAX_DEPTH      = 4      # depth of each tree — shallower than RF to reduce overfitting
LEARNING_RATE  = 0.1    # how much each tree corrects the previous (step size)
RANDOM_STATE   = 42
TEST_SIZE      = 0.2    # must match Random Forest split exactly for fair comparison


# =============================================================================
# HELPER
# =============================================================================

def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
# STEP 1 — Load data (identical to Random Forest)
# =============================================================================

def load_data(path):
    print_section("STEP 1 — Loading labeled dataset")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nERROR: {path} not found.\n"
            "Run label_dataset.py first."
        )

    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)

    # Re-downcast — pandas.read_csv re-infers float64/int64 from the CSV
    # text regardless of what dtypes were used when it was written.
    df = downcast_numeric_dtypes(df)

    print(f"  Rows: {df.shape[0]}  Cols: {df.shape[1]}")
    return df


# =============================================================================
# STEP 2 — Prepare X and y (same as Random Forest)
# =============================================================================

def prepare_features_and_label(df):
    print_section("STEP 2 — Preparing features (X) and label (y)")

    print(f"  Target column: '{FUTURE_LABEL_COL}' "
          f"(degradation state ~{FORECAST_HORIZON_SECONDS}s ahead — NOT the same-instant label)")

    columns_to_exclude = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=columns_to_exclude)
    y = df[FUTURE_LABEL_COL]

    print(f"  Feature matrix X: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Normal   (0): {(y == 0).sum()}")
    print(f"  Degraded (1): {(y == 1).sum()}")
    return X, y


# =============================================================================
# STEP 3 — Train/test split (identical settings to Random Forest)
# =============================================================================

def split_data(X, y):
    print_section("STEP 3 — Splitting data (80% train / 20% test)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=False,        # preserve time order — no data leakage
        random_state=RANDOM_STATE,
    )

    print(f"  Training set: {X_train.shape[0]} rows")
    print(f"  Test set:     {X_test.shape[0]} rows")
    print(f"  shuffle=False → time order preserved (matches RF split exactly)")
    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 4 — Train XGBoost
# =============================================================================

def train_xgboost(X_train, y_train):
    """
    Train the XGBoost classifier.

    KEY PARAMETERS EXPLAINED:

    n_estimators=100
        How many boosting rounds to run (how many trees to build).
        Same as Random Forest so comparison is fair.

    max_depth=4
        Maximum depth of each tree. XGBoost trees are usually kept
        shallower than Random Forest trees because boosting already
        corrects errors — deep trees would overfit.

    learning_rate=0.1
        Also called 'eta'. Controls how much each tree shrinks the
        correction before applying it. Lower = more conservative,
        needs more trees. 0.1 is a standard default.

    scale_pos_weight
        XGBoost's equivalent of class_weight='balanced'.
        We set it to n_negative / n_positive so the minority class
        (Degraded) gets proportionally more influence.
        Formula: count(Normal) / count(Degraded) = 1143 / 529 ≈ 2.16

    use_label_encoder=False, eval_metric='logloss'
        Suppress a deprecation warning in newer XGBoost versions.
        logloss is the standard loss function for binary classification.
    """
    print_section("STEP 4 — Training XGBoost model")

    # Calculate class weight ratio
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    print(f"\n  Model settings:")
    print(f"    n_estimators     = {N_ESTIMATORS}  (boosting rounds)")
    print(f"    max_depth        = {MAX_DEPTH}   (tree depth — shallower than RF)")
    print(f"    learning_rate    = {LEARNING_RATE}  (step size per round)")
    print(f"    scale_pos_weight = {scale_pos_weight:.2f} (handles class imbalance)")
    print(f"    random_state     = {RANDOM_STATE}")

    model = XGBClassifier(
        n_estimators     = N_ESTIMATORS,
        max_depth        = MAX_DEPTH,
        learning_rate    = LEARNING_RATE,
        scale_pos_weight = scale_pos_weight,
        random_state     = RANDOM_STATE,
        eval_metric      = "logloss",
        verbosity        = 0,        # suppress training output
    )

    print(f"\n  Training on {X_train.shape[0]} rows...")
    model.fit(X_train, y_train)
    print("  ✓ Training complete.")
    return model


# =============================================================================
# STEP 5 — Evaluate XGBoost
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model and return a metrics dictionary.
    Same evaluation code used for both RF and XGBoost for consistency.
    """
    print_section(f"STEP 5 — Evaluating {model_name}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  {model_name:<31}│")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │  Accuracy   : {accuracy:>8.4f}  ({accuracy*100:.2f}%)  │")
    print(f"  │  Precision  : {precision:>8.4f}  ({precision*100:.2f}%)  │")
    print(f"  │  Recall     : {recall:>8.4f}  ({recall*100:.2f}%)  │")
    print(f"  │  F1 Score   : {f1:>8.4f}  ({f1*100:.2f}%)  │")
    print(f"  └─────────────────────────────────┘")

    print(f"\n  Confusion Matrix:")
    print(f"                   Predicted Normal  Predicted Degraded")
    print(f"  Actual Normal  (0)     {tn:>6}             {fp:>6}")
    print(f"  Actual Degraded(1)     {fn:>6}             {tp:>6}")

    print(f"\n  Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Degraded"]))

    return {
        "model":     model_name,
        "accuracy":  round(float(accuracy),  4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "f1_score":  round(float(f1),        4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# =============================================================================
# STEP 6 — Feature importances
# =============================================================================

def print_feature_importances(model, feature_names):
    print_section("STEP 6 — XGBoost feature importances")

    importances   = model.feature_importances_
    sorted_idx    = np.argsort(importances)[::-1]

    print(f"\n  {'Rank':<6} {'Feature':<35} {'Importance':>12}  Bar")
    print(f"  {'-'*6} {'-'*35} {'-'*12}  ---")

    for rank, idx in enumerate(sorted_idx, start=1):
        feature = feature_names[idx]
        score   = importances[idx]
        bar     = "█" * int(score * 200)
        print(f"  {rank:<6} {feature:<35} {score:>12.4f}  {bar}")

    top3 = [feature_names[i] for i in sorted_idx[:3]]
    print(f"\n  Top 3 most important KPIs: {', '.join(top3)}")


# =============================================================================
# STEP 7 — Side-by-side comparison with Random Forest
# =============================================================================

def compare_models(xgb_metrics, rf_model_path, X_test, y_test):
    print_section("STEP 7 — Side-by-side comparison: XGBoost vs Random Forest")

    # Load the previously saved Random Forest model
    if not os.path.exists(rf_model_path):
        print("  ⚠ Random Forest model not found. Run train_random_forest.py first.")
        return xgb_metrics, None

    rf_model   = joblib.load(rf_model_path)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest (5s forecast)")

    # Print the comparison table
    print_section("COMPARISON TABLE")
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels  = ["Accuracy", "Precision", "Recall",  "F1 Score"]

    print(f"\n  {'Metric':<12} {'Random Forest':>16} {'XGBoost':>16}  Winner")
    print(f"  {'-'*12} {'-'*16} {'-'*16}  {'-'*10}")

    for metric, label in zip(metrics, labels):
        rf_val  = rf_metrics[metric]
        xgb_val = xgb_metrics[metric]
        winner  = "RF " if rf_val > xgb_val else "XGB" if xgb_val > rf_val else "TIE"
        bar_rf  = "▓" * int(rf_val  * 20)
        bar_xgb = "▓" * int(xgb_val * 20)
        print(f"  {label:<12} {rf_val:>14.4f}   {xgb_val:>14.4f}   {winner}")

    # Determine overall winner by F1
    rf_f1  = rf_metrics["f1_score"]
    xgb_f1 = xgb_metrics["f1_score"]
    print(f"\n  Overall (by F1 Score):")
    if xgb_f1 > rf_f1:
        print(f"  ✓ XGBoost wins  ({xgb_f1:.4f} vs {rf_f1:.4f})")
    elif rf_f1 > xgb_f1:
        print(f"  ✓ Random Forest wins  ({rf_f1:.4f} vs {xgb_f1:.4f})")
    else:
        print(f"  = Tie  (both F1 = {rf_f1:.4f})")

    return xgb_metrics, rf_metrics


# =============================================================================
# STEP 8 — Save model and comparison report
# =============================================================================

def save_outputs(model, xgb_metrics, rf_metrics, baseline_metrics=None):
    print_section("STEP 8 — Saving model and comparison report")

    # Save XGBoost model
    os.makedirs(os.path.dirname(XGB_MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, XGB_MODEL_SAVE_PATH)
    print(f"  ✓ XGBoost model saved: {XGB_MODEL_SAVE_PATH}")
    print(f"    Size: {os.path.getsize(XGB_MODEL_SAVE_PATH)/1024:.1f} KB")

    # Upsert into the shared comparison report instead of overwriting it —
    # train_random_forest.py may have already written its own entry (and the
    # naive baseline entry) to this same file; overwriting wholesale would
    # silently delete them.
    if baseline_metrics:
        upsert_comparison_report(COMPARISON_REPORT_PATH, baseline_metrics)
    if rf_metrics:
        upsert_comparison_report(COMPARISON_REPORT_PATH, rf_metrics)
    upsert_comparison_report(COMPARISON_REPORT_PATH, xgb_metrics)
    print(f"  ✓ Comparison report updated: {COMPARISON_REPORT_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  XGBoost Training — O-RAN KPI Prediction xApp")
    print("  FYP — ML-based KPI Prediction xApp")
    print("=" * 65)

    df = load_data(LABELED_DATA_PATH)
    X, y = prepare_features_and_label(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Naive baseline on this same test split (recomputed here too so this
    # script is runnable standalone, though train_random_forest.py already
    # writes an identical entry since both scripts use the same split logic).
    print_section("STEP 3b — Naive baseline on the held-out test set")
    # IMPORTANT: positional (iloc), not X_test.index (loc) — see the matching
    # comment in train_random_forest.py. Real multi-session data can have
    # duplicate 'timestamp' values, and df.loc[X_test.index] would silently
    # explode to every row sharing each duplicate label.
    current_test = df[CURRENT_LABEL_COL].iloc[len(y_train):]
    baseline_metrics = naive_baseline_metrics(current_test, y_test)

    xgb_model   = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (5s forecast)")
    transition_report(current_test, y_test, predicted_future=xgb_model.predict(X_test))

    print_feature_importances(xgb_model, list(X.columns))

    xgb_metrics, rf_metrics = compare_models(
        xgb_metrics, RF_MODEL_PATH, X_test, y_test
    )

    save_outputs(xgb_model, xgb_metrics, rf_metrics, baseline_metrics)

    print("\n" + "=" * 65)
    print("  ✓ XGBoost (5s forecast) training complete!")
    print(f"  XGBoost  F1: {xgb_metrics['f1_score']}")
    if rf_metrics:
        print(f"  RF       F1: {rf_metrics['f1_score']}")
    print(f"  Baseline F1: {baseline_metrics['f1_score']}")
    print(f"  Model saved: {XGB_MODEL_SAVE_PATH}")
    print("  Next step: Phase 5 — Connect both models to the dashboard.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
