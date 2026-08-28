# =============================================================================
# train_random_forest.py
#
# PURPOSE:
#   Train a Random Forest classifier to forecast network degradation
#   approximately 5 SECONDS AHEAD (see docs/FORECASTING_METHODOLOGY_UPDATE.md),
#   using the labeled O-RAN KPI dataset produced in Phase 1.
#
# HOW TO RUN (from the backend/ folder with venv active):
#   python app/ml/train_random_forest.py
#
# INPUT:
#   backend/data/processed/labeled_dataset.csv
#
# OUTPUT:
#   backend/models/random_forest_forecast_5s.pkl   ← saved trained model
#   backend/models/comparison_report.json          ← naive baseline + RF entries
#
# WHAT IS A RANDOM FOREST?
#   A Random Forest builds many Decision Trees, each trained on a slightly
#   different random subset of the data and features.
#   When predicting, every tree casts a vote, and the majority vote wins.
#   This "wisdom of the crowd" approach makes it more accurate and less
#   prone to overfitting than a single decision tree.
# =============================================================================

import os
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

from sklearn.ensemble import RandomForestClassifier
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
# All file paths and model settings are defined here in one place.
# =============================================================================

LABELED_DATA_PATH = os.path.join("data", "processed", "labeled_dataset.csv")
# Renamed from random_forest_model.pkl: this model now predicts the FUTURE
# forecasting target (degradation_risk_future), not the same-instant one —
# see docs/FORECASTING_METHODOLOGY_UPDATE.md. Kept as a distinct filename
# (section 28 of the correction brief) so the old same-instant model is not
# silently overwritten; app/utils/model_loader.py points at this new path.
MODEL_SAVE_PATH   = os.path.join("models", "random_forest_forecast_5s.pkl")
COMPARISON_REPORT_PATH = os.path.join("models", "comparison_report.json")

# --- Random Forest hyperparameters ---
# These are the "settings" that control how the model is built.
# They are explained in detail where each one is used below.
N_ESTIMATORS  = 100    # Number of trees in the forest
MAX_DEPTH     = None   # How deep each tree can grow (None = unlimited)
RANDOM_STATE  = 42     # Seed for reproducibility — same result every run
TEST_SIZE     = 0.2    # 20% of rows held back for testing
CLASS_WEIGHT  = "balanced"  # Compensates for the 69/31 class imbalance


# =============================================================================
# HELPER — Section printer
# =============================================================================

def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
# STEP 1 — Load the labeled dataset
# =============================================================================

def load_data(path):
    """
    Load the labeled dataset produced by label_dataset.py.
    The timestamp is the row index.
    """
    print_section("STEP 1 — Loading labeled dataset")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nERROR: Cannot find {path}\n"
            "Run label_dataset.py first to generate the labeled dataset."
        )

    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)

    # Re-downcast — pandas.read_csv re-infers float64/int64 from the CSV
    # text regardless of what dtypes were used when it was written.
    df = downcast_numeric_dtypes(df)

    print(f"  File:   {path}")
    print(f"  Rows:   {df.shape[0]}")
    print(f"  Cols:   {df.shape[1]}")

    return df


# =============================================================================
# STEP 2 — Separate features (X) and label (y)
# =============================================================================

def prepare_features_and_label(df):
    """
    Split the DataFrame into:
      X — the input features the model will learn from (18 columns)
      y — the target label the model will predict:
          'degradation_risk_future' — degradation state ~5 SECONDS AHEAD,
          NOT the same-instant 'degradation_risk'.

    We drop from X:
      - 'degradation_score'         — the intermediate score used to BUILD the label (cheating if included)
      - 'degradation_risk'          — the CURRENT-instant label ("label_now"). This is deliberately
                                       excluded from the forecasting model's inputs per the correction
                                       brief (section 9) — the main experiment predicts the future purely
                                       from the raw KPI features, not from a rule-based summary of "now".
      - 'degradation_risk_future'   — the target itself
      - 'session_id'                — recording/session metadata, not a radio-performance feature

    In machine learning, X is called the "feature matrix" and y is the
    "target vector". This split is standard for all supervised learning.
    """
    print_section("STEP 2 — Preparing features (X) and label (y)")

    print(f"\n  Target column: '{FUTURE_LABEL_COL}' "
          f"(degradation state ~{FORECAST_HORIZON_SECONDS}s ahead of each row's KPIs)")

    COLUMNS_TO_EXCLUDE = [c for c in NON_FEATURE_COLS if c in df.columns]

    X = df.drop(columns=COLUMNS_TO_EXCLUDE)
    y = df[FUTURE_LABEL_COL]

    print(f"\n  Feature matrix X: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"\n  Features used:")
    for i, col in enumerate(X.columns, start=1):
        print(f"    {i:>2}. {col}")

    print(f"\n  Target vector y: {y.shape[0]} values")
    normal_count   = (y == 0).sum()
    degraded_count = (y == 1).sum()
    print(f"    Normal   (0): {normal_count:>5}  ({normal_count/len(y)*100:.1f}%)")
    print(f"    Degraded (1): {degraded_count:>5}  ({degraded_count/len(y)*100:.1f}%)")

    # Safety: replace any infinity values before training.
    # With 19 million rows across many files, prb_grant_ratio or other
    # derived columns can contain np.inf which causes sklearn to crash
    # with "Input X contains infinity or a value too large for float32".
    inf_cols = [col for col in X.columns if np.isinf(X[col]).any()]
    if inf_cols:
        print(f"\n  Warning: infinity values found in: {inf_cols}")
        print(f"           Replacing with column medians...")
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in inf_cols:
            X[col] = X[col].fillna(X[col].median())
        print(f"           Done.")
    else:
        print(f"  Infinity check: OK")

    return X, y


# =============================================================================
# STEP 3 — Split into training set and test set
# =============================================================================

def split_data(X, y):
    """
    Divide the data into a training set and a test set.

    WHY SPLIT?
    If we train on all the data and then test on the same data, the model will
    appear perfect — it has already seen those rows! This is called "data leakage"
    and would give us a false sense of how good the model really is.

    By holding back 20% of the data (the test set) and never training on it,
    we get an honest estimate of how the model will perform on new, unseen data.

    IMPORTANT — WHY WE DO NOT SHUFFLE:
    This is a time-series dataset — readings are ordered by time.
    If we shuffled before splitting, a training row at time T+100 could
    end up right next to a test row at time T+99. The model would effectively
    have "seen the future" during training, making evaluation unreliable.

    By setting shuffle=False, the first 80% of time-ordered rows become
    training data, and the final 20% become the test set. This simulates
    the real use case: train on historical data, predict on future data.
    """
    print_section("STEP 3 — Splitting data into train and test sets")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=False,   # ← Critical for time-series data
        random_state=RANDOM_STATE,
    )

    print(f"  Split ratio:  {int((1-TEST_SIZE)*100)}% train  /  {int(TEST_SIZE*100)}% test")
    print(f"  shuffle=False → time order preserved (no data leakage)")
    print(f"\n  Training set: {X_train.shape[0]} rows")
    print(f"    Normal   (0): {(y_train == 0).sum()}")
    print(f"    Degraded (1): {(y_train == 1).sum()}")
    print(f"\n  Test set:     {X_test.shape[0]} rows")
    print(f"    Normal   (0): {(y_test == 0).sum()}")
    print(f"    Degraded (1): {(y_test == 1).sum()}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 4 — Train the Random Forest model
# =============================================================================

def train_model(X_train, y_train):
    """
    Create and train the Random Forest classifier.

    KEY PARAMETERS EXPLAINED:

    n_estimators=100
        The number of decision trees to build.
        More trees → more accurate, but slower to train.
        100 is a standard starting point that works well for most problems.

    max_depth=None
        How many levels deep each tree is allowed to grow.
        None means unlimited — trees grow until all leaves are pure
        (contain only one class) or contain only one sample.
        This can cause overfitting, but the ensemble averaging of 100 trees
        corrects for this.

    class_weight='balanced'
        Our dataset has 69% Normal and 31% Degraded.
        Without correction, the model might learn to just predict "Normal"
        for everything and still get 69% accuracy (this is called the
        "majority class problem").
        'balanced' automatically adjusts each class's weight so that the
        minority class (Degraded) gets proportionally more influence
        during training:
            weight = total_samples / (n_classes * samples_in_class)

    random_state=42
        A fixed seed for the random number generator.
        This ensures that if you run the script twice, you get the exact
        same model. Without it, the result would vary each run.
    """
    print_section("STEP 4 — Training the Random Forest model")

    # SCALE SAFETY: max_depth=None (unlimited) is fine on a small dataset —
    # trees stay shallow because there isn't much data to split on. On a
    # real O-RAN recording with tens of millions of rows, unlimited-depth
    # trees can grow enormous (each split node consumes memory, and depth
    # scales with training-set size), risking exactly the kind of
    # memory/runtime blowup this fix is meant to prevent. Rather than
    # silently changing MAX_DEPTH itself, we only cap it at fit time, and
    # only when the training set is large enough for "unlimited" to be a
    # real risk — small-dataset behaviour (and any explicit MAX_DEPTH
    # override) is unchanged.
    LARGE_TRAINING_SET_THRESHOLD = 500_000
    effective_max_depth = MAX_DEPTH
    if MAX_DEPTH is None and len(X_train) > LARGE_TRAINING_SET_THRESHOLD:
        effective_max_depth = 25
        print(f"\n  ⚠ Training set has {len(X_train):,} rows — capping max_depth "
              f"at {effective_max_depth} (was None/unlimited) to keep memory and "
              f"training time bounded. Override MAX_DEPTH explicitly above to "
              f"control this directly.")

    print(f"\n  Model settings:")
    print(f"    n_estimators  = {N_ESTIMATORS}   (number of trees)")
    print(f"    max_depth     = {effective_max_depth}  (None = unlimited depth)")
    print(f"    class_weight  = '{CLASS_WEIGHT}'  (handles class imbalance)")
    print(f"    random_state  = {RANDOM_STATE}   (for reproducibility)")

    # Create the model object with our chosen settings
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=effective_max_depth,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,   # Use all available CPU cores to train faster
    )

    print(f"\n  Training on {X_train.shape[0]} rows × {X_train.shape[1]} features...")
    print("  (This may take a while on a large dataset...)")

    # .fit() is where learning actually happens.
    model.fit(X_train, y_train)

    print("  ✓ Training complete.")

    return model


# =============================================================================
# STEP 5 — Evaluate the model on the test set
# =============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Measure how well the model performs on data it has NEVER seen before.

    We use four standard metrics. Each one tells us something different:

    ACCURACY  — "What percentage of ALL predictions were correct?"
                Good overall measure, but misleading when classes are imbalanced.
                e.g. 69% accuracy could mean "always guess Normal".

    PRECISION — "Of all the rows the model predicted as Degraded,
                 what fraction were actually Degraded?"
                High precision means few false alarms.
                → Important for operations teams: "Is this alert real?"

    RECALL    — "Of all the rows that WERE Degraded,
                 what fraction did the model correctly catch?"
                High recall means few missed degradation events.
                → Important for network reliability: "Did we miss anything?"

    F1 SCORE  — The harmonic mean of Precision and Recall.
                A single number that balances both concerns.
                Perfect = 1.0, Worst = 0.0.
                Best metric to compare models overall.

    CONFUSION MATRIX — A 2×2 table showing the four possible outcomes:
        ┌────────────────────────────────────────────────────┐
        │              Predicted Normal  Predicted Degraded  │
        │ Actual Normal      TN               FP             │
        │ Actual Degraded    FN               TP             │
        └────────────────────────────────────────────────────┘
        TN = True Negative  (correctly predicted Normal)
        FP = False Positive (predicted Degraded but was Normal — false alarm)
        FN = False Negative (predicted Normal but was Degraded — missed!)
        TP = True Positive  (correctly predicted Degraded)

        For network monitoring, FN is the worst outcome — a missed
        degradation event means users suffer without the system reacting.
    """
    print_section("STEP 5 — Evaluating model performance on test set")

    # Get predictions on the test set
    y_pred = model.predict(X_test)

    # Get prediction probabilities (confidence scores)
    # Column 0 = probability of Normal, Column 1 = probability of Degraded
    y_prob = model.predict_proba(X_test)

    # Calculate all four metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    # --- Print metrics ---
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │        MODEL METRICS            │")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │  Accuracy   : {accuracy:>8.4f}  ({accuracy*100:.2f}%)  │")
    print(f"  │  Precision  : {precision:>8.4f}  ({precision*100:.2f}%)  │")
    print(f"  │  Recall     : {recall:>8.4f}  ({recall*100:.2f}%)  │")
    print(f"  │  F1 Score   : {f1:>8.4f}  ({f1*100:.2f}%)  │")
    print(f"  └─────────────────────────────────┘")

    # --- Confusion matrix ---
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"  (Rows = Actual label, Columns = Predicted label)\n")
    print(f"                   Predicted    Predicted")
    print(f"                   Normal (0)  Degraded (1)")
    print(f"  Actual Normal  (0)  {tn:>6}       {fp:>6}")
    print(f"  Actual Degraded(1)  {fn:>6}       {tp:>6}")

    print(f"\n  Breaking that down:")
    print(f"    True Negatives  (TN) = {tn:>4}  → correctly predicted Normal")
    print(f"    False Positives (FP) = {fp:>4}  → false alarms (predicted Degraded, was Normal)")
    print(f"    False Negatives (FN) = {fn:>4}  → missed events (predicted Normal, was Degraded)")
    print(f"    True Positives  (TP) = {tp:>4}  → correctly predicted Degraded")

    # --- Full classification report ---
    print(f"\n  Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Degraded"]))

    # --- Interpretation ---
    print(f"  Interpretation:")
    if f1 >= 0.80:
        print(f"  ✓ F1 Score of {f1:.4f} is strong — the model is performing well.")
    elif f1 >= 0.65:
        print(f"  ~ F1 Score of {f1:.4f} is acceptable — consider tuning hyperparameters.")
    else:
        print(f"  ⚠ F1 Score of {f1:.4f} is low — review feature selection or thresholds.")

    if fn > tp:
        print(f"  ⚠ More missed events (FN={fn}) than correct detections (TP={tp}).")
        print(f"     Consider lowering the decision threshold to improve recall.")

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# =============================================================================
# STEP 6 — Print feature importances
# =============================================================================

def print_feature_importances(model, feature_names):
    """
    Show which KPI features the Random Forest relied on most.

    Random Forest can measure how useful each feature was across all 100 trees
    by tracking how much each feature reduced impurity (disorder) in the data.

    This is extremely useful for your dissertation and viva:
    - It validates that the model is using sensible features
    - It shows which KPIs are most predictive of degradation
    - It could guide future work on reducing the feature set
    """
    print_section("STEP 6 — Feature importances")

    importances = model.feature_importances_

    # Sort features from most important to least
    sorted_indices = np.argsort(importances)[::-1]

    print(f"\n  Feature importances (higher = more useful to the model):\n")
    print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12}  Bar")
    print(f"  {'-'*6} {'-'*35} {'-'*12}  ---")

    for rank, idx in enumerate(sorted_indices, start=1):
        feature  = feature_names[idx]
        score    = importances[idx]
        bar      = "█" * int(score * 200)   # Scale bar for readability
        print(f"  {rank:<6} {feature:<35} {score:>12.4f}  {bar}")

    # Highlight the top 3 most important features
    top3 = [feature_names[i] for i in sorted_indices[:3]]
    print(f"\n  Top 3 most important KPIs:")
    for i, feat in enumerate(top3, start=1):
        print(f"    {i}. {feat}")


# =============================================================================
# STEP 7 — Save the trained model
# =============================================================================

def save_model(model, path):
    """
    Save the trained model to disk as a .pkl (pickle) file.

    WHY SAVE THE MODEL?
    Training takes time and computation. Once trained, we save the model
    so the FastAPI backend can load it instantly and make predictions
    without retraining from scratch every time.

    joblib is the recommended way to save scikit-learn models.
    It is more efficient than Python's built-in pickle for objects
    that contain large numpy arrays (like a Random Forest with 100 trees).

    To load the model later:
        import joblib
        model = joblib.load("models/random_forest_model.pkl")
        prediction = model.predict(X_new)
    """
    print_section("STEP 7 — Saving the trained model")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ Model saved to: {path}")
    print(f"  File size: {size_kb:.1f} KB")
    print(f"\n  To load this model later:")
    print(f"    import joblib")
    print(f"    model = joblib.load('{path}')")
    print(f"    predictions = model.predict(X_new)")


# =============================================================================
# MAIN — Run all steps in order
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  Random Forest Training — O-RAN KPI Prediction xApp")
    print("  FYP — ML-based KPI Prediction xApp")
    print("=" * 65)

    # Step 1 — Load data
    df = load_data(LABELED_DATA_PATH)

    # Step 2 — Prepare X (features) and y (label)
    X, y = prepare_features_and_label(df)

    # Step 3 — Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3b — Naive baseline on the SAME test split: "assume state unchanged"
    # (must be evaluated on held-out data too, or it isn't a fair comparison)
    print_section("STEP 3b — Naive baseline on the held-out test set")
    # IMPORTANT: use POSITIONAL alignment (iloc), not X_test.index (loc).
    # With real multi-session data, 'timestamp' is NOT guaranteed unique —
    # multiple recording files can easily share timestamp values (e.g. if
    # each session's clock is relative to its own start). df.loc[X_test.index]
    # would then match EVERY row sharing each duplicate timestamp, silently
    # returning far more rows than X_test actually has. Since split_data()
    # uses shuffle=False, X_test is exactly df's last len(y_test) rows in
    # original order — iloc on that same positional boundary is always
    # correct regardless of whether the index has duplicates.
    current_test = df[CURRENT_LABEL_COL].iloc[len(y_train):]
    baseline_metrics = naive_baseline_metrics(current_test, y_test)
    transition_report(current_test, y_test, predicted_future=current_test)

    # Step 4 — Train the Random Forest
    model = train_model(X_train, y_train)

    # Step 5 — Evaluate on the test set
    metrics = evaluate_model(model, X_test, y_test)
    metrics["model"] = "Random Forest (5s forecast)"
    print_section("STEP 5b — Transition analysis for Random Forest")
    rf_test_pred = model.predict(X_test)
    transition_report(current_test, y_test, predicted_future=rf_test_pred)

    # Step 6 — Show which features mattered most
    print_feature_importances(model, list(X.columns))

    # Step 7 — Save the trained model
    save_model(model, MODEL_SAVE_PATH)

    # Step 8 — Record both entries in the shared comparison report
    upsert_comparison_report(COMPARISON_REPORT_PATH, baseline_metrics)
    upsert_comparison_report(COMPARISON_REPORT_PATH, metrics)

    # Final summary
    print("\n" + "=" * 65)
    print("  ✓ Random Forest (5s forecast) training complete!")
    print(f"  F1 Score : {metrics['f1_score']}  (baseline: {baseline_metrics['f1_score']})")
    print(f"  Accuracy : {metrics['accuracy']}  (baseline: {baseline_metrics['accuracy']})")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")
    print("  Next step: Phase 2b — Train XGBoost classifier.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
