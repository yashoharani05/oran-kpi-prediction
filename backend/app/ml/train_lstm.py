# =============================================================================
# train_lstm.py
#
# PURPOSE:
#   Train an LSTM neural network to FORECAST network degradation ~5 seconds
#   ahead (see docs/FORECASTING_METHODOLOGY_UPDATE.md), using the last 20 KPI
#   readings as context (sliding window over time).
#
# HOW TO RUN (from the backend/ folder with venv active):
#   python app/ml/train_lstm.py
#
# INPUT:  backend/data/processed/labeled_dataset.csv
# OUTPUT: backend/models/lstm_forecast_5s.keras
#         backend/models/lstm_scaler_forecast_5s.pkl
#         backend/models/comparison_report.json  (updated with all entries)
#
# WHY THIS VERSION IS DIFFERENT FROM THE ORIGINAL:
#   The original script tried to build ALL sliding-window sequences at once,
#   storing them in a numpy array. With 19 million rows, that array would
#   need ~51 GB of RAM — which no normal PC has.
#
#   This version uses a Keras Sequence generator. Instead of building all
#   windows in advance, it creates small batches of windows ON DEMAND
#   as each training step needs them. Memory usage stays near-zero regardless
#   of how many rows are in the dataset.
#
# WHAT IS AN LSTM?
#   LSTM (Long Short-Term Memory) is a neural network designed for sequences.
#   Instead of looking at one row at a time (like RF and XGBoost), LSTM
#   looks at a window of the last N rows and learns from trends over time.
#
# ARCHITECTURE: LSTM(32) -> Dropout(0.2) -> Dense(16) -> Dense(1, sigmoid)
# =============================================================================

import os
import json
import joblib
import math
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TensorFlow startup messages

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from forecast_config import (
    LSTM_WINDOW_SIZE, FORECAST_HORIZON_ROWS, FORECAST_HORIZON_SECONDS,
    SAMPLE_INTERVAL_SECONDS, CURRENT_LABEL_COL, FUTURE_LABEL_COL,
    SESSION_ID_COL,
)
from forecast_utils import (
    naive_baseline_metrics, transition_report, upsert_comparison_report,
    session_row_ranges, downcast_numeric_dtypes,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

LABELED_DATA_PATH      = os.path.join("data", "processed", "labeled_dataset.csv")
# Renamed from lstm_model.keras / lstm_scaler.pkl — this model now targets
# degradation_risk_future (section 28 of the correction brief).
LSTM_MODEL_SAVE_PATH   = os.path.join("models", "lstm_forecast_5s.keras")
SCALER_SAVE_PATH       = os.path.join("models", "lstm_scaler_forecast_5s.pkl")
COMPARISON_REPORT_PATH = os.path.join("models", "comparison_report.json")

# WINDOW_SIZE (how many past rows the LSTM looks at) is intentionally kept as
# its OWN constant, separate from FORECAST_HORIZON_ROWS (how far ahead it
# predicts) — see forecast_config.py's comment on LSTM_WINDOW_SIZE. They both
# happen to equal 20 today; that is a coincidence, not a dependency.
WINDOW_SIZE  = LSTM_WINDOW_SIZE
EPOCHS       = 30     # maximum training rounds (early stopping usually stops earlier)
BATCH_SIZE   = 512    # sequences per gradient update — larger = faster with big datasets
TEST_SIZE    = 0.2    # must match RF and XGBoost for fair comparison

# The 18 feature columns — must match what clean_dataset.py produces
FEATURE_COLS = [
    "dl_mcs", "dl_n_samples", "dl_buffer_bytes", "tx_brate_downlink_mbps",
    "tx_pkts_downlink", "dl_cqi", "ul_mcs", "ul_n_samples", "ul_buffer_bytes",
    "rx_brate_uplink_mbps", "rx_pkts_uplink", "rx_errors_uplink_pct",
    "ul_sinr", "phr", "sum_requested_prbs", "sum_granted_prbs",
    "ul_turbo_iters", "prb_grant_ratio",
]


def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
# STEP 1 — Load the labeled dataset
# =============================================================================

def load_data(path):
    print_section("STEP 1 — Loading labeled dataset")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ERROR: {path} not found.\n"
            "Run label_dataset.py first."
        )

    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)

    # Re-downcast — pandas.read_csv re-infers float64/int64 from the CSV
    # text regardless of what dtypes were used when it was written. This
    # matters even more here: scale_features() below builds a full float32
    # numpy copy of the feature matrix, so starting from float32 instead of
    # float64 halves that allocation too.
    df = downcast_numeric_dtypes(df)

    # Check all expected feature columns are present
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"ERROR: These feature columns are missing from the dataset:\n"
            f"  {missing_cols}\n"
            f"Re-run clean_dataset.py and label_dataset.py to regenerate."
        )
    if FUTURE_LABEL_COL not in df.columns:
        raise ValueError(
            f"ERROR: '{FUTURE_LABEL_COL}' column is missing from the dataset.\n"
            f"Re-run label_dataset.py — it must be the version that creates the "
            f"future forecasting target (see docs/FORECASTING_METHODOLOGY_UPDATE.md)."
        )

    print(f"  Rows:          {df.shape[0]:,}")
    print(f"  Features used: {len(FEATURE_COLS)}")
    print(f"  Target column: '{FUTURE_LABEL_COL}' "
          f"(degradation state ~{FORECAST_HORIZON_SECONDS}s ahead)")
    print(f"  Normal   (0):  {(df[FUTURE_LABEL_COL]==0).sum():,}")
    print(f"  Degraded (1):  {(df[FUTURE_LABEL_COL]==1).sum():,}")
    return df


# =============================================================================
# STEP 2 — Scale features to [0, 1]
# =============================================================================

def scale_features(df, split_idx):
    """
    Apply MinMaxScaler so all 18 features are in the range [0, 1].

    WHY SCALE FOR LSTM BUT NOT FOR RF/XGBOOST?
    Decision trees split on thresholds and are not affected by the scale
    of values. Neural networks use gradient descent — if one feature is
    0–5000 and another is 0–0.3, the large-range feature dominates the
    gradients and training becomes unstable. Scaling to [0,1] fixes this.

    WHY FIT ON TRAINING DATA ONLY?
    We fit the scaler on rows 0 to split_idx (the training set), then
    transform ALL rows with it. If we fitted on all rows, information
    from the test set (future data) would leak into the scaling — making
    our evaluation results falsely optimistic.
    """
    print_section("STEP 2 — Scaling features to [0, 1]")

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    # y_all is the FUTURE target (degradation_risk_future) — what the LSTM
    # must learn to predict, ~FORECAST_HORIZON_SECONDS ahead of each row.
    y_all = df[FUTURE_LABEL_COL].values.astype(np.int8)
    # current_all (label_now) is kept only for the naive baseline / transition
    # analysis later — it is never used as a model input.
    current_all = df[CURRENT_LABEL_COL].values.astype(np.int8)

    # Replace any infinity values before scaling (can occur with prb_grant_ratio)
    X_all = np.where(np.isinf(X_all), np.nan, X_all)
    col_medians = np.nanmedian(X_all[:split_idx], axis=0)
    for col_idx in range(X_all.shape[1]):
        nan_mask = np.isnan(X_all[:, col_idx])
        if nan_mask.any():
            X_all[nan_mask, col_idx] = col_medians[col_idx]

    scaler = MinMaxScaler()
    scaler.fit(X_all[:split_idx])       # fit on training portion only
    X_scaled = scaler.transform(X_all)  # apply to entire dataset

    print(f"  Scaler fitted on rows 0–{split_idx-1:,} (training set only)")
    print(f"  Prevents test-set information from leaking into the scaler")
    print(f"  Data type: float32 (saves memory vs float64)")

    return X_scaled, y_all, scaler


# =============================================================================
# STEP 3 — Memory-efficient sliding window generator
# =============================================================================

class WindowGenerator(Sequence):
    """
    A Keras-compatible data generator that creates sliding-window sequences
    on demand, one batch at a time, from an explicit list of valid window
    START positions (see compute_valid_window_starts()).

    METHODOLOGY NOTE (forecasting correction — see docs/FORECASTING_METHODOLOGY_UPDATE.md):
    Each sequence covers rows [start : start+window) — i.e. rows
    (t-window+1) ... t, where t = start+window-1 is the "prediction origin"
    (the last row the model actually sees).

    The target for that sequence is y[t], i.e. y[start+window-1] — NOT
    y[start+window] as the original same-instant version used. Because `y`
    here is already 'degradation_risk_future' (computed in label_dataset.py
    as label_now shifted FORECAST_HORIZON_ROWS into the future, per session),
    y[t] already means "degradation state at t + FORECAST_HORIZON_ROWS".
    Using y[start+window-1] (this version) means:
        window ends at t  ->  target = state at t + horizon  (correct, 5s ahead)
    Using y[start+window] (the old version) would have meant:
        window ends at t  ->  target = state at (t+1) + horizon (accidentally
        one extra row — and if window_size == horizon rows, close to double
        the intended horizon). This is exactly the mistake section 16 of the
        correction brief warns about, so the alignment is spelled out here
        and re-verified by print_window_alignment_example() below.

    WHY AN EXPLICIT valid_starts LIST INSTEAD OF A SIMPLE RANGE?
    With more than one recording session, a plain `range(len(X)-window)`
    would happily build a window that starts in session A and ends in
    session B. `valid_starts` is pre-computed by compute_valid_window_starts()
    to only include windows that lie entirely inside one session's
    contiguous row block (see forecast_utils.session_row_ranges).

    Args:
        X            — scaled feature array, shape (n_rows, n_features)
        y            — FUTURE-label array, shape (n_rows,)
        window       — how many past rows form one sequence (WINDOW_SIZE)
        batch_size   — how many sequences per batch
        valid_starts — sorted list/array of valid window start positions
    """

    def __init__(self, X, y, window, batch_size, valid_starts):
        self.X            = X
        self.y            = y
        self.window       = window
        self.batch_size   = batch_size
        self.valid_starts = np.asarray(valid_starts, dtype=np.int64)
        self.n_sequences  = len(self.valid_starts)

    def __len__(self):
        """Number of batches per epoch."""
        return math.ceil(self.n_sequences / self.batch_size) if self.n_sequences else 0

    def __getitem__(self, batch_idx):
        """Build and return one batch of (X_batch, y_batch)."""
        batch_starts = self.valid_starts[
            batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size
        ]

        X_batch = np.stack([self.X[s: s + self.window] for s in batch_starts])
        # Target = y at the LAST row of the window (the prediction origin t),
        # which already holds the future-shifted label. See class docstring.
        y_batch = self.y[batch_starts + self.window - 1]

        return X_batch, y_batch


def compute_valid_window_starts(session_ranges, window_size, target_lo, target_hi):
    """
    Return every window start position `s` such that:
      1. the window [s, s+window_size) lies entirely inside ONE session's
         contiguous row block (never straddles two sessions), AND
      2. the prediction origin t = s+window_size-1 falls inside
         [target_lo, target_hi) — i.e. the row being predicted FOR belongs
         to this split (train or test), even if a little context immediately
         before the split boundary is borrowed from the previous rows for
         the window itself.

    session_ranges: list of (start, end) row-POSITION tuples, e.g. from
                    forecast_utils.session_row_ranges().

    IMPLEMENTATION NOTE (scale fix): builds each session's valid-start range
    as a numpy array (np.arange) rather than a Python list built from
    range()/list.extend(). For a small dataset this makes no visible
    difference; for tens of millions of rows a Python list of that many
    individual int objects is meaningfully heavier (CPython list-of-ints
    overhead, ~28 bytes/element) than one contiguous numpy int64 array
    (8 bytes/element) — a >3x reduction for this specific structure.
    """
    chunks = []
    for s_block, e_block in session_ranges:
        s_min = max(s_block, target_lo - window_size + 1, 0)
        s_max = min(e_block - window_size, target_hi - window_size)
        if s_max >= s_min:
            chunks.append(np.arange(s_min, s_max + 1, dtype=np.int64))
    if not chunks:
        return np.array([], dtype=np.int64)
    return np.concatenate(chunks)


def print_window_alignment_example(window_size, horizon_rows, sample_interval_s):
    """
    Print a concrete worked example of the window/target alignment, as
    requested by section 16 of the correction brief ("add a small sanity
    check example in comments/tests").
    """
    origin = 100 + window_size - 1  # arbitrary example start=100
    print(f"\n  Window/target alignment example:")
    print(f"    row interval     = {sample_interval_s*1000:.0f} ms")
    print(f"    window size      = {window_size} rows")
    print(f"    forecast horizon = {horizon_rows} rows ({horizon_rows*sample_interval_s:.1f}s)")
    print(f"    input rows       = {origin - window_size + 1}-{origin}")
    print(f"    prediction origin (t) = row {origin}")
    print(f"    target = degradation_risk_future[t] "
          f"= degradation_risk[t + {horizon_rows}] = state at row {origin + horizon_rows}")
    print(f"    time ahead from row {origin} = {horizon_rows} x {sample_interval_s*1000:.0f}ms "
          f"= {horizon_rows*sample_interval_s:.1f}s")


def prepare_generators(X_scaled, y_all, df, split_idx, window_size):
    """
    Create train and validation generators using session-boundary-aware
    window starts (see compute_valid_window_starts / WindowGenerator above).
    """
    print_section(f"STEP 3 — Setting up sliding window generators (size={window_size})")

    print_window_alignment_example(window_size, FORECAST_HORIZON_ROWS, SAMPLE_INTERVAL_SECONDS)

    session_ranges = session_row_ranges(df, SESSION_ID_COL)
    n_rows = len(X_scaled)

    train_starts = compute_valid_window_starts(session_ranges, window_size, 0, split_idx)
    test_starts  = compute_valid_window_starts(session_ranges, window_size, split_idx, n_rows)

    train_gen = WindowGenerator(X_scaled, y_all, window_size, BATCH_SIZE, train_starts)
    test_gen  = WindowGenerator(X_scaled, y_all, window_size, BATCH_SIZE, test_starts)

    # Compute class distribution for class_weight (over TRAIN targets only)
    train_targets = y_all[np.asarray(train_starts, dtype=np.int64) + window_size - 1] if len(train_starts) > 0 else np.array([])
    n_normal   = int((train_targets == 0).sum())
    n_degraded = int((train_targets == 1).sum())

    print(f"\n  Sessions found:        {len(session_ranges)}")
    print(f"  Training sequences:    {len(train_starts):,}")
    print(f"  Test sequences:        {len(test_starts):,}")
    print(f"  Batches per epoch:     {len(train_gen):,} (batch size = {BATCH_SIZE})")
    print(f"  Memory per batch:      {BATCH_SIZE * window_size * len(FEATURE_COLS) * 4 / 1024:.0f} KB")
    print(f"  Train Normal:          {n_normal:,}   Degraded: {n_degraded:,}")

    class_weight = {0: 1.0, 1: (n_normal / max(n_degraded, 1))}
    print(f"  Class weights:         Normal=1.0  Degraded={class_weight[1]:.2f}")

    return train_gen, test_gen, class_weight, len(test_starts)


# =============================================================================
# STEP 4 — Build the LSTM architecture
# =============================================================================

def build_model(window_size, n_features):
    """
    Define the LSTM network.

    LSTM(32):           32 memory cells process the 20-step sequence.
                        Each cell maintains a hidden state across time steps,
                        deciding what to remember and what to forget.
    Dropout(0.2):       During training, randomly zeros 20% of LSTM outputs.
                        This prevents the network from memorising the training
                        data (overfitting). Dropout is automatically disabled
                        when model.predict() is called.
    Dense(16, relu):    A small fully-connected layer to combine LSTM outputs.
                        relu = max(0, x) — introduces non-linearity.
    Dense(1, sigmoid):  Outputs a probability between 0 and 1.
                        sigmoid(x) = 1 / (1 + e^-x)
                        >= 0.5 → Degraded, < 0.5 → Normal.
    """
    print_section("STEP 4 — Building LSTM model")

    model = Sequential([
        LSTM(32, input_shape=(window_size, n_features)),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=lambda x: print(f"  {x}"))
    return model


# =============================================================================
# STEP 5 — Train using the generator
# =============================================================================

def train_model(model, train_gen, test_gen, class_weight):
    """
    Train the LSTM using the generator-based approach.

    The generator feeds batches of 512 windows at a time.
    Keras handles calling train_gen.__getitem__() automatically.

    EarlyStopping: if validation loss does not improve for 5 consecutive
    epochs, stop training and revert to the best weights seen so far.
    This prevents wasting time on epochs that are not helping.
    """
    print_section("STEP 5 — Training (generator-based, memory-efficient)")

    print(f"  Max epochs:     {EPOCHS}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Early stopping: patience = 5 epochs")
    print(f"  Training now (this may take several minutes on large datasets)...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,   # show when early stopping triggers
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,   # show one line per epoch so you can see progress
    )

    epochs_run = len(history.history["loss"])
    best_val   = min(history.history["val_loss"])
    print(f"\n  Finished: {epochs_run} epochs  |  best val_loss = {best_val:.4f}")
    return model


# =============================================================================
# STEP 6 — Evaluate on the test set
# =============================================================================

def evaluate_model(model, test_gen, n_test_seqs):
    """
    Run predictions on the entire test set and compute metrics.

    We use the generator to keep memory usage low during evaluation too.
    model.predict(generator) processes one batch at a time.
    """
    print_section("STEP 6 — Evaluating LSTM on test sequences")

    print(f"  Running predictions on {n_test_seqs:,} test sequences...")
    y_prob = model.predict(test_gen, verbose=0).flatten()
    y_prob = y_prob[:n_test_seqs]

    # Reconstruct the true (future) labels for the test sequences directly
    # from the generator's own valid_starts + window alignment — this stays
    # correct regardless of session boundaries or dataset size, unlike
    # slicing the array by a fixed offset.
    target_idx = test_gen.valid_starts[:n_test_seqs] + test_gen.window - 1
    y_test     = test_gen.y[target_idx]

    # Apply 0.5 threshold: probability >= 0.5 → Degraded (1)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    print(f"\n  +-------------------------------------+")
    print(f"  |  LSTM (window={WINDOW_SIZE}, forecast={FORECAST_HORIZON_SECONDS}s) |")
    print(f"  +-------------------------------------+")
    print(f"  |  Accuracy   : {accuracy:>8.4f}  ({accuracy*100:.2f}%)  |")
    print(f"  |  Precision  : {precision:>8.4f}  ({precision*100:.2f}%)  |")
    print(f"  |  Recall     : {recall:>8.4f}  ({recall*100:.2f}%)  |")
    print(f"  |  F1 Score   : {f1:>8.4f}  ({f1*100:.2f}%)  |")
    print(f"  +-------------------------------------+")
    print(f"\n  Confusion matrix:")
    print(f"                   Pred Normal  Pred Degraded")
    print(f"  Actual Normal  (0)    {tn:>9,}      {fp:>9,}")
    print(f"  Actual Degraded(1)    {fn:>9,}      {tp:>9,}")
    print(f"\n  Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Degraded"], zero_division=0))

    return {
        "model":       "LSTM (5s forecast)",
        "accuracy":    round(float(accuracy),  4),
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall),    4),
        "f1_score":    round(float(f1),        4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "window_size": WINDOW_SIZE,
    }, y_test, y_pred


# =============================================================================
# STEP 7 — Three-way comparison with RF and XGBoost
# =============================================================================

def three_way_comparison(lstm_metrics):
    print_section("STEP 7 — Comparison: Naive Baseline vs RF vs XGBoost vs LSTM")

    if not os.path.exists(COMPARISON_REPORT_PATH):
        print("  Warning: comparison_report.json not found.")
        print("  Run train_random_forest.py and train_xgboost.py first.")
        return [lstm_metrics]

    with open(COMPARISON_REPORT_PATH) as f:
        existing = json.load(f)["models"]

    # Only keep non-LSTM entries — remove any old LSTM entry before adding ours
    existing = [m for m in existing if "LSTM" not in m.get("model", "")]
    all_m    = existing + [lstm_metrics]

    metric_keys   = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall",  "F1 Score"]

    print(f"\n  {'Metric':<12}", end="")
    for m in all_m:
        print(f"  {m['model'][:24]:>24}", end="")
    print("  Winner")
    print("  " + "-" * 90)

    for key, label in zip(metric_keys, metric_labels):
        vals = [m[key] for m in all_m]
        best = max(vals)
        print(f"  {label:<12}", end="")
        for m in all_m:
            marker = "*" if m[key] == best else " "
            print(f"  {m[key]*100:>22.2f}%{marker}", end="")
        winner = all_m[vals.index(best)]["model"]
        print(f"  {winner}")

    print(f"\n  Missed degradation events (False Negatives — lower is better):")
    for m in all_m:
        print(f"    {m['model']:<28}  FN = {m['fn']:,}  TP = {m['tp']:,}")

    return all_m


# =============================================================================
# STEP 8 — Save model, scaler, and comparison report
# =============================================================================

def save_outputs(model, scaler, all_metrics):
    print_section("STEP 8 — Saving model, scaler, and comparison report")

    os.makedirs("models", exist_ok=True)

    model.save(LSTM_MODEL_SAVE_PATH)
    print(f"  LSTM model saved:    {LSTM_MODEL_SAVE_PATH}")

    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"  Scaler saved:        {SCALER_SAVE_PATH}")

    with open(COMPARISON_REPORT_PATH, "w") as f:
        json.dump({"models": all_metrics}, f, indent=2)
    print(f"  Comparison report:   {COMPARISON_REPORT_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  LSTM Training — O-RAN KPI Prediction xApp")
    print(f"  Forecasting {FORECAST_HORIZON_SECONDS}s ahead — memory-efficient generator-based training")
    print("=" * 65)

    # Step 1 — Load the labeled dataset (must contain degradation_risk_future)
    df = load_data(LABELED_DATA_PATH)

    # Split index for 80/20 — must match RF and XGBoost
    split_idx = int(len(df) * (1 - TEST_SIZE))

    # Step 2 — Scale all features to [0, 1]
    X_scaled, y_all, scaler = scale_features(df, split_idx)

    # Step 3 — Create session-boundary-aware, memory-efficient generators
    train_gen, test_gen, class_weight, n_test_seqs = prepare_generators(
        X_scaled, y_all, df, split_idx, WINDOW_SIZE
    )

    # Step 4 — Build the LSTM network
    model = build_model(WINDOW_SIZE, len(FEATURE_COLS))

    # Step 5 — Train using the generator
    model = train_model(model, train_gen, test_gen, class_weight)

    # Step 6 — Evaluate on the test set
    lstm_metrics, y_test, y_pred = evaluate_model(model, test_gen, n_test_seqs)

    # Step 6b — Naive baseline + transition analysis on the SAME test sequences
    # (aligned to the same prediction-origin rows the LSTM was evaluated on)
    print_section("STEP 6b — Naive baseline on the LSTM's test sequences")
    target_idx  = test_gen.valid_starts[:n_test_seqs] + test_gen.window - 1
    current_test = pd.Series(df[CURRENT_LABEL_COL].to_numpy()[target_idx])
    future_test  = pd.Series(y_test)
    baseline_metrics = naive_baseline_metrics(current_test, future_test)
    transition_report(current_test, future_test, predicted_future=y_pred)

    # Step 7 — Print comparison with Naive Baseline, RF and XGBoost
    all_metrics = three_way_comparison(lstm_metrics)

    # Step 8 — Save everything (also (re-)upserts the baseline entry so this
    # script is runnable standalone without erasing RF/XGB's copy of it)
    upsert_comparison_report(COMPARISON_REPORT_PATH, baseline_metrics)
    save_outputs(model, scaler, all_metrics)

    print("\n" + "=" * 65)
    print("  LSTM training complete!")
    print(f"  F1 Score : {lstm_metrics['f1_score']}  (baseline: {baseline_metrics['f1_score']})")
    print(f"  Model    : {LSTM_MODEL_SAVE_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
