# =============================================================================
# train_lstm.py
#
# PURPOSE:
#   Train an LSTM neural network to predict network degradation using the
#   last 20 KPI readings as context (sliding window over time).
#
# HOW TO RUN (from the backend/ folder with venv active):
#   python app/ml/train_lstm.py
#
# INPUT:  backend/data/processed/labeled_dataset.csv
# OUTPUT: backend/models/lstm_model.keras
#         backend/models/lstm_scaler.pkl
#         backend/models/comparison_report.json  (updated with all 3 models)
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

# =============================================================================
# CONFIGURATION
# =============================================================================

LABELED_DATA_PATH      = os.path.join("data", "processed", "labeled_dataset.csv")
LSTM_MODEL_SAVE_PATH   = os.path.join("models", "lstm_model.keras")
SCALER_SAVE_PATH       = os.path.join("models", "lstm_scaler.pkl")
COMPARISON_REPORT_PATH = os.path.join("models", "comparison_report.json")

WINDOW_SIZE  = 20     # how many past readings the LSTM looks at per prediction
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

    # Check all expected feature columns are present
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"ERROR: These feature columns are missing from the dataset:\n"
            f"  {missing_cols}\n"
            f"Re-run clean_dataset.py and label_dataset.py to regenerate."
        )

    print(f"  Rows:          {df.shape[0]:,}")
    print(f"  Features used: {len(FEATURE_COLS)}")
    print(f"  Normal   (0):  {(df['degradation_risk']==0).sum():,}")
    print(f"  Degraded (1):  {(df['degradation_risk']==1).sum():,}")
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
    y_all = df["degradation_risk"].values.astype(np.int8)

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
    on demand, one batch at a time.

    WHY A GENERATOR INSTEAD OF np.array?
    Building all windows at once requires:
      19 million rows × 20 window × 18 features × 4 bytes = ~27 GB

    A generator creates only BATCH_SIZE windows at a time:
      512 windows × 20 × 18 × 4 bytes = ~750 KB per batch

    This is the standard way to train neural networks on large datasets.
    Keras calls __getitem__(batch_index) automatically during training.

    Args:
        X          — scaled feature array, shape (n_rows, n_features)
        y          — label array, shape (n_rows,)
        window     — how many past rows form one sequence (WINDOW_SIZE)
        batch_size — how many sequences per batch
        n_classes  — dict with class counts for class_weight calculation
    """

    def __init__(self, X, y, window, batch_size):
        self.X          = X
        self.y          = y
        self.window     = window
        self.batch_size = batch_size
        # Number of valid sequences: each sequence ends at row i+window
        # so the first valid end is at row 'window' (index window-1+1)
        self.n_sequences = len(X) - window

    def __len__(self):
        """Number of batches per epoch."""
        return math.ceil(self.n_sequences / self.batch_size)

    def __getitem__(self, batch_idx):
        """
        Build and return one batch of (X_batch, y_batch).

        batch_idx is the batch number (0, 1, 2, ...).
        We figure out which rows it corresponds to and slice them.
        """
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, self.n_sequences)

        # Build the batch sequences
        # Each sequence: rows [i : i+window], label: y[i+window]
        X_batch = np.stack(
            [self.X[i : i + self.window] for i in range(start, end)]
        )
        y_batch = self.y[start + self.window : end + self.window]

        return X_batch, y_batch


def prepare_generators(X_scaled, y_all, split_idx, window_size):
    """
    Create train and validation generators.

    The split point in sequence space: the last training sequence ends at
    row split_idx-1, so the training generator covers rows 0 to split_idx.
    The test generator covers rows split_idx to the end.
    """
    print_section(f"STEP 3 — Setting up sliding window generators (size={window_size})")

    # Training generator
    X_train = X_scaled[:split_idx]
    y_train = y_all[:split_idx]
    train_gen = WindowGenerator(X_train, y_train, window_size, BATCH_SIZE)

    # Test generator (used for validation during training and final evaluation)
    X_test = X_scaled[split_idx - window_size:]   # include overlap for first window
    y_test = y_all[split_idx - window_size:]
    test_gen = WindowGenerator(X_test, y_test, window_size, BATCH_SIZE)

    # Compute class distribution for class_weight
    n_normal   = int((y_train == 0).sum())
    n_degraded = int((y_train == 1).sum())

    n_train_seqs = len(X_train) - window_size
    n_test_seqs  = len(X_test)  - window_size

    print(f"  Training sequences:   {n_train_seqs:,}")
    print(f"  Test sequences:       {n_test_seqs:,}")
    print(f"  Batches per epoch:    {len(train_gen):,} (batch size = {BATCH_SIZE})")
    print(f"  Memory per batch:     {BATCH_SIZE * window_size * len(FEATURE_COLS) * 4 / 1024:.0f} KB")
    print(f"  Train Normal:         {n_normal:,}   Degraded: {n_degraded:,}")

    class_weight = {0: 1.0, 1: n_normal / max(n_degraded, 1)}
    print(f"  Class weights:        Normal=1.0  Degraded={class_weight[1]:.2f}")

    return train_gen, test_gen, class_weight, n_test_seqs


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

    # Trim to the exact number of test sequences
    # (the last batch may be padded slightly by the generator)
    y_prob = y_prob[:n_test_seqs]

    # Reconstruct the true labels for the test sequences
    # Test sequences start from index WINDOW_SIZE in the test slice
    X_test_arr = test_gen.X
    y_test_arr = test_gen.y
    y_test = y_test_arr[WINDOW_SIZE : WINDOW_SIZE + n_test_seqs]

    # Apply 0.5 threshold: probability >= 0.5 → Degraded (1)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"\n  +-------------------------------------+")
    print(f"  |  LSTM (window = {WINDOW_SIZE})               |")
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
    print(classification_report(y_test, y_pred, target_names=["Normal", "Degraded"]))

    return {
        "model":       "LSTM",
        "accuracy":    round(float(accuracy),  4),
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall),    4),
        "f1_score":    round(float(f1),        4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "window_size": WINDOW_SIZE,
    }


# =============================================================================
# STEP 7 — Three-way comparison with RF and XGBoost
# =============================================================================

def three_way_comparison(lstm_metrics):
    print_section("STEP 7 — Three-way comparison: RF vs XGBoost vs LSTM")

    if not os.path.exists(COMPARISON_REPORT_PATH):
        print("  Warning: comparison_report.json not found.")
        print("  Run train_random_forest.py and train_xgboost.py first.")
        return [lstm_metrics]

    with open(COMPARISON_REPORT_PATH) as f:
        existing = json.load(f)["models"]

    # Only keep RF and XGBoost entries — remove any old LSTM entry
    existing = [m for m in existing if "LSTM" not in m.get("model", "")]
    all_m    = existing + [lstm_metrics]

    metric_keys   = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall",  "F1 Score"]

    print(f"\n  {'Metric':<12}", end="")
    for m in all_m:
        print(f"  {m['model'][:18]:>18}", end="")
    print("  Winner")
    print("  " + "-" * 70)

    for key, label in zip(metric_keys, metric_labels):
        vals = [m[key] for m in all_m]
        best = max(vals)
        print(f"  {label:<12}", end="")
        for m in all_m:
            marker = "*" if m[key] == best else " "
            print(f"  {m[key]*100:>16.2f}%{marker}", end="")
        winner = all_m[vals.index(best)]["model"]
        print(f"  {winner}")

    print(f"\n  Missed degradation events (False Negatives — lower is better):")
    for m in all_m:
        print(f"    {m['model']:<22}  FN = {m['fn']:,}  TP = {m['tp']:,}")

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
    print("  Memory-efficient generator-based training")
    print("=" * 65)

    # Step 1 — Load the labeled dataset
    df = load_data(LABELED_DATA_PATH)

    # Split index for 80/20 — must match RF and XGBoost
    split_idx = int(len(df) * (1 - TEST_SIZE))

    # Step 2 — Scale all features to [0, 1]
    X_scaled, y_all, scaler = scale_features(df, split_idx)

    # Step 3 — Create memory-efficient generators (no giant array in RAM)
    train_gen, test_gen, class_weight, n_test_seqs = prepare_generators(
        X_scaled, y_all, split_idx, WINDOW_SIZE
    )

    # Step 4 — Build the LSTM network
    model = build_model(WINDOW_SIZE, len(FEATURE_COLS))

    # Step 5 — Train using the generator
    model = train_model(model, train_gen, test_gen, class_weight)

    # Step 6 — Evaluate on the test set
    lstm_metrics = evaluate_model(model, test_gen, n_test_seqs)

    # Step 7 — Print three-way comparison with RF and XGBoost
    all_metrics = three_way_comparison(lstm_metrics)

    # Step 8 — Save everything
    save_outputs(model, scaler, all_metrics)

    print("\n" + "=" * 65)
    print("  LSTM training complete!")
    print(f"  F1 Score : {lstm_metrics['f1_score']}")
    print(f"  Model    : {LSTM_MODEL_SAVE_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
