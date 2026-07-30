# ML Module — Data Preparation & Model Training

This folder contains all scripts for preparing data and training ML models.
Run them **in order** — each script depends on the output of the previous one.

```
app/ml/
├── clean_dataset.py        ← Step 1: Load ALL CSVs, merge, clean
├── label_dataset.py        ← Step 2: Feature engineering + labelling
├── train_random_forest.py  ← Step 3: Train Random Forest classifier
├── train_xgboost.py        ← Step 4: Train XGBoost classifier
├── train_lstm.py           ← Step 5: Train LSTM (time-series)
└── README.md               ← This file
```

**Run order (from the `backend/` folder with venv active):**
```
python app\ml\clean_dataset.py
python app\ml\label_dataset.py
python app\ml\train_random_forest.py
python app\ml\train_xgboost.py
python app\ml\train_lstm.py
```

---

## Data Flow

```
backend/data/raw/
  *.csv  (one or more files)
        │
        ▼
  clean_dataset.py
        │  validates, merges, deduplicates, cleans
        ├── combined_dataset.csv   ← raw merge checkpoint
        └── cleaned_dataset.csv   ← fully cleaned, ready for labelling
        │
        ▼
  label_dataset.py
        │  scores 7 KPI conditions, assigns degradation_risk label
        └── labeled_dataset.csv
        │
        ▼
  train_*.py (three separate scripts)
        └── models/*.pkl / *.keras
```

---

## Step 1 — clean_dataset.py

**Input:**  `data/raw/*.csv` — **all** CSV files in that folder  
**Output:**
- `data/processed/combined_dataset.csv` — raw merged data (checkpoint)
- `data/processed/cleaned_dataset.csv` — cleaned and ready for labelling

### What it does (15 steps)

| Step | Action |
|---|---|
| 1 | Scan `data/raw/` for all `.csv` files |
| 2 | Validate each file (skip empty, corrupted, or column-mismatched files) |
| 3 | Merge all valid files into one DataFrame |
| 4 | Save the raw combined data as a checkpoint |
| 5 | Drop completely empty columns (Unnamed separator columns) |
| 6 | Drop the internal `_source_file` tracking column |
| 7 | Drop zero-variance columns (columns that never change across ALL files) |
| 8 | Remove duplicate rows (can appear when sessions overlap) |
| 9 | Rename columns to clean Python snake_case names |
| 10 | Convert timestamp from Unix milliseconds → datetime |
| 11 | Fill missing values with column medians |
| 12 | Run validity checks (negative values, CQI range, MCS range, error %) |
| 13 | Engineer `prb_grant_ratio` feature |
| 14 | Sort by timestamp, set as row index |
| 15 | Print preprocessing summary, save `cleaned_dataset.csv` |

### Multi-file validation rules

Each file in `data/raw/` is checked before being included:

| Rule | What happens if it fails |
|---|---|
| File is not empty (0 bytes) | Skipped with warning |
| File can be parsed as valid CSV | Skipped with warning |
| File has at least one data row | Skipped with warning |
| File columns match the first valid file | Skipped with warning |

Only files that pass all four checks are merged.

### Preprocessing summary printed at the end

```
┌─────────────────────────────────────────────────────┐
│              PREPROCESSING SUMMARY                  │
├───────────────────────────────┬─────────────────────┤
│  Files found in data/raw/     │  3                   │
│  Files successfully loaded    │  3                   │
│  Files skipped (with reason)  │  0                   │
├───────────────────────────────┼─────────────────────┤
│  Total rows before merge      │  6270                │
│  Rows in combined_dataset     │  6270                │
│  Duplicate rows removed       │  12                  │
│  Rows in cleaned_dataset      │  6258                │
├───────────────────────────────┼─────────────────────┤
│  Columns in cleaned dataset   │  18                  │
│  Missing values remaining     │  0                   │
└───────────────────────────────┴─────────────────────┘
```

### Note on zero-variance columns with multiple files

With a single CSV, many columns like `slice_id` and `num_ues` were constant
and therefore dropped. With multiple sessions from different experiments,
those same columns might now vary (e.g. different slice configurations per session).

The script correctly handles this: it only drops a column if it has **one unique
value across the entire merged dataset**, not just within a single file.

---

## Step 2 — label_dataset.py

**Input:**  `data/processed/cleaned_dataset.csv`  
**Output:** `data/processed/labeled_dataset.csv`

Creates the `degradation_risk` binary label (0 = Normal, 1 = Degraded)
using a quantile-based scoring system that adapts to the merged dataset's
own distribution — no hardcoded thresholds.

### Scoring conditions (7 total)

A row is labelled **Degraded** if it scores ≥ 2 bad conditions:

| # | Feature | Condition | Quantile |
|---|---|---|---|
| 1 | `rx_errors_uplink_pct` | Above Q90 | Worst 10% error rate |
| 2 | `dl_cqi` | Below Q25 | Worst 25% channel quality |
| 3 | `tx_brate_downlink_mbps` | Below Q10 | Worst 10% throughput |
| 4 | `prb_grant_ratio` | Below Q25 | Worst 25% bandwidth satisfaction |
| 5 | `dl_mcs` | Below Q25 | Worst 25% modulation quality |
| 6 | `ul_sinr` | Below Q25 (active only) | Worst 25% when transmitting |
| 7 | `ul_turbo_iters` | Above Q75 (active only) | Hardest 25% to decode |

Because thresholds are computed from the **merged dataset**, adding more files
will automatically recalibrate what counts as "bad" — the labelling logic
remains data-driven regardless of how many files you add.

---

## Steps 3–5 — Training scripts

**Input (all three):** `data/processed/labeled_dataset.csv`  
**No changes needed** — the training scripts read the same file as before.

| Script | Output model | F1 Score (1-file baseline) |
|---|---|---|
| `train_random_forest.py` | `models/random_forest_model.pkl` | 97.37% |
| `train_xgboost.py` | `models/xgboost_model.pkl` | 99.14% |
| `train_lstm.py` | `models/lstm_model.keras` + `lstm_scaler.pkl` | 31.65% |

Results will change (likely improve) when more CSV files are added to `data/raw/`.

---

## Adding More CSV Files

1. Place the new CSV file(s) in `backend/data/raw/`
2. Re-run the full pipeline from Step 1:

```
python app\ml\clean_dataset.py
python app\ml\label_dataset.py
python app\ml\train_random_forest.py
python app\ml\train_xgboost.py
python app\ml\train_lstm.py
```

The script will automatically discover, validate, and merge all CSV files
in `data/raw/` — no configuration changes required.

---

## File Summary

| File | Changed? | Reason |
|---|---|---|
| `clean_dataset.py` | **YES — rewritten** | Now handles multiple CSV files |
| `label_dataset.py` | **YES — minor** | Input path updated to `cleaned_dataset.csv`; uses `pathlib` |
| `train_random_forest.py` | No change | Still reads `labeled_dataset.csv` |
| `train_xgboost.py` | No change | Still reads `labeled_dataset.csv` |
| `train_lstm.py` | No change | Still reads `labeled_dataset.csv` |

No backend API files, frontend files, or Docker files were modified.

---

## Windows and Linux Compatibility

All path operations now use `pathlib.Path` instead of `os.path.join()`.
`pathlib.Path("data") / "raw"` works correctly on both Windows (backslash)
and Linux/Mac (forward slash).

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `No CSV files found in data/raw/` | Raw folder is empty | Add at least one CSV to `data/raw/` |
| `No valid CSV files could be loaded` | All files failed validation | Check the warning messages for each file |
| `column mismatch` warning on a file | That file has different columns | Either fix the CSV or leave it — it will be skipped |
| `cleaned_dataset.csv not found` in label step | clean_dataset.py not run yet | Run `clean_dataset.py` first |
