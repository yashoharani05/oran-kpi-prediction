# Architecture & System Flow Diagrams

---

## 1. System Architecture

This project has three layers: a data pipeline that prepares ML models, a REST API that serves predictions, and a web dashboard that displays results.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION  (run once)                      │
│                                                                      │
│  backend/data/raw/*.csv  (one or more CSV files)                     │
│         │                                                            │
│         ▼                                                            │
│  clean_dataset.py                                                    │
│    - Validates and merges all CSVs                                   │
│    - Drops identity/config columns (IMSI, RNTI, slice_id, etc.)      │
│    - Removes duplicates, fills missing values                        │
│    - Engineers prb_grant_ratio feature                               │
│    - Saves combined_dataset.csv + cleaned_dataset.csv                │
│         │                                                            │
│         ▼                                                            │
│  label_dataset.py                                                    │
│    - Scores each row against 7 quantile-based KPI conditions         │
│    - Labels rows: score >= 2 → Degraded (1), else Normal (0)         │
│    - Saves labeled_dataset.csv                                       │
│         │                                                            │
│         ▼                                                            │
│  train_random_forest.py  →  random_forest_model.pkl                  │
│  train_xgboost.py        →  xgboost_model.pkl                        │
│  train_lstm.py           →  lstm_model.keras + lstm_scaler.pkl       │
│                              comparison_report.json                  │
└──────────────────────────────────────────────────────────────────────┘
                                  │  trained models loaded at startup
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    BACKEND  (FastAPI, port 8000)                     │
│                                                                      │
│  GET  /api/health          →  status of all three models             │
│  POST /api/predict         →  single row prediction (18 KPI values)  │
│  GET  /api/comparison      →  RF vs XGBoost vs LSTM metrics JSON     │
│  GET  /api/stream/next     →  next CSV row + model prediction        │
│  GET  /api/stream/status   →  cursor position in dataset             │
│  GET  /api/stream/reset    →  rewind cursor to row 0                 │
└──────────────────────────────────────────────────────────────────────┘
                                  │  JSON over HTTP
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FRONTEND  (Next.js, port 3000)                    │
│                                                                      │
│  /       →  Manual Prediction Dashboard                              │
│              - 18 KPI input fields (grouped by downlink/uplink)      │
│              - Model selector: Random Forest | XGBoost | LSTM        │
│              - Probability gauge (SVG arc, green → amber → red)      │
│              - Alert box with recommendation text                    │
│              - Model comparison table + bar chart                    │
│              - How It Differs panel (explains all 3 models)          │
│                                                                      │
│  /live   →  Live Monitoring Dashboard                                │
│              - Polls GET /api/stream/next every 2 seconds            │
│              - 8 live KPI cards with colour coding                   │
│              - Scrolling line chart (throughput + error rate)        │
│              - Probability gauge + NORMAL/DEGRADED label             │
│              - Ground truth vs prediction panel                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Pipeline Flowchart (Multi-File Version)

```
START
  │
  ▼
Scan backend/data/raw/ for all .csv files
  │  Found: N files (one or more recording sessions)
  │
  ▼
Validate each file:
  │  - Skip if empty (0 bytes)
  │  - Skip if cannot be parsed as CSV
  │  - Skip if 0 data rows
  │  - Skip if columns do not match the first valid file
  │  (Valid files continue; skipped files print a warning)
  │
  ▼
Merge all valid files into one DataFrame
  │  e.g. 3 files × 6,000,000 rows = 18,000,000 combined rows
  │
  ▼
Save combined_dataset.csv  ← raw merge checkpoint
  │
  ▼
Drop completely empty columns (Unnamed separator artefacts)
  │
  ▼
Drop known identity/config columns (hardcoded list):
  │  IMSI, RNTI, num_ues, slicing_enabled, slice_id, slice_prb,
  │  power_multiplier, scheduling_policy, ul_rssi, dl_pmi, dl_ri,
  │  ul_n, tx_errors_downlink_pct
  │  WHY: These are experiment metadata, not radio performance.
  │  With multiple sessions they may no longer be zero-variance,
  │  so a hardcoded drop list is used instead of the variance check.
  │
  ▼
Drop any remaining zero-variance columns
  │  (columns still constant across ALL merged rows)
  │
  ▼
Remove duplicate rows
  │  WHY: Same moment may appear in overlapping recording sessions
  │
  ▼
Rename columns to snake_case
  │  e.g. "rx_errors uplink (%)" → "rx_errors_uplink_pct"
  │
  ▼
Convert timestamp → datetime index
  │  Unix milliseconds → Python datetime, sorted chronologically
  │
  ▼
Fill missing values with column medians
  │  Median used (not mean) — not affected by outliers
  │
  ▼
Validate ranges (check for negatives, CQI > 15, MCS > 28, etc.)
  │
  ▼
Engineer prb_grant_ratio feature
  │  = sum_granted_prbs / (sum_requested_prbs + 1)
  │  Clamp infinity values: replace inf with median, clip to [0, 2]
  │  WHY +1: prevents divide-by-zero when UE is idle
  │
  ▼
Sort by timestamp → set as row index
  │
  ▼
Print preprocessing summary:
  │  Files found | Files loaded | Files skipped
  │  Rows before merge | Duplicates removed | Rows after cleaning
  │  Columns in cleaned dataset | Missing values remaining
  │
  ▼
Save cleaned_dataset.csv
  │
  ▼
Score each row against 7 KPI conditions (quantile-based):
  │
  │  Each bad condition adds 1 point to the degradation score:
  │    1. rx_errors_uplink_pct > Q90   (worst 10% error rate)
  │    2. dl_cqi               < Q25   (worst 25% channel quality)
  │    3. tx_brate_downlink    < Q10   (worst 10% throughput)
  │    4. prb_grant_ratio      < Q25   (worst 25% bandwidth satisfaction)
  │    5. dl_mcs               < Q25   (worst 25% modulation quality)
  │    6. ul_sinr < Q25  (only when UE is actively transmitting)
  │    7. ul_turbo_iters > Q75 (only when UE is actively transmitting)
  │
  │  Thresholds computed from the MERGED dataset — self-adapting.
  │  Adding more CSV files automatically recalibrates them.
  │
  ▼
Assign label:
  │  score >= 2  →  Degraded (1)
  │  score  < 2  →  Normal   (0)
  │
  │  WHY require 2: one bad reading can happen for innocent reasons.
  │  Two simultaneous bad KPIs = stronger, more reliable evidence.
  │
  ▼
Save labeled_dataset.csv
  │  Approx. 71% Normal, 29% Degraded (varies by dataset)
  │
  ▼
Train 3 ML models (each reads labeled_dataset.csv):
  │
  ├── Random Forest  →  100 trees in parallel, majority vote
  │                      saved: random_forest_model.pkl
  │
  ├── XGBoost        →  100 trees sequentially, each corrects the last
  │                      saved: xgboost_model.pkl
  │                             comparison_report.json (RF vs XGB)
  │
  └── LSTM           →  20-step sliding window, memory-efficient generator
                         saved: lstm_model.keras
                                lstm_scaler.pkl
                                comparison_report.json (all 3 models)
  │
  ▼
END — run uvicorn main:app --reload to start the API server
```

---

## 3. Prediction Request Flow (Manual Dashboard)

```
User enters 18 KPI values in the dashboard form
  │
  ▼
Selects model: [Random Forest] [XGBoost] [LSTM]
  │
  ▼
Clicks "Predict Network Risk" button
  │
  ▼
Axios sends:
  POST /api/predict?model=xgboost
  Body: { "dl_mcs": 9.6, "dl_cqi": 7.0, ... (18 fields) }
  │
  ▼
Next.js rewrites to: http://localhost:8000/api/predict?model=xgboost
(or http://backend:8000/api/predict in Docker)
  │
  ▼
FastAPI validates request body using Pydantic (KpiInput schema)
  │  All 18 fields required. Wrong type → 422 error returned automatically.
  │
  ▼
Route handler retrieves model from app.state.models["xgboost"]
  │  (Model was loaded into RAM at server startup — no disk access here)
  │
  ▼
Build named pandas DataFrame (1 row × 18 columns)
  │
  │  For LSTM only:
  │    Scale with lstm_scaler.transform(row)
  │    Replicate row 20 times → shape (1, 20, 18)
  │    (Simplification — a rolling buffer would be more accurate)
  │
  ▼
model.predict(X)         →  risk_code  ∈ {0, 1}
model.predict_proba(X)   →  probability ∈ [0.0, 1.0]
  │
  ▼
get_recommendation(risk_code, probability)
  │  < 0.35 + Normal   → "No action required"
  │  0.35–1.0 + Normal → "Continue monitoring"
  │  0.50–0.60 + Degraded → "LOW RISK"
  │  0.60–0.85 + Degraded → "MODERATE RISK"
  │  >= 0.85 + Degraded   → "HIGH RISK: Immediate investigation"
  │
  ▼
Return JSON:
  {
    "risk_label":     "Degraded",
    "risk_code":      1,
    "probability":    0.96,
    "recommendation": "HIGH RISK: Network degradation detected...",
    "model_used":     "XGBoost"
  }
  │
  ▼
Frontend re-renders:
  - Probability gauge arc fills to 96% (red)
  - "DEGRADED" label appears in red, 4xl font
  - Alert box pulses red with recommendation text
  - KPI cards colour coding updates
```

---

## 4. Live Stream Flow (Monitoring Page)

```
User opens /live and clicks "Start Monitoring"
  │
  ▼
React: setInterval starts — fires every 2000 milliseconds
  │
  ▼  (every 2 seconds)
  ▼
Frontend calls: GET /api/stream/next
  │
  ▼
Backend: StreamState reads df.iloc[cursor]
  │  StreamState holds the entire labeled_dataset.csv in RAM
  │  (loaded at server startup, never re-read from disk)
  │
  ▼
cursor = (cursor + 1) % total_rows
  │  Wraps back to 0 after the final row — loops forever
  │
  ▼
Build feature DataFrame from row values
  │
  ▼
model.predict(X) + model.predict_proba(X)
  │
  ▼
Returns StreamRow JSON:
  {
    "row_index":      42,
    "timestamp":      "2021-03-30 02:15:33",
    "total_rows":     19053569,
    "dl_mcs":         9.63,
    "dl_cqi":         7.0,
    "rx_errors_uplink_pct": 0.0,
    ... (all 18 KPI fields) ...
    "actual_risk":    0,      ← ground truth from label_dataset.py
    "predicted_risk": 0,      ← what the model predicts NOW
    "probability":    0.23,
    "recommendation": "Network is operating normally..."
  }
  │
  ▼
Frontend state updates:
  currentRow = response
  history.push(response)  ← keeps last 60 rows for the line chart
  stats.total += 1
  stats.normal or stats.degraded += 1
  │
  ▼
React re-renders all elements simultaneously:
  - 8 KPI cards update with new values + colour coding
  - Line chart appends new throughput + error rate points (shows 30)
  - Probability gauge animates to new probability value
  - NORMAL/DEGRADED label updates
  - Alert box shows new recommendation
  - Session stats counters increment
  - Progress bar moves forward (row_index / total_rows × 100%)
  - Ground truth panel shows actual vs predicted + ✓ or ✗
  │
  ▼
Wait 2000 ms → repeat from top
  │
  User clicks "Pause" → clearInterval() → updates stop
  User clicks "Reset" → GET /api/stream/reset → cursor = 0
                      → history cleared, stats reset to zero
```

---

## 5. Model Architecture Comparison

```
RANDOM FOREST                 XGBOOST                    LSTM
──────────────────────        ──────────────────────     ──────────────────────
Input: 1 row (18 features)    Input: 1 row (18 features) Input: 20 rows (20×18)
No feature scaling needed     No feature scaling needed   MinMaxScaler → [0,1]
         │                             │                          │
  ┌──────┴──────┐             ┌────────┴───────┐          ┌──────┴──────┐
  │  Tree 1     │  ← random   │    Tree 1      │          │  LSTM(32)   │
  │  Tree 2     │    subset   │       ↓        │          │  32 memory  │
  │  Tree 3     │             │  (errors→)     │ boost    │  cells      │
  │  ...        │  parallel   │    Tree 2      │ ──────   │      ↓      │
  │  Tree 100   │  training   │       ↓        │          │ Dropout(0.2)│
  └─────────────┘             │    Tree N      │          │      ↓      │
         │                    └────────────────┘          │  Dense(16)  │
  100 trees vote              weighted sum of all         │      ↓      │
  majority wins               tree predictions            │  Dense(1)   │
         │                             │                  └─────────────┘
       0 or 1                        0 or 1                      │
                                                          sigmoid → 0.0–1.0
                                                          >= 0.5 → Degraded

Settings:                     Settings:                   Settings:
  n_estimators=100              n_estimators=100            window_size=20
  max_depth=None                max_depth=4                 batch_size=512
  class_weight='balanced'       learning_rate=0.1           epochs=30 (max)
                                scale_pos_weight=2.38       early_stopping=5
                                                            Generator: 750KB/batch
                                                            (not 51GB all at once)

F1: 97.37%                    F1: 99.14%                  F1: 31.65% (*)

(*) Lower because 20-step window = only 5 seconds of history.
    LSTM needs hours of sequences to learn gradual trends.
    Precision is still 100% — zero false alarms.
```

---

## 6. KPI Feature Importance (Random Forest)

The Random Forest reveals which of the 18 KPIs matter most for predicting degradation. These importances are computed from how much each feature reduced impurity across all 100 trees.

```
Rank  Feature                   Importance   Bar
────  ─────────────────────     ──────────   ───────────────────────
  1   dl_mcs                    22.97%       ████████████████████████
  2   dl_cqi                    17.44%       ██████████████████
  3   ul_turbo_iters            10.08%       ██████████
  4   sum_granted_prbs           7.06%       ███████
  5   prb_grant_ratio            6.56%       ███████   ← engineered feature!
  6   ul_sinr                    5.90%       ██████
  7   rx_errors_uplink_pct       5.13%       █████
  8   dl_n_samples               4.88%       █████
      (remaining 10 features)   20.00%
```

Key observations:
- dl_mcs and dl_cqi are the top 2 — both measure downlink channel quality directly
- prb_grant_ratio ranks 5th, confirming the engineered feature added genuine value
- rx_errors_uplink_pct ranks only 7th despite being the most obvious degradation signal,
  because MCS and CQI are earlier warning signs that change before errors peak

---

## 7. Docker Networking Diagram

```
Your PC (Windows/Linux/Mac)
  │
  ├── Browser at http://localhost:3000
  │        │
  │        │  HTTP requests
  │        ▼
  │   ┌──────────────────────────────────────┐
  │   │  Docker private network: oran-network│
  │   │                                      │
  │   │  ┌─────────────────────┐             │
  │   │  │  oran-frontend      │  port 3000  │
  │   │  │  (Node.js, Next.js) │ ◄───────────┼── browser
  │   │  │                     │             │
  │   │  │  /api/* rewrites to │             │
  │   │  │  http://backend:8000│             │
  │   │  └──────────┬──────────┘             │
  │   │             │ internal HTTP          │
  │   │             ▼                        │
  │   │  ┌─────────────────────┐             │
  │   │  │  oran-backend       │  port 8000  │
  │   │  │  (Python, FastAPI)  │ ◄───────────┼── http://localhost:8000/docs
  │   │  │                     │             │
  │   │  │  3 ML models in RAM │             │
  │   │  │  labeled_dataset.csv│             │
  │   │  └─────────────────────┘             │
  │   │                                      │
  │   │  Shared volumes (survive restarts):  │
  │   │    ./backend/data/raw    ← your CSVs │
  │   │    ./backend/data/processed          │
  │   │    ./backend/models    ← .pkl files  │
  │   └──────────────────────────────────────┘
```

"backend" in the URL is resolved by Docker's internal DNS to the backend container's IP. This is why next.config.js uses NEXT_PUBLIC_BACKEND_URL=http://backend:8000 in Docker mode.
