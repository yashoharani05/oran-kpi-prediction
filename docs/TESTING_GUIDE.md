# Testing Guide

How to verify each part of the project is working correctly.

---

## 1. Verify the ML Pipeline Output

After running all five training scripts, check these files exist:

```
backend\data\processed\combined_dataset.csv     ← raw merged checkpoint
backend\data\processed\cleaned_dataset.csv      ← fully cleaned
backend\data\processed\labeled_dataset.csv      ← with labels
backend\models\random_forest_model.pkl
backend\models\xgboost_model.pkl
backend\models\lstm_model.keras
backend\models\lstm_scaler.pkl
backend\models\comparison_report.json
```

Check the comparison report:
```
type backend\models\comparison_report.json
```

You should see three model entries with F1 scores. Values will depend on your dataset size.

---

## 2. Quick Pipeline Smoke Test

Run this from the `backend/` folder (venv active) to check imports and model loading:

```
python -c "
import sys; sys.path.insert(0, '.')
from app.utils.model_loader import load_all_models
models = load_all_models()
print('Loaded:', {k: v is not None for k, v in models.items()})
"
```

Expected output:
```
{'random_forest': True, 'xgboost': True, 'lstm': True, 'lstm_scaler': True}
```

---

## 3. Test the Backend API

### Health check
Open `http://localhost:8000/api/health`

Expected:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "RF + XGB + LSTM",
  "message": "Loaded: RF=yes  XGB=yes  LSTM=yes"
}
```

If any model shows `no`, check startup logs in the uvicorn terminal.

### Test healthy prediction
Open `http://localhost:8000/docs` → POST /api/predict → Try it out → Execute

Expected:
```json
{ "risk_label": "Normal", "risk_code": 0, "probability": 0.0, ... }
```

### Test degraded prediction
In the Swagger UI, change these four fields before executing:
```json
{
  "dl_mcs": 0.18,
  "dl_cqi": 4.5,
  "rx_errors_uplink_pct": 100.0,
  "tx_brate_downlink_mbps": 0.003
}
```

Expected:
```json
{ "risk_label": "Degraded", "risk_code": 1, "probability": 1.0, ... }
```

### Test XGBoost and LSTM
Add `?model=xgboost` or `?model=lstm` to the URL and repeat both tests.

### Test stream endpoints
```
http://localhost:8000/api/stream/reset   → {"message": "Stream reset to row 0."}
http://localhost:8000/api/stream/next    → StreamRow JSON with row_index: 0
http://localhost:8000/api/stream/next    → StreamRow JSON with row_index: 1
http://localhost:8000/api/stream/status  → {"cursor": 2, "total_rows": 19053569, ...}
http://localhost:8000/api/comparison     → {"models": [...]} with three entries
```

---

## 4. Test the Frontend

### Home page
Open `http://localhost:3000`

Checklist:
- [ ] "Network Degradation Dashboard" title visible
- [ ] Header shows "Model Ready" in green
- [ ] Three model buttons: Random Forest, XGBoost, LSTM
- [ ] "Live Monitoring" button visible

### Healthy prediction flow
1. Click "Load Healthy Sample"
2. KPI cards should show mostly sky-blue values
3. Click "Predict Network Risk"
4. Expected: green alert box, NORMAL label, gauge near 0%, "No action required"

### Degraded prediction flow
1. Click "Load Degraded Sample"
2. UL Error Rate card should be red (100%)
3. Click "Predict Network Risk"
4. Expected: red pulsing alert box, DEGRADED label, gauge high, HIGH RISK recommendation

### Model switching
1. Predict with Random Forest → note result
2. Click XGBoost → prediction clears
3. Predict again → `model_used` should say "XGBoost"
4. Repeat for LSTM

### Model comparison panel
Scroll to bottom of home page:
- [ ] Table shows 3 rows (RF, XGBoost, LSTM)
- [ ] XGBoost has the best F1 score
- [ ] Bar chart is visible

### Live monitoring page
1. Open `http://localhost:3000/live`
2. Click "Start Monitoring"
3. After 2 seconds: KPI cards update, row_index = 1
4. After 10 seconds: line chart shows ~5 points, stats show 5 rows seen
5. Click "Pause" — updates stop
6. Click "Reset" — counters and chart clear, progress bar resets
7. Start again — row_index starts from 0

---

## 5. Expected Results

| Test | Expected |
|---|---|
| Healthy values → predict | risk_label: "Normal", probability: 0.0–0.3 |
| Degraded values → predict | risk_label: "Degraded", probability: 0.85–1.0 |
| RF vs XGBoost on same input | Same label, slightly different probability |
| LSTM on healthy values | risk_label: "Normal", probability varies |
| /stream/next repeated calls | row_index increments by 1 each call |
| /stream/next after last row | row_index wraps back to 0 |

---

## 6. Full End-to-End Test (start fresh)

1. Stop the backend server (Ctrl+C)
2. Delete processed files and models:
   ```
   del backend\data\processed\*.csv
   del backend\models\*.pkl
   del backend\models\*.keras
   del backend\models\*.json
   ```
3. Re-run the full pipeline:
   ```
   python app\ml\clean_dataset.py
   python app\ml\label_dataset.py
   python app\ml\train_random_forest.py
   python app\ml\train_xgboost.py
   python app\ml\train_lstm.py
   ```
4. Start backend: `uvicorn main:app --reload`
5. Run a degraded prediction on the dashboard
6. Verify DEGRADED with high probability

---

## 7. Known Expected Behaviours (Not Bugs)

**LSTM probability varies on single-row input.**
The API replicates one row 20 times to fill the LSTM window — it is not using genuine temporal context. This is documented as a limitation. LSTM results on the prediction page are less reliable than RF and XGBoost.

**clean_dataset.py drops different columns depending on your files.**
With multiple sessions, columns that were constant in one session may now vary. The pipeline uses a `COLUMNS_TO_ALWAYS_DROP` hardcoded list to ensure identity columns (IMSI, RNTI, slice_id, etc.) are always removed. If you see these columns in your training features list, you have the old version of `clean_dataset.py`.

**LSTM training is slow with 19M rows.**
The generator-based training shows progress per epoch. This is normal. Each epoch may take several minutes depending on CPU speed.

**The stream always uses Random Forest.**
The `/api/stream/next` endpoint uses whichever of RF or XGBoost is available, defaulting to RF. It does not support model switching via query parameter. This is intentional — the stream was designed for automatic monitoring, not interactive model comparison.

**Rows near the start of the merged dataset may be labelled Degraded.**
The scoring thresholds are computed from the entire merged dataset, not per-file. Rows that just cross the score=2 boundary (borderline cases) may appear in unexpected places.
