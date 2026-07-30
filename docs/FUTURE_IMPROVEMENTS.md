# Future Improvements

Realistic suggestions for extending this project, ordered from most impactful to most ambitious.

---

## High Priority (addresses current limitations directly)

### 1. Rolling Buffer for LSTM API Predictions

Currently, the `/api/predict?model=lstm` endpoint replicates one row 20 times to fill the window. This means LSTM gets no real temporal context and its predictions are no better than a single-row model.

**Fix:** Maintain a `collections.deque(maxlen=20)` in the backend. Each call to `/stream/next` appends the new row. When the buffer has 20 rows, pass it as a genuine sequence to LSTM.

```python
from collections import deque
lstm_buffer = deque(maxlen=20)
lstm_buffer.append(current_row_scaled)
if len(lstm_buffer) == 20:
    X_seq = np.array(list(lstm_buffer))[np.newaxis, :]
    prob = lstm_model.predict(X_seq)[0][0]
```

### 2. Larger LSTM Window Size

The current 20-step window covers only 5 seconds. Increase `WINDOW_SIZE` in `train_lstm.py` to 200 (50 seconds) or 500 (2 minutes). The generator-based training handles this without memory issues — the batch memory stays the same regardless of window size.

### 3. SHAP Explanations

SHAP (SHapley Additive exPlanations) explains individual predictions. Instead of just "Degraded", the API could say "because dl_mcs was in the bottom 10% and rx_errors_uplink_pct was in the top 5%".

```python
pip install shap
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)
```

Add a new endpoint: `GET /api/explain?model=random_forest` that returns feature-level contributions for the last prediction.

---

## Medium Priority (meaningful improvements with moderate effort)

### 4. Confidence Threshold Slider

Add a slider on the dashboard to adjust the decision threshold from 0.5 (default) to lower values like 0.3. This trades more false alarms for fewer missed events. Useful for demonstrating the precision/recall trade-off interactively.

### 5. Hyperparameter Tuning

All three models use standard starting-point hyperparameters. A systematic search would likely improve performance:

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 2, 5],
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="f1")
```

### 6. Automated Retraining Endpoint

Add `POST /api/retrain?model=random_forest` that re-runs the training pipeline without restarting the server. Useful for keeping models up to date as new CSV files are added to `data/raw/`.

### 7. Prediction History Export

A "Download History" button on the live monitoring page that exports all predictions made in the current session as a CSV file, including timestamp, all 18 KPI values, actual_risk, predicted_risk, and probability.

---

## Longer Term (significant additional work)

### 8. Real E2 Interface Connection

Replace the CSV stream with a genuine connection to a Near-RT RIC via the E2 interface using OpenAirInterface or srsRAN. This would convert the project from a simulation into a real xApp. Requires physical radio hardware and a server running the RIC software.

### 9. Multi-Class Labels

Replace the binary Normal/Degraded label with four classes: Normal, Warning, Degraded, Critical. This gives operators more graduated alerts and reduces alarm fatigue from jumping straight from Normal to Degraded.

### 10. Anomaly Detection Alternative

Replace the manually engineered label with an unsupervised Isolation Forest or Autoencoder. The model would discover degradation patterns from the data itself, removing the subjectivity of the current quantile-based approach.

### 11. Multi-UE Monitoring

Collect data from multiple UEs simultaneously. Show a summary view of all UEs on the dashboard, with per-UE gauges and the ability to click into any UE for detailed monitoring.

### 12. API Authentication

Add API key middleware before any external deployment:

```python
from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")
```

Prevents unauthorized access to the prediction endpoints.
