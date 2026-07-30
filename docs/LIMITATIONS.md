# Limitations

An honest account of what this project cannot do and why. Being aware of limitations demonstrates academic maturity and is expected in a university submission.

---

## 1. Simulated Live Stream (not a real O-RAN connection)

The live monitoring page replays the labeled CSV one row at a time — it is not connected to a real base station. A real O-RAN xApp would subscribe to the E2 interface, receive live RIC Indication messages every 250ms, and could send back RIC Control messages to adjust the scheduler. None of this is implemented.

**Impact:** The project demonstrates the ML models and the dashboard, but does not constitute a deployable xApp.

**Mitigation:** Implement a real E2 interface using OpenAirInterface or srsRAN with a Near-RT RIC.

---

## 2. LSTM Receives Replicated Single-Row Input

The `/api/predict?model=lstm` endpoint receives one row and replicates it 20 times to fill the sequence window. This means the LSTM is not using any temporal context — it is predicting from 20 identical rows, which is functionally equivalent to a single-row snapshot.

**Impact:** LSTM predictions via the manual dashboard are not meaningfully different from RF/XGBoost predictions.

**Mitigation:** Maintain a rolling buffer of the last 20 real readings in the API. Each call to `/stream/next` appends to this buffer and passes it to LSTM.

---

## 3. LSTM Low Recall on Short Windows

Even with 19 million rows, each LSTM window covers only 20 rows × 250ms = 5 seconds of history. LSTM needs windows spanning minutes or hours to learn gradual degradation trends. The 20-step window size is the binding constraint, not total dataset size.

**Impact:** LSTM recall is low (≈19%) — it catches roughly 1 in 5 degradation events.

**Mitigation:** Increase WINDOW_SIZE in `train_lstm.py` to 200–500 (representing 50–125 seconds), retrain, and monitor memory and training time.

---

## 4. Student-Defined Labels

The `degradation_risk` label was engineered by the student using quantile thresholds. Different threshold choices produce different labels and different model performance numbers. There is no ground truth to validate against.

**Impact:** Models are predicting the student's definition of degradation, not an independently verified one.

**Mitigation:** Have a radio network engineer review the thresholds, or obtain a dataset where operators have manually labelled degradation events.

---

## 5. Multi-File Data from Same Testbed

The multiple CSV files all come from the same srsRAN software-defined radio testbed with a single UE setup. They represent different recording sessions but not fundamentally different network conditions.

**Impact:** The merged dataset is larger but not necessarily more diverse. Multi-UE scenarios, handover events, and real-world interference patterns are not represented.

**Mitigation:** Collect data from different testbed configurations and real 5G deployments.

---

## 6. No Authentication on the API

The API has no authentication or authorisation. Any process that can reach port 8000 can make predictions or stream data.

**Impact:** Acceptable for a local university project. Not safe for internet deployment.

**Mitigation:** Add API key middleware or OAuth2 before any external deployment.

---

## 7. No Automated Tests

The project has no unit tests (pytest) or frontend tests (Jest). All verification was manual.

**Mitigation:** Add `pytest` tests for each ML script's key functions, and Jest tests for the React components.

---

## 8. Model Hyperparameters Not Tuned

All three models use default or standard starting-point hyperparameters. No systematic search was done.

**Mitigation:** Use `GridSearchCV` for RF and XGBoost. Use Keras Tuner or Optuna for LSTM.

---

## 9. No Concept Drift Handling

Models are trained once and never updated. If network conditions change significantly (new interference, firmware update), predictions may degrade over time.

**Mitigation:** Implement periodic retraining with new data, or add a `/api/retrain` endpoint.

---

## Summary Table

| Limitation | Severity | Mitigation |
|---|---|---|
| Simulated stream, no real E2 interface | High | Implement E2AP connection |
| LSTM uses replicated single row | High | Rolling buffer in API |
| LSTM low recall (short window) | High | Increase WINDOW_SIZE |
| Student-defined labels | Medium | Expert validation |
| Same testbed, single UE | Medium | More diverse data collection |
| No API authentication | Low (local) | API key middleware |
| No automated tests | Low | pytest + Jest |
| Untuned hyperparameters | Low | GridSearchCV / Optuna |
| No concept drift handling | Low | Periodic retraining |
