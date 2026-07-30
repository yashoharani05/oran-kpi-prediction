# Viva Questions and Suggested Answers

---

## Background and Motivation

**Q: What is O-RAN and why does it matter for this project?**

O-RAN stands for Open Radio Access Network. Traditional 5G equipment is proprietary — you buy everything from one vendor. O-RAN breaks this using open standards, letting different vendors' hardware and software work together. The key component here is the Near-RT RIC (RAN Intelligent Controller), which hosts small applications called xApps that can monitor and influence the network in near real-time. This project builds the ML and dashboard components of such an xApp.

---

**Q: Is this project connected to a real 5G network?**

No, but it uses real 5G data. The dataset was collected from a real srsRAN software-defined radio testbed — actual radio signals between a simulated base station and a simulated phone. In the project, we replay the CSV row by row to simulate a live feed. A real E2 interface connection requires physical radio hardware and is listed as future work.

---

**Q: What is a KPI in this context?**

A Key Performance Indicator is a radio performance measurement. The base station records 18 of them every 250 milliseconds. Examples: dl_cqi (signal quality rated 0–15 by the phone itself), rx_errors_uplink_pct (percentage of corrupted packets from the phone), and tx_brate_downlink_mbps (actual download speed). These 18 values together describe the health of the radio link at any given moment.

---

## Dataset and Preprocessing

**Q: Why do you have multiple CSV files and how do you handle them?**

Each CSV file represents one recording session from the testbed. By merging them we get more training data, which generally leads to better models. The `clean_dataset.py` script scans `data/raw/`, validates each file (checking it is not empty, parseable, and has matching columns), merges all valid files with `pd.concat`, removes duplicate rows that appear in overlapping sessions, and applies the full cleaning pipeline to the combined data.

---

**Q: Why did you drop columns like IMSI and slice_id with a hardcoded list instead of the zero-variance check?**

With a single CSV, those columns never changed, so the zero-variance check removed them. With multiple recording sessions, different experiments used different settings — so IMSI might still be constant but slice_id might now have two values. The zero-variance check would keep slice_id, but it should never be an ML feature because it is experiment configuration, not radio performance. The hardcoded `COLUMNS_TO_ALWAYS_DROP` list ensures these 13 identity and config columns are always removed regardless of how many files are merged.

---

**Q: How did you create the degradation_risk label?**

The raw data has no pre-existing label. I created one using a 7-condition scoring system based on domain knowledge. Each row is checked against seven KPI thresholds — for example, error rate above the 90th percentile, CQI below the 25th percentile, throughput below the 10th percentile. Each bad condition adds 1 point. A score of 2 or more means Degraded. I used quantile thresholds rather than hardcoded numbers so they adapt to the merged dataset's own distribution — adding more files automatically recalibrates what counts as "bad".

---

**Q: Why require 2 conditions for Degraded instead of 1?**

A single bad reading can be innocent. Low throughput might mean the UE has nothing to send. Low CQI for one sample might be a measurement blip. Requiring two simultaneous bad KPIs is far more reliable — it mirrors how real engineers diagnose problems, looking for correlated evidence rather than one-off anomalies. It also explains why all three models achieve 100% Precision: the labels are conservative, and the models learn that conservative boundary.

---

**Q: Why did you not remove any rows during cleaning?**

Zero-throughput rows represent idle state — a real and important condition. 100%-error rows represent the worst degradation events — exactly what the model needs to learn to detect. Removing them would make the model blind to the very states we are trying to predict. The cleaning step removes empty and constant columns, but all rows are kept.

---

## Machine Learning

**Q: What is the difference between Random Forest and XGBoost?**

Both build decision trees, but differently. Random Forest builds 100 trees in parallel, each on a random data subset, then takes a majority vote. XGBoost builds trees sequentially — each new tree specifically targets the rows the previous trees got wrong. This "boosting" mechanism corrects its own mistakes iteratively. On tabular data, XGBoost almost always wins because it learns from its errors. In this project, XGBoost missed only 2 degradation events versus Random Forest's 6.

---

**Q: Why does LSTM have low recall even with 19 million rows?**

More rows did not help because the window size is still only 20 rows covering 5 seconds. With 19 million rows from many sessions, the data is richer overall, but each individual LSTM prediction still sees only 5 seconds of history. LSTM is designed to learn patterns that develop over minutes or hours — things like CQI slowly dropping over 30 seconds before an error spike. A 5-second window is not enough context for those patterns. The total data size helps general coverage, but the temporal window is the binding constraint.

---

**Q: What does class_weight="balanced" do?**

The dataset is approximately 71% Normal and 29% Degraded. Without correction, a model could achieve 71% accuracy by always predicting Normal — never learning to detect degradation at all. "balanced" adjusts the training so the minority class (Degraded) gets proportionally more influence: weight = total_samples / (n_classes × class_count). XGBoost uses an equivalent called scale_pos_weight, set to Normal_count / Degraded_count ≈ 2.4.

---

**Q: Why no shuffle in the train/test split?**

This is time-series data. If we shuffled before splitting, a training row from time T+100 could sit next to a test row from T+99. The model would effectively see the future during training — called data leakage. shuffle=False preserves chronological order, so the first 80% of rows train the model and the last 20% test it. This simulates real deployment: train on historical data, predict on future data.

---

**Q: What is overfitting and how did you prevent it?**

Overfitting is when a model memorises training data rather than learning general patterns — it scores perfectly on training data but poorly on new data. Prevention: 80/20 train/test split without shuffling (model never sees test data during training). For Random Forest: 100 independent trees cancel each other's noise. For XGBoost: shallow depth (max_depth=4) and conservative learning rate (0.1). For LSTM: Dropout(0.2) randomly zeros 20% of outputs during training, and EarlyStopping reverts to the best weights if validation loss stops improving.

---

**Q: Why did the original LSTM crash with a memory error, and how did you fix it?**

The original code tried to pre-build all sliding-window sequences at once: 19 million rows × 20 steps × 18 features × 4 bytes = 27 GB RAM. No normal PC can allocate that. The fix uses a Keras Sequence generator — a class that creates only one batch of 512 windows at a time (about 750 KB), feeds it to the model, and discards it. Keras calls the generator automatically during training. Memory usage stays around 750 KB per batch regardless of how large the dataset is.

---

## System Design

**Q: How does the live stream work technically?**

At startup, the server loads the entire labeled_dataset.csv into a pandas DataFrame and stores an integer cursor. When the frontend calls GET /api/stream/next, the backend reads df.iloc[cursor], runs the model on that row, increments the cursor (wrapping back to 0 at the end), and returns everything as JSON. The frontend uses setInterval to call this endpoint every 2000 ms. No WebSocket, no database, no message queue — just a DataFrame in RAM and a counter.

---

**Q: Why FastAPI and not Flask?**

FastAPI auto-generates an interactive Swagger UI at /docs — every endpoint, parameter, and response is documented without any extra code. It also validates request bodies automatically via Pydantic: if any of the 18 KPI fields is missing or the wrong type, a clear 422 error is returned before the prediction code runs. Flask would need separate validation and documentation libraries.

---

**Q: What is CORS and why does the backend need it?**

Cross-Origin Resource Sharing is a browser security feature. Browsers block JavaScript from calling a different port or domain than the page was loaded from. The frontend runs on port 3000 and calls the backend on port 8000. Without the CORS middleware in main.py, the browser would block all requests. The middleware adds a header telling the browser that port 3000 is allowed to make requests.

---

**Q: What does actual_risk vs predicted_risk mean in the stream?**

actual_risk is the ground truth label that label_dataset.py assigned — based on the quantile scoring rules applied to the CSV data. predicted_risk is what the currently loaded ML model computes fresh from the 18 KPI values right now, at the moment the /stream/next request arrives. They usually match. When they differ, the Ground Truth panel shows ✗ Mismatch — these are the model's false negatives (said Normal, was Degraded) or false positives (said Degraded, was Normal).

---

## Reflection

**Q: What would you improve first with more time?**

Three things: First, a rolling buffer for LSTM — currently the API replicates one row 20 times, which means LSTM gets no real temporal context. A genuine 20-row buffer of recent measurements would make the LSTM predictions meaningful. Second, SHAP explanations — not just "Degraded" but "because dl_mcs was in the bottom 25% and rx_errors_uplink_pct was above the 90th percentile." Third, increase the LSTM window size to 200+ rows to give it enough history to learn gradual trends.

---

**Q: What surprised you most about this project?**

Two things. First, the feature importance results — I expected rx_errors_uplink_pct (error rate) to rank first because it is the most visible sign of degradation. But dl_mcs and dl_cqi ranked 1 and 2. This makes sense in hindsight: the base station lowers the MCS before errors become severe, so MCS is an earlier warning signal. Second, the LSTM memory crash. I did not anticipate that 19 million rows × 20 window × 18 features would require 27 GB of RAM. Learning to use a generator was the most technically challenging part of the project.

---

## Live Data and Real 5G

**Q: Why did you not use live data from a real 5G network?**

The data I used was collected from a real srsRAN software-defined radio testbed — so it is genuine 5G radio measurements, not made-up or synthetic numbers. The only difference from a fully live system is that I replay it from a saved CSV file rather than receiving it from a running antenna in real time. The ML models, the prediction logic, and the dashboard all work exactly the same way they would with a live feed.

Connecting to a live 5G base station requires physical SDR hardware — devices like a USRP B200 which cost around £500 to £1000 — plus a Linux server running srsRAN and a test SIM card. That hardware was not available to me as an individual student. Implementing the live E2 interface connection is listed as future work. The backend is already designed to support it — the only change needed would be replacing the CSV row reader in `stream.py` with a file reader that watches the live srsRAN log file for new rows.

---

**Q: So is the live monitoring just fake then?**

No. The predictions are completely real. The ML model genuinely analyses each row of KPI values and makes a real decision — Normal or Degraded — based on everything it learned during training. What is simulated is only the data source: a saved file instead of a live antenna. Think of it like a weather model being tested against last year's weather data — the model and its predictions are real, the test uses historical recordings rather than a live atmosphere.

This approach is also the standard methodology in network ML research. Papers published in IEEE journals on 5G network management follow exactly the same workflow — train on a collected dataset, validate on a test split, describe live deployment as future work. My project follows that same academically accepted approach.

---

**Q: What would you need to make it fully live?**

Three things. First, SDR hardware — a USRP B200 or similar device that acts as a real radio transmitter and receiver. Second, a Linux server running srsRAN with a test SIM or simulated UE connected to generate live KPI measurements. Third, a small code change in `stream.py` — instead of reading from `df.iloc[cursor]`, the backend would watch the srsRAN log file for new lines using Python's file tail functionality, parse each new line into a row of 18 KPI values, and pass it to the model. The rest of the system — the FastAPI endpoints, the dashboard, the gauge, the alert box — would all work without any changes at all.

---

**Q: If it is so easy to change, why did you not just do it?**

Because the hardware is the blocker, not the code. I cannot write code that reads from a radio antenna I do not have. The CSV replay approach lets me demonstrate the full system — data ingestion, model inference, real-time dashboard — using real radio measurements without needing £1000 of lab equipment. The software architecture is production-ready; it just needs hardware to attach to.
