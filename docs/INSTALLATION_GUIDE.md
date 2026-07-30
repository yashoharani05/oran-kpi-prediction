# Installation Guide

**Platform:** Windows 10/11
**Time required:** 20–40 minutes (first time, depending on download speed)

---

## What You Need to Install

| Software | Version | Where to get it |
|---|---|---|
| Python | 3.10–3.12 (not 3.13) | https://www.python.org/downloads/ |
| Node.js | 18 or higher | https://nodejs.org/en (choose LTS) |
| Git (optional) | Any | https://git-scm.com/downloads |

> **Important:** TensorFlow does not yet support Python 3.13. Install Python 3.11 or 3.12.

---

## Step 1 — Install Python

1. Download Python 3.11 from https://www.python.org/downloads/
2. Run the installer
3. On the first screen, tick **"Add Python to PATH"** before clicking Install
4. Verify:
   ```
   python --version
   ```
   Expected: `Python 3.11.x`

---

## Step 2 — Install Node.js

1. Download Node.js LTS from https://nodejs.org/en
2. Run the installer (keep default settings)
3. Verify:
   ```
   node --version
   npm --version
   ```

---

## Step 3 — Get the Project

If you have Git:
```
git clone <your-repo-url>
cd fyp-oran-kpi-prediction
```

Or download and extract the ZIP.

---

## Step 4 — Set Up the Backend

Open **Command Prompt** in the `backend/` folder.

### Create a virtual environment
```
py -3.12 -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

> Every time you open a new terminal for this project, run `venv\Scripts\activate` first.

### Install Python packages
```
pip install -r requirements.txt
```

> TensorFlow is large (~500 MB). First install takes 5–15 minutes.

---

## Step 5 — Place Your Raw CSV Files

Put all your raw KPI CSV files inside:
```
backend\data\raw\
```

You can have one file or many files — the pipeline will find and merge all of them automatically. Each file must have the same column structure (the script will warn you and skip any that do not match).

---

## Step 6 — Run the ML Pipeline

Still in the `backend/` folder with venv active:

```
python app\ml\clean_dataset.py
python app\ml\label_dataset.py
python app\ml\train_random_forest.py
python app\ml\train_xgboost.py
python app\ml\train_lstm.py
```

Each script prints what it is doing. Look for `✓ complete` or `✓ Training complete` at the end of each one.

**Expected output files after all five scripts:**
```
backend\data\processed\combined_dataset.csv      ← raw merged data
backend\data\processed\cleaned_dataset.csv       ← fully cleaned
backend\data\processed\labeled_dataset.csv       ← with degradation labels
backend\models\random_forest_model.pkl
backend\models\xgboost_model.pkl
backend\models\lstm_model.keras
backend\models\lstm_scaler.pkl
backend\models\comparison_report.json
```

> **Training time:** With 19 million rows, Random Forest and XGBoost take 10–30 minutes each.
> LSTM training uses a memory-efficient generator and will take longer — progress is shown per epoch.

---

## Step 7 — Start the Backend Server

```
uvicorn main:app --reload
```

Expected startup output:
```
[Startup] O-RAN KPI Prediction xApp API starting...
[Model Loader] Loading Random Forest...
[Model Loader] Loading XGBoost...
[Model Loader] Loading LSTM...
[StreamState] Loaded 19053569 rows...
[Startup] Server ready.
INFO: Uvicorn running on http://127.0.0.1:8000
```

Verify: open `http://localhost:8000/api/health` in your browser.

---

## Step 8 — Set Up the Frontend

Open a **second** Command Prompt (keep the backend running in the first one).

```
cd path\to\fyp-oran-kpi-prediction\frontend
npm install
npm run dev
```

Expected output:
```
ready - started server on 0.0.0.0:3000
```

---

## Step 9 — Open the Dashboard

- **http://localhost:3000** — manual prediction dashboard
- **http://localhost:3000/live** — live monitoring
- **http://localhost:8000/docs** — interactive API documentation

The header shows `Model Ready` in green when everything is working.

---

## Common Problems and Fixes

### "python is not recognized"
Python not added to PATH. Reinstall and tick "Add Python to PATH" on the first screen.

### venv\Scripts\activate fails in PowerShell
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### pip install fails for TensorFlow
You have Python 3.13 installed. TensorFlow requires 3.10–3.12. Reinstall Python 3.11.

### Training crashes with "Unable to allocate X GiB"
This is the LSTM memory error. Make sure you are running the updated `train_lstm.py` (the memory-efficient generator version). Check line 1 of the file — it should say "Memory-efficient generator-based training".

### Training crashes with "Input X contains infinity"
The `prb_grant_ratio` column has infinity values. Make sure you are using the updated `clean_dataset.py` that includes infinity clamping. Re-run `clean_dataset.py` and `label_dataset.py` before retraining.

### Wrong features in training (IMSI, RNTI showing up)
You are using the old `clean_dataset.py`. The updated version has a `COLUMNS_TO_ALWAYS_DROP` list that removes identity columns even when they vary across multiple files. Replace your file and re-run the pipeline from `clean_dataset.py`.

### "npm is not recognized"
Node.js not installed. Download from https://nodejs.org.

### Port 8000 already in use
```
uvicorn main:app --reload --port 8001
```
Then update `frontend/next.config.js` to use port 8001.

### Port 3000 already in use
```
npm run dev -- --port 3001
```

---

## Returning After a Break

You do not need to retrain models. Just activate venv and start the servers:

**Terminal 1 (backend):**
```
cd backend
venv\Scripts\activate
uvicorn main:app --reload
```

**Terminal 2 (frontend):**
```
cd frontend
npm run dev
```
