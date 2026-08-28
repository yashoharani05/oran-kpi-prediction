# =============================================================================
# app/utils/model_loader.py
# Loads Random Forest, XGBoost, and LSTM models at startup.
#
# METHODOLOGY NOTE: these paths point at the "_forecast_5s" model artefacts
# produced by the corrected training scripts — each predicts degradation
# ~5 seconds AHEAD of the input KPIs, not the same-instant state. See
# docs/FORECASTING_METHODOLOGY_UPDATE.md.
# =============================================================================

import os
import joblib

RF_MODEL_PATH   = os.path.join("models", "random_forest_forecast_5s.pkl")
XGB_MODEL_PATH  = os.path.join("models", "xgboost_forecast_5s.pkl")
LSTM_MODEL_PATH = os.path.join("models", "lstm_forecast_5s.keras")
LSTM_SCALER_PATH= os.path.join("models", "lstm_scaler_forecast_5s.pkl")


def load_model(path, name):
    if not os.path.exists(path):
        print(f"[Model Loader] Warning: {name} not found at {path}")
        return None
    print(f"[Model Loader] Loading {name}...")
    model = joblib.load(path)
    print(f"[Model Loader] {name} loaded (features: {model.n_features_in_})")
    return model


def load_lstm():
    """
    LSTM needs two artefacts:
      - the Keras model file (.keras)
      - the MinMaxScaler (.pkl) used during training
    Both must be loaded together; the scaler is needed to preprocess
    new inputs before passing them to the model.
    """
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"[Model Loader] Warning: LSTM model not found at {LSTM_MODEL_PATH}")
        return None, None
    if not os.path.exists(LSTM_SCALER_PATH):
        print(f"[Model Loader] Warning: LSTM scaler not found at {LSTM_SCALER_PATH}")
        return None, None

    # Import TF only when actually needed (avoids slow import if not used)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.models import load_model as keras_load

    print("[Model Loader] Loading LSTM model...")
    lstm_model  = keras_load(LSTM_MODEL_PATH)
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    print(f"[Model Loader] LSTM loaded (params: {lstm_model.count_params():,})")
    return lstm_model, lstm_scaler


def load_all_models():
    """
    Load all three models and return them in a dict.
    LSTM also returns its scaler under a separate key.
    """
    lstm_model, lstm_scaler = load_lstm()
    return {
        "random_forest": load_model(RF_MODEL_PATH,  "Random Forest"),
        "xgboost":       load_model(XGB_MODEL_PATH, "XGBoost"),
        "lstm":          lstm_model,
        "lstm_scaler":   lstm_scaler,
    }
