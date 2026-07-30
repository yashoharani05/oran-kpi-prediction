# =============================================================================
# app/schemas.py
#
# PURPOSE:
#   Define the exact shape of JSON data that flows IN and OUT of the API.
#
#   Pydantic schemas do three things automatically:
#     1. Validate incoming JSON — if a required field is missing or has the
#        wrong type, FastAPI returns a clear 422 error instead of crashing.
#     2. Generate API documentation — FastAPI reads these classes and builds
#        the interactive Swagger UI at /docs automatically.
#     3. Provide type hints — your editor knows exactly what fields exist.
#
# PYDANTIC V2 NOTES:
#   - model_config = {"protected_namespaces": ()} suppresses the UserWarning
#     about fields whose names start with "model_" (model_used, model_name, etc.)
#   - Examples are placed in json_schema_extra, not example= in Field()
#     (example= was deprecated in Pydantic v2 and removed in v3)
# =============================================================================

from pydantic import BaseModel, Field


# =============================================================================
# REQUEST SCHEMA — what the client sends to POST /api/predict
# =============================================================================

class KpiInput(BaseModel):
    """
    The 18 KPI features the ML model expects.
    Every field maps directly to a column in the labeled_dataset.csv.
    Field(...) means the field is REQUIRED.
    """

    # --- Downlink (base station → phone) features ---
    dl_mcs: float = Field(
        ..., description="Downlink Modulation and Coding Scheme (0–28). Higher = better channel."
    )
    dl_n_samples: int = Field(
        ..., description="Number of downlink transmission samples in this window."
    )
    dl_buffer_bytes: int = Field(
        ..., description="Bytes of downlink data waiting in the queue (backlog)."
    )
    tx_brate_downlink_mbps: float = Field(
        ..., description="Actual downlink throughput in Megabits per second."
    )
    tx_pkts_downlink: int = Field(
        ..., description="Number of downlink packets transmitted in this window."
    )
    dl_cqi: float = Field(
        ..., description="Channel Quality Indicator reported by the UE (0–15). Lower = worse."
    )

    # --- Uplink (phone → base station) features ---
    ul_mcs: float = Field(
        ..., description="Uplink Modulation and Coding Scheme (0–28)."
    )
    ul_n_samples: int = Field(
        ..., description="Number of uplink transmission samples in this window."
    )
    ul_buffer_bytes: int = Field(
        ..., description="Bytes of uplink data waiting in the queue."
    )
    rx_brate_uplink_mbps: float = Field(
        ..., description="Actual uplink throughput in Megabits per second."
    )
    rx_pkts_uplink: int = Field(
        ..., description="Number of uplink packets received by the base station."
    )
    rx_errors_uplink_pct: float = Field(
        ..., description="Uplink packet error rate as a percentage (0–100). Higher = worse."
    )

    # --- Signal quality and resource features ---
    ul_sinr: float = Field(
        ..., description="Signal-to-Interference-plus-Noise Ratio in dB. 0 when UE is idle."
    )
    phr: int = Field(
        ..., description="Power Headroom Report — remaining transmit power the UE has."
    )
    sum_requested_prbs: int = Field(
        ..., description="Physical Resource Blocks the UE requested this window."
    )
    sum_granted_prbs: int = Field(
        ..., description="Physical Resource Blocks the scheduler actually granted."
    )
    ul_turbo_iters: float = Field(
        ..., description="Average turbo decoder iterations per uplink packet. 0 when UE is idle."
    )
    prb_grant_ratio: float = Field(
        ..., description="Derived: sum_granted_prbs / (sum_requested_prbs + 1). Congestion indicator."
    )

    model_config = {
        # Provide a working example payload in Swagger UI
        "json_schema_extra": {
            "example": {
                "dl_mcs": 9.6,
                "dl_n_samples": 147,
                "dl_buffer_bytes": 0,
                "tx_brate_downlink_mbps": 0.115,
                "tx_pkts_downlink": 42,
                "dl_cqi": 7.0,
                "ul_mcs": 0.0,
                "ul_n_samples": 0,
                "ul_buffer_bytes": 0,
                "rx_brate_uplink_mbps": 0.0,
                "rx_pkts_uplink": 0,
                "rx_errors_uplink_pct": 0.0,
                "ul_sinr": 0.0,
                "phr": 0,
                "sum_requested_prbs": 790,
                "sum_granted_prbs": 174,
                "ul_turbo_iters": 0.0,
                "prb_grant_ratio": 0.22,
            }
        }
    }


# =============================================================================
# RESPONSE SCHEMA — what the API sends back after a prediction
# =============================================================================

class PredictionResponse(BaseModel):
    """
    The prediction result returned by POST /api/predict.

    risk_label      — "Normal" or "Degraded"
    risk_code       — 0 (Normal) or 1 (Degraded)
    probability     — model confidence that network IS degraded (0.0–1.0)
    recommendation  — plain-English action advice
    model_used      — name of the model that made the prediction
    """

    # model_config suppresses the Pydantic UserWarning about fields
    # whose names begin with "model_" (model_used in this case).
    model_config = {"protected_namespaces": ()}

    risk_label: str = Field(
        description="Human-readable prediction: 'Normal' or 'Degraded'"
    )
    risk_code: int = Field(
        description="Machine-readable prediction: 0 = Normal, 1 = Degraded"
    )
    probability: float = Field(
        description="Model confidence that the network IS degraded (0.0–1.0)"
    )
    recommendation: str = Field(
        description="Plain-English action recommendation based on the prediction"
    )
    model_used: str = Field(
        description="The ML model that produced this prediction"
    )


# =============================================================================
# HEALTH CHECK RESPONSE SCHEMA — returned by GET /api/health
# =============================================================================

class HealthResponse(BaseModel):
    """
    Response returned by GET /api/health.
    Confirms the API is running and reports which models are loaded.
    """

    # Suppresses UserWarning for model_loaded and model_name fields
    model_config = {"protected_namespaces": ()}

    status: str = Field(
        description="'ok' if the API is healthy, 'error' if something is wrong."
    )
    model_loaded: bool = Field(
        description="True if at least one ML model was loaded successfully."
    )
    model_name: str = Field(
        description="Names of the loaded models, e.g. 'RF + XGB + LSTM'."
    )
    message: str = Field(
        description="Human-readable status message showing per-model load status."
    )
