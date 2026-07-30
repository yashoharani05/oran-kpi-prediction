# =============================================================================
# app/utils/recommender.py
#
# PURPOSE:
#   Translate a model prediction and probability into a plain-English
#   recommendation that a network operator can act on.
#
#   Keeping this logic here (separate from the route handler) means:
#   - The API route stays short and readable
#   - Recommendations can be improved without touching routing logic
#   - The same function can be reused by other endpoints (e.g. XGBoost)
# =============================================================================


def get_recommendation(risk_code: int, probability: float) -> str:
    """
    Return a plain-English recommendation based on the prediction.

    We use both the binary label (risk_code) AND the probability to give
    more nuanced advice:
      - A "Degraded" prediction at 60% confidence is borderline
      - A "Degraded" prediction at 95% confidence is a clear emergency

    Args:
        risk_code   (int):   0 = Normal, 1 = Degraded
        probability (float): Confidence that the network IS degraded (0.0–1.0)

    Returns:
        A plain-English string describing the recommended action.
    """

    if risk_code == 0:
        # Network is predicted to be in a Normal state
        if probability < 0.20:
            # Very confident it is normal
            return (
                "Network is operating normally. "
                "All KPIs are within healthy ranges. No action required."
            )
        else:
            # Normal prediction but with some uncertainty — mild warning
            return (
                "Network is mostly normal but some KPIs show borderline values. "
                "Continue monitoring. Consider reviewing uplink error rates "
                "and channel quality in the next reporting window."
            )

    else:
        # Network is predicted to be in a Degraded state
        if probability >= 0.85:
            # High confidence — serious degradation
            return (
                "HIGH RISK: Network degradation detected with high confidence. "
                "Immediate investigation recommended. "
                "Check uplink error rates, CQI, and MCS values. "
                "Consider adjusting scheduler policy or power settings."
            )
        elif probability >= 0.60:
            # Moderate confidence — notable degradation
            return (
                "MODERATE RISK: Network degradation likely. "
                "Review KPI trends over the last 10–15 readings. "
                "Monitor uplink SINR and PRB grant ratio closely. "
                "Prepare to adjust radio parameters if degradation persists."
            )
        else:
            # Low-confidence degradation — borderline
            return (
                "LOW RISK: Marginal degradation detected. "
                "One or more KPIs are below normal thresholds. "
                "Increase monitoring frequency and watch for worsening trends. "
                "No immediate action required unless values deteriorate further."
            )
