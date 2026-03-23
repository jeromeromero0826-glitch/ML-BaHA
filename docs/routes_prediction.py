"""
routes_prediction.py — Optimized FastAPI router.

Improvements over original:
- /scenarios endpoint for frontend history panel
- Structured HTTPException error responses (no raw 500s)
- Request-level timing header for performance visibility
- /predict response unchanged for schema compatibility
"""

import time

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.prediction_request import PredictionRequest
from app.schemas.prediction_response import PredictionResponse
from app.services.prediction_service import run_prediction, get_scenario_history

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", summary="Health check")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Flood Hazard",
)
def predict_flood_hazard(payload: PredictionRequest, request: Request):
    assets = request.app.state.assets
    t0 = time.perf_counter()

    try:
        result = run_prediction(
            duration=payload.duration,
            depth=payload.depth,
            antecedent=payload.antecedent,
            assets=assets,
        )
    except ValueError as exc:
        # Input validation errors from rainfall_features.validate_inputs
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        # Unexpected errors — surface message without leaking traceback to client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    response_data = {
        "message":            "Prediction completed successfully.",
        "input_rainfall":     result["rainfall"],
        "summary":            result["summary"],
        "hazard_class_counts":result["summary"]["hazard_class_counts"],
        "outputs":            result["outputs"],
        "map_outputs":        result["map_outputs"],
    }

    response = JSONResponse(content=response_data)
    response.headers["X-Prediction-Time-Ms"] = str(elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Scenario history
# ---------------------------------------------------------------------------

@router.get("/scenarios", summary="List past prediction scenarios (newest first)")
def list_scenarios(limit: int = 20):
    """
    Returns up to `limit` past prediction scenarios from the in-memory registry.
    Each entry contains scenario_id, timestamp, input rainfall, summary stats,
    and output file URLs — enough for the frontend history panel.
    """
    history = get_scenario_history()[:limit]
    return {
        "count":     len(history),
        "scenarios": history,
    }


@router.get("/scenarios/{scenario_id}", summary="Get a single scenario by ID")
def get_scenario(scenario_id: str):
    history = get_scenario_history()
    for entry in history:
        if entry["scenario_id"] == scenario_id:
            return entry
    raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")
