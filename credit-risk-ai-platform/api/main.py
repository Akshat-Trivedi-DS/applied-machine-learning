"""FastAPI service for the credit risk model.

The model is loaded once at startup (via lifespan) and reused across
requests instead of being re-loaded from disk on every call.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.model_service import ModelService
from api.schemas import CreditApplication, HealthResponse, PredictionResponse

service_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    service_state["model_service"] = ModelService()
    yield
    service_state.clear()


app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts credit card default probability and risk category.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_service() -> ModelService:
    service = service_state.get("model_service")
    if service is None:
        raise HTTPException(status_code=503, detail="Model service not ready")
    return service


@app.get("/health", response_model=HealthResponse)
async def health():
    service = service_state.get("model_service")
    return HealthResponse(
        status="ok" if service else "starting",
        model_loaded=service is not None,
        model_name=service.model_name if service else "",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
    service = get_service()
    try:
        result = service.predict(application)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictionResponse(**result)
