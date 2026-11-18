"""FastAPI route definitions."""
from fastapi import APIRouter
from .schemas import HealthResponse, PredictionRequest, PredictionResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service liveness."""
    return HealthResponse(status="skeleton")

@router.get("/trips", response_model=list[str])
def trips() -> list[str]:
    """List approved trips. TODO: implement without ground truth."""
    raise NotImplementedError

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict a route. TODO: integrate approved model pipeline."""
    raise NotImplementedError
