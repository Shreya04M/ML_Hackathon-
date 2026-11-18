
"""FastAPI route definitions."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse
)

from .services import (
    list_available_trips,
    predict_trip,
    generate_trip_map
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.get("/trips", response_model=list[str])
def trips() -> list[str]:
    """Return approved evaluation trip IDs."""

    try:
        return list_available_trips()

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Trip dataset not available"
        )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Run route reconstruction."""

    try:
        return predict_trip(request.trip_id)

    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Invalid trip ID"
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Trip dataset not available"
        )

    except NotImplementedError:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline integration pending"
        )
    

@router.get("/map/{trip_id}", response_class=HTMLResponse)
def trip_map(trip_id: str):
    """Return an interactive GPS trajectory map."""

    try:
        return HTMLResponse(
            content=generate_trip_map(trip_id)
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Trip dataset not available"
        )