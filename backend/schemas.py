
"""Pydantic API schemas that exclude reference labels."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Backend health status."""

    status: str


class PredictionRequest(BaseModel):
    """Request for route reconstruction."""

    trip_id: str


class RouteMetrics(BaseModel):
    """Server-computed evaluation metrics."""

    accuracy: float | None = None
    route_precision: float
    route_recall: float
    route_f1: float
    jaccard_similarity: float
    exact_route_match: bool


class PredictionResponse(BaseModel):
    map_html: str | None = None
    """Prediction results without ground-truth labels."""

    trip_id: str
    status: str

    predicted_route: list[int] = Field(default_factory=list)

    metrics: RouteMetrics | None = None

    execution_time: float | None = None