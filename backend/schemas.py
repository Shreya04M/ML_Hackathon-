"""Pydantic API schemas that exclude reference labels."""
from pydantic import BaseModel

class HealthResponse(BaseModel):
    """Safe liveness response."""
    status: str

class PredictionRequest(BaseModel):
    """Placeholder prediction request."""
    trip_id: str

class PredictionResponse(BaseModel):
    """Placeholder prediction response without ground truth."""
    trip_id: str
    status: str
