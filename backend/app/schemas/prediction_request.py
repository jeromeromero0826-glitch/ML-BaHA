from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    duration: float = Field(..., gt=0, description="Rainfall duration in hours")
    depth: float = Field(..., ge=0, description="Rainfall depth in mm")
    antecedent: float = Field(..., ge=0, description="Antecedent rainfall in mm")