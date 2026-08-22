from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    plant: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_index: int = Field(..., ge=0)
