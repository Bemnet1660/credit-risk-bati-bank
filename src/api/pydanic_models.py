from pydantic import BaseModel
from typing import Dict, Any

class CustomerFeatures(BaseModel):
    features: Dict[str, Any]

class RiskPredictionResponse(BaseModel):
    risk_probability: float
    credit_score: int
