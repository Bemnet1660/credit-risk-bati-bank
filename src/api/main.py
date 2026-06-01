from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
import pandas as pd
import numpy as np
from src.api.pydantic_models import CustomerFeatures, RiskPredictionResponse

app = FastAPI()

# Load model from MLflow registry (adjust URI if needed)
model_uri = "models:/CreditRiskXGB/latest"
try:
    model = mlflow.pyfunc.load_model(model_uri)
except:
    # Fallback to a local model path if registry not set
    model = mlflow.pyfunc.load_model("mlruns/.../best_model")  # update after training

def prob_to_score(prob):
    return max(300, min(850, int(850 - prob * 550)))

@app.post("/predict", response_model=RiskPredictionResponse)
async def predict(customer: CustomerFeatures):
    try:
        input_df = pd.DataFrame([customer.features])
        proba = model.predict_proba(input_df)[0, 1]
        return RiskPredictionResponse(risk_probability=float(proba), credit_score=prob_to_score(proba))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
