"""
FastAPI backend for the CA housing price model.

Run from the backend/ folder:
    uvicorn app.main:app --reload --port 8000
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importing ClusterSimilarity registers it at "app.ml_pipeline.ClusterSimilarity",
# which is the path joblib needs to find when unpickling the model below.
from app.ml_pipeline import ClusterSimilarity, RAW_FEATURE_COLUMNS  # noqa: F401

MODEL_PATH = Path(__file__).parent / "model" / "house_price_calculator_ml.pkl"

app = FastAPI(title="California Housing Price Predictor")

# Allow the local frontend (opened as a static file or via a dev server) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python train_model.py` from the backend/ folder first, "
            "or copy your existing .pkl there (see README)."
        )
    model = joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    longitude: float = Field(..., example=-122.27, description="District longitude")
    latitude: float = Field(..., example=37.80, description="District latitude")
    housing_median_age: float = Field(..., example=25, ge=0)
    total_rooms: float = Field(..., example=1200, gt=0)
    total_bedrooms: float = Field(..., example=250, gt=0)
    population: float = Field(..., example=900, gt=0)
    households: float = Field(..., example=300, gt=0)
    median_income: float = Field(
        ..., example=4.5, gt=0, description="Median income in tens of thousands, e.g. 4.5 = $45,000"
    )
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"] = Field(
        ..., example="NEAR BAY"
    )


class PredictionResponse(BaseModel):
    predicted_price: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = pd.DataFrame([features.model_dump()], columns=RAW_FEATURE_COLUMNS)

    try:
        prediction = model.predict(row)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(predicted_price=round(float(prediction), 2))
