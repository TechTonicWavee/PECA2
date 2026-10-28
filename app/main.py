
from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "california_housing_rf_model.pkl"
METADATA_PATH = ROOT / "california_housing_rf_metadata.json"
if not METADATA_PATH.exists() and (ROOT / "california_housing_rf_metadata").exists():
    METADATA_PATH = ROOT / "california_housing_rf_metadata"

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

app = FastAPI(
    title="California Housing Price Prediction API",
    description="Predict California housing values using Random Forest.",
    version="1.0.0",
)


class HousingInput(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float = Field(ge=0)
    population: float = Field(ge=0)
    households: float = Field(ge=0)
    median_income: float = Field(ge=0)
    ocean_proximity: str


@app.get("/")
def home():
    return {
        "message": "California Housing Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info():
    return metadata


@app.post("/predict")
def predict(data: HousingInput):
    try:
        # Preserve the feature names and order expected by training.
        features = pd.DataFrame([data.model_dump()])
        prediction = float(model.predict(features)[0])

        return {
            "predicted_median_house_value": round(prediction, 2),
            "currency": "USD",
            "model": "Random Forest Regressor",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the input schema and model artifacts.",
        ) from exc
