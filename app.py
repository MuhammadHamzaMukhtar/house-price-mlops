# app.py - house-price API that answers questions using champion.pkl,
# keeps a diary of every prediction, and exposes metrics for Prometheus.
#
# Run:  python -m uvicorn app:app --reload --port 8000   then open http://127.0.0.1:8000/docs

import csv
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

app = FastAPI(title="House Price API", version="1.0.0")

# Load the model ONCE when the API starts (not on every request).
# MODEL_PATH / LOG_FILE can be changed with environment variables (handy for tests).
MODEL_PATH = os.getenv("MODEL_PATH", "champion.pkl")
model = joblib.load(MODEL_PATH)

# The prediction diary: one row per request (used later for drift checks)
LOG_FILE = os.getenv("LOG_FILE", "predictions.csv")

# Metrics: numbers Prometheus collects every few seconds
PREDICTIONS = Counter("predictions_total", "How many predictions the API has made")
LATENCY = Histogram("prediction_seconds", "How long one prediction takes")
SIZES = Histogram("input_size_sqft", "House sizes users ask about",
                  buckets=[500, 1000, 1500, 2000, 3000, 4000, 6000, 10000])
PRICES = Histogram("predicted_price_lakh", "Prices the model predicts",
                   buckets=[25, 50, 75, 100, 150, 200, 300, 500])


# The shape of a request. FastAPI rejects anything that doesn't match (HTTP 422).
class House(BaseModel):
    rooms: int = Field(gt=0, description="Number of rooms", examples=[3])
    size_sqft: float = Field(gt=0, description="Size in square feet", examples=[1350])


class Prediction(BaseModel):
    price_lakh: float


def log_prediction(house: House, price: float) -> None:
    is_new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["time", "rooms", "size_sqft", "predicted_price_lakh"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), house.rooms, house.size_sqft, price])


# Door 1: "are you alive?" (used by Kubernetes readiness/liveness probes)
@app.get("/health")
def health():
    return {"status": "ok"}


# Door 2: the real job
@app.post("/predict", response_model=Prediction)
def predict(house: House):
    with LATENCY.time():
        data = pd.DataFrame([{"rooms": house.rooms, "size_sqft": house.size_sqft}])
        price = round(float(model.predict(data)[0]), 1)

    PREDICTIONS.inc()
    SIZES.observe(house.size_sqft)
    PRICES.observe(price)
    log_prediction(house, price)
    return {"price_lakh": price}


# Door 3: metrics for Prometheus
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
