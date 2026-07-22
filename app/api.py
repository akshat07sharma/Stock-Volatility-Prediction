import os
import sys

from fastapi import FastAPI
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(__file__))

from predictor import predict_volatility


app = FastAPI(
    title="Stock Volatility Forecaster API",
    description="Predicts next-day stock volatility using a trained LightGBM model.",
    version="2.0.0"
)


@app.get("/")
def home():

    return {
        "status": "Running",
        "project": "Stock Volatility Forecaster",
        "model": "LightGBM"
    }


@app.get("/predict")
def predict():

    result = predict_volatility()

    if "error" in result:
        raise HTTPException(
            status_code=400,
            detail=result["error"]
        )

    return result


@app.get("/health")
def health():

    return {
        "status": "Healthy"
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
