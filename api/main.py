from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

app = FastAPI(title="Favorita Time Series Forecast API")

# Define the expected JSON payload
class SalesHistory(BaseModel):
    sales: List[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
def get_forecast(history: SalesHistory):
    # We need at least 2 weeks of data to capture weekly seasonality
    if len(history.sales) < 14:
        raise HTTPException(status_code=400, detail="Need at least 14 days of historical sales to forecast.")
    
    try:
        # Train the ETS model on the provided history
        model = ExponentialSmoothing(
            history.sales, 
            trend="add", 
            seasonal="add", 
            seasonal_periods=7
        )
        fitted = model.fit()
        
        # Forecast the next 7 days
        forecast = fitted.forecast(7)
        
        return {
            "forecast_next_7_days": [round(float(x), 2) for x in forecast],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
