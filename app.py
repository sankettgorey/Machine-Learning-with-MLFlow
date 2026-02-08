from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

import mlflow

import os
import mlflow.pyfunc


# Now load the model by stage
# import mlflow.pyfunc
# model = mlflow.pyfunc.load_model("models:/BikePredictionModel@staging")
# print(model)




from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="Bike Prediction API")

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Load model once at startup


# @app.on_event("startup")
# def load_model():
#     global model
#     model = mlflow.pyfunc.load_model(MODEL_URI)
#     print("Model loaded successfully")


@app.get("/reloadModel")
def reload_model():
    MODEL_URI = "models:/BikePredictionModel@prod"
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)
    return {"message": "Model Loaded Successfully"}


# ----------- INPUT SCHEMA -----------
class PredictionInput(BaseModel):
    season: float
    holiday: float
    workingday: float
    weather: float
    temp: float
    atemp: float
    humidity: float
    windspeed: float
    casual: float
    registered: float
    hour: float
    day_of_week: float
    month: float
    is_clear_weather: float
    is_rainy_weather: float
    is_holiday_workingday: float


# ----------- INFERENCE ENDPOINT -----------
@app.post("/predict1")
def predict1(data: PredictionInput):
    # Convert input to DataFrame (MLflow expects DataFrame)
    df = pd.DataFrame([data.model_dump()])

    print("created df")

    # Predict
    prediction = model.predict(df)

    return {
        "prediction": float(prediction[0])
    }