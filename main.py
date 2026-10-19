from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib


app = FastAPI()


# Load trained model
model = joblib.load(
    "bank_loan_model.pkl"
)


# Define input structure
class CustomerData(BaseModel):
    AGE: float
    EMPLOY: float
    ADDRESS: float
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float


@app.get("/")
def home():
    return {
        "message": "Bank Loan Default Prediction API",
        "endpoint": "/predict"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([{
        "AGE": data.AGE,
        "EMPLOY": data.EMPLOY,
        "ADDRESS": data.ADDRESS,
        "DEBTINC": data.DEBTINC,
        "CREDDEBT": data.CREDDEBT,
        "OTHDEBT": data.OTHDEBT
    }])

    probability = model.predict_proba(
        input_data
    )[0][1]

    return {
        "probability_of_default":
        round(float(probability), 4)
    }