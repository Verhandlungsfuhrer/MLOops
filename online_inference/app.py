import sys
import logging

from fastapi import FastAPI
from pydantic import BaseModel, validator

from src.constants import MODEL_NAME
from src.model import PredictionModel

app = FastAPI()
prediction_model = PredictionModel(MODEL_NAME)
logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)


class PredictionRow(BaseModel):
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    @validator("type")
    def validate_type(cls, v):
        if v in ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']:
            return v
        logger.error("type must be in 'PAYMENT' 'TRANSFER' 'CASH_OUT' 'DEBIT' 'CASH_IN'")
        raise ValueError("type must be in 'PAYMENT' 'TRANSFER' 'CASH_OUT' 'DEBIT' 'CASH_IN'")

    @validator("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
    def validate_moneys(cls, v):
        if v >= 0:
            return v
        logger.error("money must be greater or equal to 0")
        raise ValueError(f"money must be greater or equal to 0")


@app.post("/predict")
def predict_function(data_model: PredictionRow):
    logger.info("Get row for prediction")
    predict = prediction_model.predict(data_model.dict())
    logger.info(f"{data_model} is {predict}")
    return {"prediction": predict}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8090)
