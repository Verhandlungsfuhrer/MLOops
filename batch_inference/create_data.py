import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, validator

possible_type_values = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
RAW_DATA_FOLDER = Path("raw_data")


class PredictionRow(BaseModel):
    type: str  # noqa: A003, VNE003
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    @validator("type")
    def validate_type(cls, v: str) -> str:
        if v in possible_type_values:
            return v
        raise ValueError("type must be in 'PAYMENT' 'TRANSFER' 'CASH_OUT' 'DEBIT' 'CASH_IN'")

    @validator("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
    def validate_moneys(cls, v: float) -> float:
        if v >= 0:
            return v
        raise ValueError("money must be greater or equal to 0")


result_data = []
for _ in range(1000):
    choosen_type = random.choice(possible_type_values)
    choosen_amount = random.randint(0, 1_000_000)
    oldbalanceOrg = random.randint(0, 1_000_000)
    newbalanceOrig = random.randint(0, 1_000_000)
    oldbalanceDest = random.randint(0, 1_000_000)
    newbalanceDest = random.randint(0, 1_000_000)
    result_data.append(PredictionRow(type=choosen_type, amount=choosen_amount, oldbalanceOrg=oldbalanceOrg,
                                     newbalanceOrig=newbalanceOrig, oldbalanceDest=oldbalanceDest,
                                     newbalanceDest=newbalanceDest).dict())
result_df = pd.DataFrame(result_data)
current_date = datetime.now().strftime("%Y-%m-%d")
result_df.to_csv(RAW_DATA_FOLDER / f"{current_date}.csv", index=False)
