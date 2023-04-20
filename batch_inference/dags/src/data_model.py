from pydantic import BaseModel, validator


class PredictionRow(BaseModel):
    type: str  # noqa: A003, VNE003
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    @validator("type")
    def validate_type(cls, v: str) -> str:
        if v in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]:
            return v
        raise ValueError("type must be in 'PAYMENT' 'TRANSFER' 'CASH_OUT' 'DEBIT' 'CASH_IN'")

    @validator("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
    def validate_moneys(cls, v: float) -> float:
        if v >= 0:
            return v
        raise ValueError("money must be greater or equal to 0")
