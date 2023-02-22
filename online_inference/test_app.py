import pytest
from fastapi.testclient import TestClient

from app import app
from src.constants import IS_FRAUD_NAME, NON_FRAUD_NAME
from src.data_model import PredictionRow

test_client = TestClient(app)
TEST_ROW = PredictionRow(
    type="TRANSFER",
    amount=200,
    oldbalanceOrg=200,
    newbalanceOrig=0,
    oldbalanceDest=0,
    newbalanceDest=200
)


def test_predict():
    result = test_client.post("/predict", json=TEST_ROW.dict())
    result_json = result.json()
    assert "prediction" in result_json
    assert result_json.get("prediction") in [IS_FRAUD_NAME, NON_FRAUD_NAME]


def test_validation_type_error():
    with pytest.raises(ValueError):
        PredictionRow(
            type="random_string",
            amount=200,
            oldbalanceOrg=200,
            newbalanceOrig=0,
            oldbalanceDest=0,
            newbalanceDest=200
        )
