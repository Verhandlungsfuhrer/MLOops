import pickle

import pandas as pd
from sklearn.pipeline import Pipeline

from .constants import IS_FRAUD_LABEL, IS_FRAUD_NAME, NON_FRAUD_NAME
from .data_model import PredictionRow


class PredictionModel:
    def __init__(self, path_to_file: str) -> None:
        with open(path_to_file, "rb") as f:
            self.model: Pipeline = pickle.load(f)

    def predict(self, data_model: PredictionRow) -> str:
        prediction_row = data_model.dict()
        series = pd.Series(prediction_row)
        df = pd.DataFrame(data=[series])
        prediction = self.model.predict(df)
        return IS_FRAUD_NAME if prediction[0] == IS_FRAUD_LABEL else NON_FRAUD_NAME
