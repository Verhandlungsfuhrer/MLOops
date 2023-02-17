import pandas as pd
import pickle
from .constants import NON_FRAUD_NAME, IS_FRAUD_LABEL, IS_FRAUD_NAME


class PredictionModel:

    def __init__(self, path_to_file):
        with open(path_to_file, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, prediction_row):
        series = pd.Series(prediction_row)
        df = pd.DataFrame(data=[series])
        prediction = self.model.predict(df)
        return IS_FRAUD_NAME if prediction[0] == IS_FRAUD_LABEL else NON_FRAUD_NAME
