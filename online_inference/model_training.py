import pickle

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.constants import MODEL_NAME

EXPERIMENT_NAME = "OnlineFraudRandomForest1"
mlflow.set_tracking_uri("http://0.0.0.0:5000/")
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(experiment_id=experiment_id):

    mlflow.set_registry_uri("s3://localhost:9002/mlflow")
    rf_params = {
        "n_estimators": 50,
        "TEST_SIZE": 0.2,
        "RANDOM_STATE": 42
    }

    FILEPATH = "data/onlinefraud.csv"
    mlflow.log_artifact(FILEPATH)
    TARGET_COL = "isFraud"

    COLUMNS = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "isFraud"]

    REAL_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    TEXT_COLS = []
    CAT_COLS = ["type"]

    print("read data")
    df = pd.read_csv(FILEPATH)
    df = df.loc[:, COLUMNS]
    y = df[TARGET_COL]

    print("start splitting")
    train_X, test_X, train_y, test_y = train_test_split(
        df.drop(columns=[TARGET_COL]), y, test_size=rf_params["TEST_SIZE"], random_state=rf_params["RANDOM_STATE"],
        shuffle=True, stratify=y
    )

    transforms = ColumnTransformer(
        [
            ("normalizer", Pipeline(
                [
                    ("impute", SimpleImputer(strategy="mean")),
                    ("norm", StandardScaler())
                ]
            ), REAL_COLS),
            ("encoder", Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value="")),
                    ("encode", OneHotEncoder())
                ]
            ), CAT_COLS),
            ("text", Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value="")),
                    ("vectorizer", TfidfVectorizer())
                ]
            ), TEXT_COLS)
        ]
    )

    mlflow.log_params(rf_params)
    pipeline = Pipeline([
        ("transforms", transforms),
        ("model", RandomForestClassifier(n_estimators=50, n_jobs=-1))
    ])

    pipeline.fit(train_X, train_y)
    pred = pipeline.predict(test_X)
    mlflow.log_metric("f1_score", f1_score(test_y, pred))

    with open(MODEL_NAME, "wb") as f:
        pickle.dump(pipeline, f)

    mlflow.sklearn.load_model("rf_fraud", MODEL_NAME)
