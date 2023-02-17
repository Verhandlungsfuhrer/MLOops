import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import MODEL_NAME

COLUMNS = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
REAL_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CAT_COLS = ["type"]
TARGET_COL = "isFraud"

TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv("data/onlinefraud.csv")
df = df.loc[:, COLUMNS]
print(df["type"].unique())
y = df[TARGET_COL]
print("splitting")
train_X, test_X, train_y, test_y = train_test_split(df.drop(columns=TARGET_COL), df[TARGET_COL],
                                                    test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True,
                                                    stratify=df[TARGET_COL])

transforms = ColumnTransformer([
    ("normalizer", StandardScaler(), REAL_COLS),
    ("encoding", OneHotEncoder(), CAT_COLS)
])
pipeline = Pipeline([
    ("transforms", transforms),
    ("model", RandomForestClassifier(n_jobs=-1))
])

print("training")
pipeline.fit(train_X, train_y)
print("prediction")
pred = pipeline.predict(test_X)
print(f1_score(test_y, pred))


with open(MODEL_NAME, "wb") as f:
    pickle.dump(pipeline, f)
