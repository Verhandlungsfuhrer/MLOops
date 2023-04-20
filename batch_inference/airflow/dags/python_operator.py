import random
from datetime import timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from src.data_model import PredictionRow


def generate_data(input_dir):

    input_dir_path = Path(input_dir)
    input_dir_path.mkdir(parents=True, exist_ok=True)
    possible_type_values = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    data = []
    for _ in range(1000):
        choosen_type = random.choice(possible_type_values)
        choosen_amount = random.randint(0, 1_000_000)
        oldbalance_org = random.randint(0, 1_000_000)
        newbalance_orig = random.randint(0, 1_000_000)
        oldbalance_dest = random.randint(0, 1_000_000)
        newbalance_dest = random.randint(0, 1_000_000)
        pred_row = PredictionRow(type=choosen_type, amount=choosen_amount, oldbalanceOrg=oldbalance_org,
                                 newbalanceOrig=newbalance_orig, oldbalanceDest=oldbalance_dest,
                                 newbalanceDest=newbalance_dest)
        data.append(pred_row.dict())
    df = pd.DataFrame(data)
    df.to_csv(input_dir_path / "generated_data.csv")


default_args = {
    "owner": "imd",
    "email": ["imdxdd@gmail.com"],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(default_args=default_args, schedule_interval="@daily", dag_id="python_operator", start_date=days_ago(1)):
    po = PythonOperator(python_callable=generate_data, op_args=["{{ds}}"], task_id="generate_data")
