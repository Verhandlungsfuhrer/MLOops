from datetime import timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "imd",
    "email": ["imdxdd@gmail.com"],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
    "dummy",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(7),
) as dag:
    operator = EmptyOperator(task_id="empty_task")
