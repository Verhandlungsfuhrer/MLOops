from datetime import timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "imd",
    "email": ["imdxdd@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
    "dummy",
    default_args=default_args,
    schedule_interval="0 * * * *",
    start_date=days_ago(1),
) as dag:
    operator = EmptyOperator(
        task_id="empty_task", email_on_failure=True, email_on_retry=True
    )
