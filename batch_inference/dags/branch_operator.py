import random
from datetime import timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "imd",
    "email": ["imdxdd@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
FIRST_TASK_ID = "first_task"
ZERO_TASK_ID = "second_task"


def branch_operator():
    if random.randint(0, 1) == 1:
        return FIRST_TASK_ID
    else:
        return ZERO_TASK_ID


def first_task():
    print(1)


def zero_task():
    print(0)


with DAG(
    default_args=default_args,
    schedule_interval="@daily",
    dag_id="branching",
    start_date=days_ago(1),
):
    branch = BranchPythonOperator(python_callable=branch_operator, task_id="branch")
    first_op = PythonOperator(python_callable=first_task, task_id=FIRST_TASK_ID)
    second_op = PythonOperator(python_callable=zero_task, task_id=ZERO_TASK_ID)
    third_op = EmptyOperator(task_id="third")
    forth_op = EmptyOperator(task_id="forth")
    branch >> [first_op, second_op] >> third_op
    [first_op, second_op] >> forth_op
