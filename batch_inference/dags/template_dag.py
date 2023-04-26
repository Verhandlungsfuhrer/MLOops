from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


def days_to_now(starting_date):
    return (datetime.now() - starting_date).days


with DAG(
    dag_id="demo_template",
    start_date=datetime(2021, 1, 1),
    schedule="@daily",
    user_defined_macros={
        "starting_date": datetime(2015, 5, 1),  # Macro can be a variable
        "days_to_now": days_to_now,  # Macro can also be a function
    },
) as dag:
    print_days = BashOperator(
        task_id="print_days",
        bash_command="echo Days since {{ starting_date }} is {{ days_to_now(starting_date) }}",
    )
