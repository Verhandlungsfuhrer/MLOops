from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor


default_args = {
    "owner": "imd",
    "email": ["imdxdd@gmail.com"],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
        "bash",
        default_args=default_args,
        schedule_interval="0 0 * * 5",
        start_date=days_ago(1),
) as dag:
    make_dir = BashOperator(
        task_id="mkdir",
        bash_command="mkdir -p ../{{ds}}"
    )
    get_csv = BashOperator(
        task_id="get_data",
        bash_command="wget https://data.cdc.gov/api/views/yni7-er2q/rows.csv -O ../{{ds}}/data.csv"
    )
    wait_file = FileSensor(task_id="wait", filepath="../{{ds}}/data.csv")
    get_wc = BashOperator(
        task_id="calculate_word_count",
        bash_command="wc ../{{ds}}/data.csv >> ../{{ds}}/wc.txt"
    )
    make_dir >> get_csv >> wait_file >> get_wc
