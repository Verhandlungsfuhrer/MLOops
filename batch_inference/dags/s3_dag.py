from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonVirtualenvOperator
from airflow.utils.dates import days_ago

s3_address = Variable.get("s3_server")
s3_bucket = Variable.get("data_bucket")


def upload_data(data_name, s3_address, s3_bucket):
    import minio

    client = minio.Minio(
        s3_address, secure=False, access_key="miniouser", secret_key="minioser"
    )
    content = client.get_object(s3_bucket, data_name)
    with open(data_name, "wb") as fio:
        fio.write(content.data)


default_args = {
    "owner": "imd",
    "email": ["imdxdd@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
    "s3_uploader",
    default_args=default_args,
    schedule_interval="0 12 * * 1-5",
    start_date=days_ago(1),
):
    PythonVirtualenvOperator(
        task_id="upload_data",
        python_callable=upload_data,
        op_args=["{{ds}}", s3_address, s3_bucket],
        requirements=["minio"],
    )
