FROM python:3.8.15

WORKDIR /app

COPY src /app/src
COPY requirements.txt /app/requirements.txt
COPY demo_producer.py /app/demo_producer.py
RUN pip3 install -r requirements.txt

CMD ["python", "demo_producer.py"]
