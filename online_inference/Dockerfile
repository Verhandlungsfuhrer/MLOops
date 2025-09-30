FROM python:3.10

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
COPY labels.txt /app/labels.txt
COPY app.py /app/app.py
