FROM python:3.8.15

WORKDIR /app
COPY model.pkl /app/model.pkl
COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY app.py /app/app.py

RUN pip3 install -r requirements.txt

EXPOSE 8090
CMD ["uvicorn", "app:app", "--port", "8090", "--host", "0.0.0.0"]
