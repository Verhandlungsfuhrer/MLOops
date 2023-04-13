import os
import time
import uuid
import random
from kafka import KafkaProducer
from src.data_model import PredictionRow

time.sleep(300)
bs = os.getenv("KAFKA_CLUSTERS_0_BOOTSTRAP_SERVERS")
possible_type_values = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
producer = KafkaProducer(bootstrap_servers=bs)

while True:
    choosen_type = random.choice(possible_type_values)
    choosen_amount = random.randint(0, 1_000_000)
    oldbalanceOrg = random.randint(0, 1_000_000)
    newbalanceOrig = random.randint(0, 1_000_000)
    oldbalanceDest = random.randint(0, 1_000_000)
    newbalanceDest = random.randint(0, 1_000_000)
    pred_row = PredictionRow(type=choosen_type, amount=choosen_amount, oldbalanceOrg=oldbalanceOrg,
                             newbalanceOrig=newbalanceOrig, oldbalanceDest=oldbalanceDest,
                             newbalanceDest=newbalanceDest)
    producer.send("mltopic", key=str(uuid.uuid4()).encode("utf8"), value=pred_row.json().encode("utf8"))
    time.sleep(5)
