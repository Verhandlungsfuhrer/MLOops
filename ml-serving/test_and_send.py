import json
import requests
import numpy as np

# ������� �������� ������ ���������� �����: [1, 1, 28, 28]
test_data = np.ones((1, 1, 28, 28)) * 0.1

data = {
    "input": test_data.tolist()
}

print("���������� ������ � �������...")
try:
    response = requests.post(
        "http://localhost:8080/predict",
        json=data,
        timeout=30
    )
    print(f"������ ���: {response.status_code}")
    if response.status_code == 200:
        print("�������� �����:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"������: {response.text}")
except Exception as e:
    print(f"������ ��� �������� �������: {e}")
