import json
import numpy as np

# ������� �������� ������ ���������� �����: [1, 1, 28, 28]
test_data = np.ones((1, 1, 28, 28)) * 0.1

data = {
    "input": test_data.tolist()
}

with open("correct_input.json", "w") as f:
    json.dump(data, f)

print("���� correct_input.json ������ � ���������� ������ [1, 1, 28, 28]")
