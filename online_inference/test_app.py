import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from PIL import UnidentifiedImageError
import numpy as np

from app import app

test_client = TestClient(app)


def test_root():
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.text == "server is live"


def test_infer():
    app.triton_client = MagicMock()
    infer = app.triton_client.infer
    infer_result = MagicMock()
    infer.return_value = infer_result
    numpy_result = MagicMock()
    infer_result.as_numpy = numpy_result
    numpy_result.return_value = np.arange(1000)
    with open("labels.txt") as f:
        app.labels = [line.strip() for line in f.readlines()]
    with open("test.jpg", "rb") as fio:
        responce = test_client.post("/infer", files={"image": fio})
    assert responce.status_code == 200
    assert responce.text == app.labels[-1]


def test_exception():
    with pytest.raises(UnidentifiedImageError):
        app.triton_client = MagicMock()
        infer = app.triton_client.infer
        infer_result = MagicMock()
        infer.return_value = infer_result
        numpy_result = MagicMock()
        infer_result.as_numpy = numpy_result
        numpy_result.return_value = np.arange(1000)
        with open("labels.txt") as f:
            app.labels = [line.strip() for line in f.readlines()]
        with open("labels.txt", "rb") as fio:
            test_client.post("/infer", files={"image": fio})
