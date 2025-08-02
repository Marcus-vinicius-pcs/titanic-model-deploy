import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main import app

client = TestClient(app)

@pytest.fixture
def passenger_example():
    return {
        "PassengerId": 1,
        "Pclass": 3,
        "Name": "John Doe",
        "Sex": "male",
        "Age": 22,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": None,
        "Embarked": "S"
    }

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_history_endpoint():
    response = client.get("/history")
    assert response.status_code == 200
    assert "total_predictions" in response.json()
    assert "history" in response.json()