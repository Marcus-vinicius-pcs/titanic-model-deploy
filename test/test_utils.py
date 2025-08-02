import pytest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import ModelManager

def test_load_model_file_not_found(tmp_path):
    manager = ModelManager(model_path=str(tmp_path / "not_exist.pkl"))
    with pytest.raises(FileNotFoundError):
        manager.load_model()

def test_load_model_success(tmp_path):
    # Usa um objeto simples que pode ser picklado
    dummy_model = {"foo": "bar"}
    model_path = tmp_path / "dummy.pkl"
    import pickle
    with open(model_path, "wb") as f:
        pickle.dump(dummy_model, f)
    manager = ModelManager(model_path=str(model_path))
    manager.load_model()
    assert manager.model == dummy_model

def test_load_model_from_bytes_success():
    # Usa um objeto simples que pode ser picklado
    dummy_model = [1, 2, 3]
    import pickle
    model_bytes = pickle.dumps(dummy_model)
    manager = ModelManager()
    manager.load_model_from_bytes(model_bytes)
    assert manager.model == dummy_model

def test_is_model_loaded_true_false():
    manager = ModelManager()
    assert not manager.is_model_loaded()
    manager.model = MagicMock()
    assert manager.is_model_loaded()

def test_predict_returns_expected_dict():
    manager = ModelManager()
    # Mocka apenas os métodos necessários do modelo
    class DummyModel:
        def predict(self, df):
            return np.array([1])
        def predict_proba(self, df):
            return np.array([[0.3, 0.7]])
    manager.model = DummyModel()

    data = {'PassengerId': 123, 'feature1': 2.0, 'feature2': 3.0}
    result = manager.predict(data)
    assert result['passenger_id'] == 123
    assert result['prediction'] == 1
    assert isinstance(result['probability'], np.ndarray)
    assert result['probability'].shape == (1, 2)