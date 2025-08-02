import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import CategoricalFeatureEngineer, TitanicPreprocessor
from unittest.mock import MagicMock, patch

def test_categorical_feature_engineer_transform_basic():
    df = pd.DataFrame({
        'Name': ['Smith, Mr. John', 'Brown, Miss. Anna'],
        'Sex': ['male', 'female'],
        'Cabin': [np.nan, 'C85'],
        'Ticket': ['A/5 21171', 'PC 17599'],
        'Age': [22, np.nan],
        'SibSp': [1, 0],
        'Parch': [0, 0],
        'Fare': [7.25, 71.2833],
        'Embarked': ['S', None]
    })
    transformer = CategoricalFeatureEngineer()
    result = transformer.transform(df)
    # Test if new columns exist
    assert 'Title' in result.columns
    assert 'Title_Mapped' in result.columns
    assert 'Deck' in result.columns
    assert 'Cabin_Count' in result.columns
    assert 'Ticket_Prefix' in result.columns
    assert 'Age_Group' in result.columns
    assert 'Fare_Group' in result.columns
    assert 'Family_Size' in result.columns
    assert 'Alone' in result.columns
    assert 'Embarked' in result.columns
    assert 'Ticket_Group_Size' in result.columns
    # Test if Sex is binarized
    assert set(result['Sex'].unique()).issubset({0, 1})

def test_categorical_feature_engineer_alone_family_size():
    df = pd.DataFrame({
        'Name': ['Smith, Mr. John'],
        'Sex': ['male'],
        'Cabin': [np.nan],
        'Ticket': ['A/5 21171'],
        'Age': [22],
        'SibSp': [0],
        'Parch': [0],
        'Fare': [7.25],
        'Embarked': ['S']
    })
    transformer = CategoricalFeatureEngineer()
    result = transformer.transform(df)
    assert result['Family_Size'].iloc[0] == 0
    assert result['Alone'].iloc[0] == 1

def test_titanic_preprocessor_process(monkeypatch):
    # Mock pipeline and its transform method
    mock_pipeline = MagicMock()
    mock_pipeline.transform.return_value = np.array([[1, 2, 3]])
    # Mock feature names
    monkeypatch.setattr(TitanicPreprocessor, "_get_feature_names", lambda self: ["a", "b", "c"])
    # Patch pickle.load to return the mock pipeline
    with patch("builtins.open", create=True), patch("pickle.load", return_value=mock_pipeline):
        preprocessor = TitanicPreprocessor(pipeline_path="dummy_path.pkl")
        data = {'PassengerId': 1, 'Pclass': 1, 'Name': 'Smith, Mr. John', 'Sex': 'male', 'Age': 22,
                'SibSp': 0, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25, 'Cabin': None, 'Embarked': 'S'}
        result = preprocessor.process(data)
        assert result['PassengerId'] == 1
        assert result['a'] == 1
        assert result['b'] == 2
        assert result['c'] == 3