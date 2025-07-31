import pickle
import logging
import numpy as np
import pandas as pd
from typing import Optional, Any
import os

logger = logging.getLogger('src.utils')

class ModelManager:
    """Manages model loading, prediction, and state"""
    
    def __init__(self, model_path: str = "src/model/model.pkl"):
        self.model_path = model_path
        self.model: Optional[Any] = None
    
    def load_model(self, path: Optional[str] = None) -> None:
        """Load model from pickle file"""
        model_path = path or self.model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_model_from_bytes(self, model_bytes: bytes) -> None:
        """Load model from bytes (for file upload)"""
        try:
            self.model = pickle.loads(model_bytes)
            logger.info("Model loaded from uploaded file")
        except Exception as e:
            logger.error(f"Failed to load model from bytes: {e}")
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def predict(self, data: dict) -> dict:
        """
        Realiza a predição usando o modelo carregado.
        
        Args:
            data (dict): dicionário com as features transformadas
        
        Returns:
            dict com 'prediction' e 'probability'
        """
        # Separa PassengerId (se presente) e transforma os dados em DataFrame
        passenger_id = data.get('PassengerId', None)
        data = {k: v for k, v in data.items() if k != 'PassengerId'}
        
        df = pd.DataFrame([data])

        # Realiza a predição
        prediction = self.model.predict(df)
        probability = self.model.predict_proba(df)

        return {
            'passenger_id': passenger_id,
            'prediction': int(prediction),
            'probability': probability
        }
