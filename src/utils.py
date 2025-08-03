import pickle
import logging
import pandas as pd
from typing import Optional, Any
import os

logger = logging.getLogger("src.utils")


class ModelManager:
    """Gerencia o carregamento, predição e estado do modelo"""

    def __init__(self, model_path: str = "src/model/logistic_regression.pkl"):
        self.model_path = model_path
        self.model: Optional[Any] = None

    def load_model(self, path: Optional[str] = None) -> None:
        """Carrega o modelo a partir de um arquivo pickle"""
        model_path = path or self.model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Modelo carregado de {model_path}")
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo: {e}")
            raise

    def load_model_from_bytes(self, model_bytes: bytes) -> None:
        """Carrega o modelo a partir de bytes (upload de arquivo)"""
        try:
            self.model = pickle.loads(model_bytes)
            logger.info("Modelo carregado a partir do arquivo enviado")
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo a partir dos bytes: {e}")
            raise

    def is_model_loaded(self) -> bool:
        """Verifica se o modelo está carregado"""
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
        passenger_id = data.get("PassengerId", None)
        data = {k: v for k, v in data.items() if k != "PassengerId"}

        df = pd.DataFrame([data])

        # Realiza a predição
        prediction = self.model.predict(df)
        probability = self.model.predict_proba(df)

        return {
            "passenger_id": passenger_id,
            "prediction": int(prediction),
            "probability": probability,
        }