import pandas as pd
import pickle
import logging
import re
from typing import Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("src.data_processing")


class CategoricalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["Title"] = X["Name"].apply(
            lambda name: (
                re.search(r" ([A-Za-z]+)\.", name).group(1)
                if re.search(r" ([A-Za-z]+)\.", name)
                else ""
            )
        )
        X["Title_Mapped"] = (
            X["Title"]
            .map(
                {
                    "Mr": "Mr",
                    "Miss": "Miss",
                    "Mrs": "Mrs",
                    "Master": "Master",
                    "Dr": "Rare",
                    "Rev": "Rare",
                    "Mlle": "Miss",
                    "Major": "Rare",
                    "Col": "Rare",
                    "Sir": "Rare",
                    "Mme": "Mrs",
                    "Don": "Rare",
                    "Lady": "Rare",
                    "Countess": "Rare",
                    "Jonkheer": "Rare",
                    "Dona": "Rare",
                    "Capt": "Rare",
                }
            )
            .fillna("Rare")
        )

        # Sex: binarizar
        X["Sex"] = X["Sex"].apply(lambda s: 1 if s == "male" else 0)

        # Deck e quantidade de cabines
        X["Deck"] = X["Cabin"].apply(lambda c: c[0] if pd.notna(c) else "Unknown")
        X["Cabin_Count"] = X["Cabin"].apply(
            lambda c: len(c.split()) if pd.notna(c) else 0
        )

        # Prefixo do ticket
        X["Ticket_Prefix"] = X["Ticket"].apply(
            lambda t: (
                re.match(r"([A-Z]+)", str(t).upper()).group(1)
                if re.match(r"([A-Z]+)", str(t).upper())
                else "Numeric"
            )
        )

        # Faixa etária
        X["Age"] = X["Age"].fillna(X["Age"].median())
        X["Age_Group"] = pd.cut(
            X["Age"],
            bins=[0, 12, 18, 35, 60, 100],
            labels=["Child", "Teenager", "Young", "Adult", "Elderly"],
        )

        # Faixa de tarifa
        X["Fare_Group"] = X["Fare"].apply(
            lambda f: (
                (
                    "Low"
                    if f <= 7.91
                    else "Medium" if f <= 14.454 else "High" if f <= 31 else "Very_High"
                )
                if pd.notna(f)
                else "Unknown"
            )
        )

        # Sozinho + tamanho da família
        X["Family_Size"] = X["Parch"] + X["SibSp"]
        X["Alone"] = (X["Family_Size"] == 0).astype(int)

        # Preencher 'Embarked' ausente
        X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

        # Tamanho do grupo do ticket (quantidade de pessoas com o mesmo ticket)
        ticket_counts = X["Ticket"].value_counts()
        X["Ticket_Group_Size"] = X["Ticket"].map(ticket_counts)

        return X


class TitanicPreprocessor:
    """Usa a pipeline treinada para preprocessar os dados"""

    def __init__(self, pipeline_path="src/model/pipeline.pkl"):
        with open(pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)

        # Recupera os nomes das colunas transformadas
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self):
        """
        Extrai os nomes das features transformadas da pipeline.
        """
        try:
            column_transformer = self.pipeline.named_steps["preprocessing"]
            ohe = column_transformer.named_transformers_["ohe"]
            ohe_features = list(
                ohe.get_feature_names_out(column_transformer.transformers[0][2])
            )  # cat_ohe_cols

            passthrough_features = column_transformer.transformers[1][
                2
            ]  # ordinal_cat_features
            numeric_features = [
                "Family_Size",
                "Cabin_Count",
                "Ticket_Group_Size",
                "Age",
                "Fare",
            ]

            return ohe_features + passthrough_features + numeric_features

        except Exception as e:
            logger.warning(f"Não foi possível extrair nomes das features: {e}")
            return [
                f"feature_{i}"
                for i in range(
                    self.pipeline.transform(
                        pd.DataFrame(
                            [
                                {
                                    "PassengerId": 0,
                                    "Pclass": 1,
                                    "Name": "",
                                    "Sex": "male",
                                    "Age": 30,
                                    "SibSp": 0,
                                    "Parch": 0,
                                    "Ticket": "12345",
                                    "Fare": 10.0,
                                    "Cabin": None,
                                    "Embarked": "S",
                                }
                            ]
                        )
                    ).shape[1]
                )
            ]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa os dados de entrada usando a pipeline salva e retorna como dicionário, incluindo PassengerId.
        """
        try:
            df_input = pd.DataFrame([data])
            transformed = self.pipeline.transform(df_input)
            df_transformed = pd.DataFrame(transformed, columns=self.feature_names)

            # Adiciona PassengerId de volta
            df_transformed.insert(0, "PassengerId", df_input["PassengerId"].values)

            logger.info("Dados processados com sucesso pela pipeline.")
            return df_transformed.iloc[0].to_dict()

        except Exception as e:
            logger.error(f"Erro ao processar os dados com a pipeline: {e}")
            raise