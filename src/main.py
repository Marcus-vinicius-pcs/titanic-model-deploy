from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime
from typing import Dict, Any, List
import uvicorn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import ModelManager
from src.data_processing import TitanicPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.main")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API para predição de sobrevivência no Titanic",
    version="1.0.0",
)

model_manager = ModelManager()
preprocessor = TitanicPreprocessor()
prediction_history: List[Dict[str, Any]] = []


class PassengerData(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
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
                "Embarked": "S",
            }
        }


class PredictionResponse(BaseModel):
    PassengerId: int
    prediction: int
    probability: list
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Carrega o modelo ao iniciar a aplicação"""
    try:
        model_manager.load_model()
        logger.info("Modelo carregado com sucesso ao iniciar a aplicação")
    except Exception as e:
        logger.warning(f"Falha ao carregar o modelo ao iniciar: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerData):
    """Realiza a predição de sobrevivência de um passageiro"""
    try:
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=503, detail="Modelo não carregado")

        passenger_dict = passenger.dict()
        logger.info(
            f"Requisição de predição recebida para PassengerId {passenger.PassengerId}"
        )
        logger.debug(f"Dados do passageiro: {passenger_dict}")

        processed_data = preprocessor.process(passenger_dict)
        logger.debug(f"Dados processados: {processed_data}")
        logger.info(f"Usando modelo: {model_manager.model}")
        prediction = model_manager.predict(processed_data)

        logger.info(
            f"Predição realizada para PassengerId {passenger.PassengerId}: {prediction['prediction']} (prob: {prediction['probability']})"
        )
        response = PredictionResponse(
            PassengerId=passenger.PassengerId,
            prediction=prediction["prediction"],
            probability=prediction["probability"][0],
            timestamp=datetime.now().isoformat(),
        )

        prediction_history.append(
            {
                "input": passenger_dict,
                "output": response.dict(),
                "timestamp": response.timestamp,
            }
        )

        logger.info(f"Resposta da predição: {response}")
        return response

    except Exception as e:
        logger.error(f"Erro ao realizar predição: {e}")
        raise HTTPException(status_code=500, detail=f"Falha na predição: {str(e)}")


@app.post("/load")
async def load_model(file: UploadFile = File(...)):
    """Carrega um novo modelo a partir de um arquivo pickle enviado"""
    try:
        if not file.filename.endswith(".pkl"):
            raise HTTPException(status_code=400, detail="O arquivo deve ser .pkl")

        contents = await file.read()
        model_manager.load_model_from_bytes(contents)

        logger.info(f"Novo modelo carregado: {file.filename}")
        return {"message": f"Modelo {file.filename} carregado com sucesso"}

    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo: {str(e)}")


@app.get("/history")
async def get_history():
    """Retorna o histórico das predições realizadas"""
    return {
        "total_predictions": len(prediction_history),
        "history": prediction_history[-100:],
    }


@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde da aplicação"""
    return {
        "status": "ok",
        "modelo_carregado": model_manager.is_model_loaded(),
        "timestamp": datetime.now().isoformat(),
        "total_predicoes": len(prediction_history),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
