from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List
import uvicorn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import ModelManager
from src.data_processing import TitanicPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('src.main')

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting passenger survival on the Titanic",
    version="1.0.0"
)

# Global instances
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
                "Embarked": "S"
            }
        }

class PredictionResponse(BaseModel):
    PassengerId: int
    prediction: int
    probability: list
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model_manager.load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Failed to load model on startup: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerData):
    """Predict passenger survival"""
    try:
        if not model_manager.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")

        passenger_dict = passenger.dict()
        logger.info(f"Received prediction request for PassengerId {passenger.PassengerId}")
        logger.debug(f"Passenger data: {passenger_dict}")

        processed_data = preprocessor.process(passenger_dict)
        logger.debug(f"Processed data: {processed_data}")
        logger.info(f"Usando modelo: {model_manager.model}")
        prediction = model_manager.predict(processed_data)

        logger.info(f"Prediction made for PassengerId {passenger.PassengerId}: {prediction['prediction']} (prob: {prediction['probability']})")
        response = PredictionResponse(
            PassengerId=passenger.PassengerId,
            prediction=prediction['prediction'],
            probability=prediction['probability'][0], 
            timestamp=datetime.now().isoformat()
        )

        prediction_history.append({
            "input": passenger_dict,
            "output": response.dict(),
            "timestamp": response.timestamp
        })

        logger.info(f"Prediction response: {response}")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/load")
async def load_model(file: UploadFile = File(...)):
    """Load a new model from uploaded pickle file"""
    try:
        if not file.filename.endswith('.pkl'):
            raise HTTPException(status_code=400, detail="File must be a .pkl file")
        
        contents = await file.read()
        model_manager.load_model_from_bytes(contents)
        
        logger.info(f"New model loaded: {file.filename}")
        return {"message": f"Model {file.filename} loaded successfully"}
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/history")
async def get_history():
    """Get prediction history"""
    return {
        "total_predictions": len(prediction_history),
        "history": prediction_history[-100:]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_model_loaded(),
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(prediction_history)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)