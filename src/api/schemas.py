"""
Schemas Pydantic para validação de dados da API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Schema para requisição de previsão."""
    
    days: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Número de dias para prever (1-30)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "days": 5
            }
        }


class PredictionResponse(BaseModel):
    """Schema para resposta de previsão."""
    
    predictions: List[float] = Field(description="Preços previstos")
    dates: List[str] = Field(description="Datas das previsões")
    last_price: float = Field(description="Último preço real conhecido")
    confidence: str = Field(description="Nível de confiança da previsão")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [32.45, 32.78, 33.12],
                "dates": ["2025-12-29", "2025-12-30", "2025-12-31"],
                "last_price": 32.10,
                "confidence": "high"
            }
        }


class ModelInfo(BaseModel):
    """Schema para informações do modelo."""
    
    model_name: str
    version: str
    training_date: str
    metrics: dict
    architecture: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "LSTM Stock Predictor",
                "version": "1.0.0",
                "training_date": "2025-12-28",
                "metrics": {
                    "MAE": 0.5432,
                    "RMSE": 0.7123,
                    "R2": 0.8567,
                    "MAPE": 3.45
                },
                "architecture": {
                    "lstm_units": [128, 64, 32],
                    "dropout": 0.2,
                    "sequence_length": 60
                }
            }
        }


class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    
    status: str
    timestamp: str
    model_loaded: bool
    version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-12-28T10:30:00",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Schema para resposta de erro."""
    
    error: str
    detail: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "PredictionError",
                "detail": "Não foi possível fazer a previsão",
                "timestamp": "2025-12-28T10:30:00"
            }
        }
