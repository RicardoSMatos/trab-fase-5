"""
Módulo de configuração do projeto.

Centraliza todas as configurações e parâmetros do projeto.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Diretórios base
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Criar diretórios se não existirem
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configurações de dados
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL', 'PETR4.SA')
START_DATE = os.getenv('START_DATE', '2020-01-01')
END_DATE = os.getenv('END_DATE', '2024-12-31')

# Configurações de modelo
TRAIN_TEST_SPLIT = 0.8  # 80% treino, 20% teste
VALIDATION_SPLIT = 0.2  # 20% do treino para validação
RANDOM_STATE = 42

# Configurações LSTM
LSTM_CONFIG = {
    'sequence_length': 60,  # Número de dias para prever o próximo
    'units': [50, 50],  # Neurônios em cada camada LSTM
    'dropout': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001
}

# Configurações ARIMA
ARIMA_CONFIG = {
    'order': (5, 1, 0),  # (p, d, q) - será otimizado
    'seasonal_order': (0, 0, 0, 0),
    'max_p': 5,
    'max_d': 2,
    'max_q': 5
}

# Configurações Prophet
PROPHET_CONFIG = {
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10,
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False
}

# Configurações da API
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
API_RELOAD = os.getenv('API_RELOAD', 'True').lower() == 'true'

# Configurações de monitoramento
ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'True').lower() == 'true'
METRICS_PORT = int(os.getenv('METRICS_PORT', 8001))

# Configurações de logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = LOGS_DIR / 'app.log'

# Configurações MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'stock_prediction')

# Caminhos de modelos
MODEL_PATH = MODELS_DIR / 'best_model.pkl'
SCALER_PATH = MODELS_DIR / 'scaler.pkl'
METADATA_PATH = MODELS_DIR / 'model_metadata.json'
