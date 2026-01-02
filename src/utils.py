"""
Módulo de utilidades gerais do projeto.

Funções auxiliares reutilizáveis em diferentes partes do projeto.
"""

import json
import joblib
import logging
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import pandas as pd
import numpy as np


def setup_logging(log_file: Path, log_level: str = 'INFO') -> logging.Logger:
    """
    Configura o sistema de logging.
    
    Args:
        log_file: Caminho para o arquivo de log
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Logger configurado
    """
    # Criar diretório de logs se não existir
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configurar formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_json(data: Dict, filepath: Path) -> None:
    """
    Salva dicionário em arquivo JSON.
    
    Args:
        data: Dicionário a ser salvo
        filepath: Caminho do arquivo
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filepath: Path) -> Dict:
    """
    Carrega dados de arquivo JSON.
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Dicionário com os dados
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_model(model: Any, filepath: Path) -> None:
    """
    Serializa e salva um modelo.
    
    Args:
        model: Modelo a ser salvo
        filepath: Caminho para salvar
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logging.info(f"Modelo salvo em: {filepath}")


def load_model(filepath: Path) -> Any:
    """
    Carrega um modelo serializado.
    
    Args:
        filepath: Caminho do modelo
        
    Returns:
        Modelo carregado
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
    
    model = joblib.load(filepath)
    logging.info(f"Modelo carregado de: {filepath}")
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de avaliação para regressão.
    
    Args:
        y_true: Valores reais
        y_pred: Valores previstos
        
    Returns:
        Dicionário com as métricas
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Remover valores NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Calcular métricas
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAPE': float(mape)
    }
    
    return metrics


def create_sequences(data: np.ndarray, sequence_length: int) -> tuple:
    """
    Cria sequências para modelos de séries temporais (LSTM).
    
    Args:
        data: Array com os dados
        sequence_length: Comprimento da sequência
        
    Returns:
        Tupla (X, y) com sequências e targets
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)


def get_timestamp() -> str:
    """
    Retorna timestamp formatado.
    
    Returns:
        String com timestamp
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_currency(value: float, currency: str = 'R$') -> str:
    """
    Formata valor como moeda.
    
    Args:
        value: Valor numérico
        currency: Símbolo da moeda
        
    Returns:
        String formatada
    """
    return f"{currency} {value:,.2f}"


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Valida se DataFrame possui colunas necessárias.
    
    Args:
        df: DataFrame a validar
        required_columns: Lista de colunas necessárias
        
    Returns:
        True se válido, False caso contrário
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        logging.error(f"Colunas faltantes: {missing_columns}")
        return False
    
    return True


def get_model_info(model_path: Path) -> Dict:
    """
    Obtém informações sobre um modelo salvo.
    
    Args:
        model_path: Caminho do modelo
        
    Returns:
        Dicionário com informações
    """
    if not model_path.exists():
        return {'exists': False}
    
    stat = model_path.stat()
    
    return {
        'exists': True,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
    }
