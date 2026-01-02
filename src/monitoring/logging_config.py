"""
Configuração de logging para a API.

Este módulo configura logging estruturado com rotação de arquivos
e diferentes níveis de log para desenvolvimento e produção.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_to_file: bool = True):
    """
    Configura o sistema de logging.
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Se deve salvar logs em arquivo
    """
    # Criar diretório de logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Formatar logs
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Converter string para nível
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remover handlers existentes
    root_logger.handlers = []
    
    # Handler para console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Handler para arquivo (se habilitado)
    if log_to_file:
        # Arquivo geral
        file_handler = RotatingFileHandler(
            log_dir / "api.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Arquivo de erros
        error_handler = RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    # Silenciar logs muito verbosos de bibliotecas externas
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logging.info(f"Logging configurado - Nível: {log_level}, Arquivo: {log_to_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Obtém um logger configurado.
    
    Args:
        name: Nome do logger (geralmente __name__ do módulo)
    
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)
