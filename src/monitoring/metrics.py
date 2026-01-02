"""
Sistema de monitoramento e métricas para a API.

Este módulo implementa tracking de métricas de performance,
logging estruturado e health checks para produção.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class APIMetrics:
    """
    Classe para coletar e armazenar métricas da API.
    
    Métricas coletadas:
    - Total de requisições
    - Tempo médio de resposta
    - Taxa de sucesso/erro
    - Predições realizadas
    - Performance do modelo
    """
    
    def __init__(self):
        """Inicializa o sistema de métricas."""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_predictions": 0,
            "response_times": [],
            "errors": [],
            "start_time": datetime.now().isoformat()
        }
        
        # Arquivo para persistir métricas
        self.metrics_file = Path("logs/metrics.json")
        self.metrics_file.parent.mkdir(exist_ok=True)
        
        logger.info("Sistema de métricas inicializado")
    
    def record_request(self, endpoint: str, success: bool, response_time: float):
        """
        Registra uma requisição.
        
        Args:
            endpoint: Nome do endpoint chamado
            success: Se a requisição foi bem-sucedida
            response_time: Tempo de resposta em segundos
        """
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        self.metrics["response_times"].append({
            "endpoint": endpoint,
            "time": response_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Manter apenas últimas 1000 medições
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
    
    def record_prediction(self, days: int, confidence: str):
        """
        Registra uma predição realizada.
        
        Args:
            days: Número de dias previstos
            confidence: Nível de confiança (high/medium/low)
        """
        self.metrics["total_predictions"] += 1
        logger.info(f"Predição registrada: {days} dias, confiança {confidence}")
    
    def record_error(self, error_type: str, error_msg: str):
        """
        Registra um erro.
        
        Args:
            error_type: Tipo do erro
            error_msg: Mensagem de erro
        """
        self.metrics["errors"].append({
            "type": error_type,
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        # Manter apenas últimos 100 erros
        if len(self.metrics["errors"]) > 100:
            self.metrics["errors"] = self.metrics["errors"][-100:]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna um resumo das métricas.
        
        Returns:
            Dicionário com resumo das métricas
        """
        response_times = [r["time"] for r in self.metrics["response_times"]]
        
        avg_response_time = (
            sum(response_times) / len(response_times) 
            if response_times else 0
        )
        
        success_rate = (
            (self.metrics["successful_requests"] / self.metrics["total_requests"] * 100)
            if self.metrics["total_requests"] > 0 else 0
        )
        
        return {
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": f"{success_rate:.2f}%",
            "total_predictions": self.metrics["total_predictions"],
            "avg_response_time": f"{avg_response_time:.3f}s",
            "total_errors": len(self.metrics["errors"]),
            "uptime": self._calculate_uptime(),
            "last_errors": self.metrics["errors"][-5:] if self.metrics["errors"] else []
        }
    
    def _calculate_uptime(self) -> str:
        """Calcula o tempo de atividade da API."""
        start = datetime.fromisoformat(self.metrics["start_time"])
        uptime = datetime.now() - start
        
        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        
        return f"{days}d {hours}h {minutes}m"
    
    def save_to_file(self):
        """Salva métricas em arquivo JSON."""
        try:
            summary = self.get_summary()
            summary["detailed_metrics"] = self.metrics
            
            with open(self.metrics_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Métricas salvas em {self.metrics_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")


# Timer de contexto para medir tempo de resposta
class ResponseTimer:
    """Context manager para medir tempo de resposta."""
    
    def __init__(self, metrics: APIMetrics, endpoint: str):
        """
        Inicializa o timer.
        
        Args:
            metrics: Instância de APIMetrics
            endpoint: Nome do endpoint sendo medido
        """
        self.metrics = metrics
        self.endpoint = endpoint
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        """Inicia a medição."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finaliza a medição e registra."""
        response_time = time.time() - self.start_time
        
        if exc_type is not None:
            self.success = False
            self.metrics.record_error(
                error_type=exc_type.__name__,
                error_msg=str(exc_val)
            )
        
        self.metrics.record_request(
            endpoint=self.endpoint,
            success=self.success,
            response_time=response_time
        )
        
        return False  # Não suprimir exceções
