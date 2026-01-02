"""
Gerenciador de Portfolio de Ações
Permite adicionar, remover e listar ações no portfolio
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Gerenciador de portfolio de ações com persistência em JSON"""
    
    def __init__(self, portfolio_file: str = "data/portfolio.json"):
        """
        Inicializa o gerenciador de portfolio
        
        Args:
            portfolio_file: Caminho para arquivo JSON do portfolio
        """
        self.portfolio_file = Path(portfolio_file)
        self.portfolio = self._load_portfolio()
        logger.info(f"Portfolio Manager inicializado: {len(self.portfolio)} ações")
    
    def _load_portfolio(self) -> Dict:
        """Carrega portfolio do arquivo JSON"""
        if self.portfolio_file.exists():
            try:
                with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Portfolio carregado: {len(data.get('stocks', {}))} ações")
                return data
            except Exception as e:
                logger.error(f"Erro ao carregar portfolio: {e}")
                return self._create_empty_portfolio()
        else:
            logger.info("Arquivo de portfolio não encontrado, criando novo")
            return self._create_empty_portfolio()
    
    def _create_empty_portfolio(self) -> Dict:
        """Cria estrutura de portfolio vazio"""
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "stocks": {}
        }
    
    def _save_portfolio(self):
        """Salva portfolio no arquivo JSON"""
        try:
            self.portfolio["updated_at"] = datetime.now().isoformat()
            self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(self.portfolio, f, indent=2, ensure_ascii=False)
            logger.info(f"Portfolio salvo: {len(self.portfolio['stocks'])} ações")
        except Exception as e:
            logger.error(f"Erro ao salvar portfolio: {e}")
            raise
    
    def add_stock(self, symbol: str, name: Optional[str] = None, 
                  period: str = "2y", description: Optional[str] = None) -> Dict:
        """
        Adiciona ação ao portfolio
        
        Args:
            symbol: Código da ação (ex: PETR4.SA)
            name: Nome da empresa
            period: Período de dados históricos
            description: Descrição adicional
            
        Returns:
            Dados da ação adicionada
        """
        if symbol in self.portfolio["stocks"]:
            logger.warning(f"Ação {symbol} já existe no portfolio")
            raise ValueError(f"Ação {symbol} já existe no portfolio")
        
        stock_data = {
            "symbol": symbol,
            "name": name or symbol,
            "description": description,
            "period": period,
            "added_at": datetime.now().isoformat(),
            "status": "pending",  # pending, training, trained, error
            "model_path": None,
            "last_trained": None,
            "metrics": {}
        }
        
        self.portfolio["stocks"][symbol] = stock_data
        self._save_portfolio()
        
        logger.info(f"Ação {symbol} adicionada ao portfolio")
        return stock_data
    
    def remove_stock(self, symbol: str) -> bool:
        """
        Remove ação do portfolio e seus artefatos
        
        Args:
            symbol: Código da ação
            
        Returns:
            True se removido com sucesso
        """
        if symbol not in self.portfolio["stocks"]:
            logger.warning(f"Ação {symbol} não encontrada no portfolio")
            return False
        
        # Remover arquivos associados
        models_dir = Path("models")
        patterns = [
            f"{symbol}_model.keras",
            f"{symbol}_scaler.pkl",
            f"{symbol}_target_scaler.pkl",
            f"{symbol}_metadata.json"
        ]
        
        removed_files = []
        for pattern in patterns:
            file_path = models_dir / pattern
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    logger.info(f"Arquivo removido: {file_path}")
                except Exception as e:
                    logger.error(f"Erro ao remover {file_path}: {e}")
        
        # Remover do portfolio
        del self.portfolio["stocks"][symbol]
        self._save_portfolio()
        
        logger.info(f"Ação {symbol} removida do portfolio. Arquivos deletados: {len(removed_files)}")
        return True
    
    def get_stock(self, symbol: str) -> Optional[Dict]:
        """
        Obtém dados de uma ação
        
        Args:
            symbol: Código da ação
            
        Returns:
            Dados da ação ou None
        """
        return self.portfolio["stocks"].get(symbol)
    
    def list_stocks(self) -> List[Dict]:
        """
        Lista todas as ações do portfolio
        
        Returns:
            Lista de ações
        """
        return list(self.portfolio["stocks"].values())
    
    def update_stock_status(self, symbol: str, status: str, 
                           model_path: Optional[str] = None,
                           metrics: Optional[Dict] = None):
        """
        Atualiza status de uma ação
        
        Args:
            symbol: Código da ação
            status: Novo status (pending, training, trained, error)
            model_path: Caminho do modelo treinado
            metrics: Métricas de avaliação
        """
        if symbol not in self.portfolio["stocks"]:
            raise ValueError(f"Ação {symbol} não encontrada")
        
        stock = self.portfolio["stocks"][symbol]
        stock["status"] = status
        stock["updated_at"] = datetime.now().isoformat()
        
        if status == "trained":
            stock["last_trained"] = datetime.now().isoformat()
        
        if model_path:
            stock["model_path"] = model_path
        
        if metrics:
            stock["metrics"] = metrics
        
        self._save_portfolio()
        logger.info(f"Status da ação {symbol} atualizado para: {status}")
    
    def update_training_status(self, symbol: str, status: str, 
                              error: Optional[str] = None):
        """
        Atualiza status de treinamento
        
        Args:
            symbol: Código da ação
            status: Status (training, completed, failed)
            error: Mensagem de erro se falhou
        """
        if symbol not in self.portfolio["stocks"]:
            raise ValueError(f"Ação {symbol} não encontrada")
        
        stock = self.portfolio["stocks"][symbol]
        stock["training_status"] = status
        stock["updated_at"] = datetime.now().isoformat()
        
        if status == "training":
            stock["training_started"] = datetime.now().isoformat()
        elif status == "completed":
            stock["status"] = "trained"
            stock["last_trained"] = datetime.now().isoformat()
        elif status == "failed":
            stock["status"] = "error"
            stock["error"] = error
        
        self._save_portfolio()
        logger.info(f"Status de treinamento da ação {symbol}: {status}")
    
    def get_summary(self) -> Dict:
        """
        Obtém resumo do portfolio
        
        Returns:
            Resumo com estatísticas
        """
        stocks = self.portfolio["stocks"]
        
        total = len(stocks)
        by_status = {}
        for stock in stocks.values():
            status = stock.get("status", "pending")
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_stocks": total,
            "by_status": by_status,
            "created_at": self.portfolio.get("created_at"),
            "updated_at": self.portfolio.get("updated_at")
        }
    
    def get_trained_stocks(self) -> List[str]:
        """
        Retorna lista de símbolos de ações treinadas
        
        Returns:
            Lista de símbolos com modelos treinados
        """
        trained = []
        for symbol, stock in self.portfolio["stocks"].items():
            if stock.get("status") == "trained" or stock.get("training_status") == "completed":
                trained.append(symbol)
        
        logger.info(f"Encontradas {len(trained)} ações treinadas: {trained}")
        return trained
    
    def update_prediction(self, symbol: str, predictions: List[float], 
                         confidence: float):
        """
        Atualiza predições de uma ação no portfolio
        
        Args:
            symbol: Código da ação
            predictions: Lista de preços previstos
            confidence: Intervalo de confiança (RMSE)
        """
        if symbol not in self.portfolio["stocks"]:
            logger.warning(f"Ação {symbol} não encontrada no portfolio")
            return
        
        stock = self.portfolio["stocks"][symbol]
        stock["last_prediction"] = {
            "predictions": predictions,
            "confidence": confidence,
            "predicted_at": datetime.now().isoformat()
        }
        stock["updated_at"] = datetime.now().isoformat()
        
        self._save_portfolio()
        logger.info(f"Predição atualizada para {symbol}")
    
    def clear_portfolio(self):
        """Remove todas as ações do portfolio"""
        symbols = list(self.portfolio["stocks"].keys())
        for symbol in symbols:
            self.remove_stock(symbol)
        logger.info("Portfolio limpo completamente")
