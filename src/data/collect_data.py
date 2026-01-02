"""
Script para coleta de dados históricos de ações da bolsa de valores.

Este módulo utiliza a biblioteca yfinance para baixar dados históricos
de ações e salvá-los em formato CSV.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional
import os

# Forçar yfinance a usar urllib em vez de curl_cffi
os.environ['YF_FORCE_URLLIB'] = '1'

import yfinance as yf

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    Classe para coletar dados históricos de ações.
    
    Attributes:
        ticker (str): Símbolo da ação (ex: 'PETR4.SA')
        start_date (str): Data inicial no formato 'YYYY-MM-DD'
        end_date (str): Data final no formato 'YYYY-MM-DD'
    """
    
    def __init__(
        self,
        ticker: str,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ):
        """
        Inicializa o coletor de dados.
        
        Args:
            ticker: Símbolo da ação a ser coletada
            start_date: Data inicial para coleta
            end_date: Data final para coleta (padrão: hoje)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Criar diretórios se não existirem
        self.raw_data_path = Path('data/raw')
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Coletor inicializado para {ticker}")
    
    def download_data(self) -> pd.DataFrame:
        """
        Baixa os dados históricos da ação.
        
        Returns:
            DataFrame com os dados históricos
            
        Raises:
            Exception: Se houver erro no download
        """
        try:
            logger.info(
                f"Baixando dados de {self.ticker} "
                f"de {self.start_date} até {self.end_date}"
            )
            
            # Configurar sessão para evitar erro de impersonation
            import yfinance as yf_module
            
            try:
                # Primeira tentativa: usar download direto (mais simples)
                data = yf_module.download(
                    self.ticker,
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    repair=True,
                    keepna=False,
                    actions=False,
                    timeout=30
                )
            except Exception as first_error:
                logger.warning(f"Primeira tentativa falhou: {first_error}. Tentando método alternativo...")
                
                # Segunda tentativa: usar Ticker.history sem impersonation
                ticker_obj = yf_module.Ticker(self.ticker)
                data = ticker_obj.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval="1d",
                    auto_adjust=True,
                    back_adjust=False,
                    repair=True,
                    keepna=False,
                    actions=False,
                    timeout=30
                )
            
            if data.empty:
                raise ValueError(f"Nenhum dado encontrado para {self.ticker}")
            
            # Corrigir MultiIndex se presente (quando yf.download retorna tuplas)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Renomear colunas para padrão consistente
            if 'Close' not in data.columns and 'close' in [c.lower() for c in data.columns]:
                data.columns = [col.capitalize() for col in data.columns]
            
            # Garantir que o índice é Date
            if data.index.name != 'Date':
                data.index.name = 'Date'
            
            logger.info(f"Download concluído: {len(data)} registros")
            return data
            
        except Exception as e:
            logger.error(f"Erro ao baixar dados: {str(e)}")
            raise
    
    def get_stock_info(self) -> dict:
        """
        Obtém informações sobre a ação.
        
        Returns:
            Dicionário com informações da ação
        """
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            # Selecionar informações relevantes
            relevant_info = {
                'symbol': info.get('symbol', 'N/A'),
                'longName': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'currency': info.get('currency', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
            }
            
            logger.info(f"Informações obtidas para {self.ticker}")
            return relevant_info
            
        except Exception as e:
            logger.warning(f"Não foi possível obter informações: {str(e)}")
            return {}
    
    def collect(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Método de conveniência para coletar dados.
        
        Args:
            start_date: Data inicial (sobrescreve a do construtor)
            end_date: Data final (sobrescreve a do construtor)
            
        Returns:
            DataFrame com os dados coletados ou None se houver erro
        """
        try:
            # Atualizar datas se fornecidas
            if start_date:
                self.start_date = start_date
            if end_date:
                self.end_date = end_date
            
            # Baixar e retornar dados
            return self.download_data()
            
        except Exception as e:
            logger.error(f"Erro ao coletar dados de {self.ticker}: {str(e)}")
            return None
    
    def save_data(self, data: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Salva os dados em arquivo CSV.
        
        Args:
            data: DataFrame com os dados
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho do arquivo salvo
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.ticker}_{timestamp}.csv"
        
        filepath = self.raw_data_path / filename
        
        try:
            data.to_csv(filepath)
            logger.info(f"Dados salvos em: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {str(e)}")
            raise
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Gera um resumo estatístico dos dados.
        
        Args:
            data: DataFrame com os dados
            
        Returns:
            Dicionário com o resumo
        """
        summary = {
            'total_records': len(data),
            'start_date': str(data.index.min()),
            'end_date': str(data.index.max()),
            'missing_values': data.isnull().sum().to_dict(),
            'price_range': {
                'min': float(data['Close'].min().iloc[0] if hasattr(data['Close'].min(), 'iloc') else data['Close'].min()),
                'max': float(data['Close'].max().iloc[0] if hasattr(data['Close'].max(), 'iloc') else data['Close'].max()),
                'mean': float(data['Close'].mean().iloc[0] if hasattr(data['Close'].mean(), 'iloc') else data['Close'].mean()),
                'std': float(data['Close'].std().iloc[0] if hasattr(data['Close'].std(), 'iloc') else data['Close'].std())
            }
        }
        
        return summary


def main():
    """
    Função principal para executar a coleta de dados.
    """
    # Configurações
    TICKER = "PETR4.SA"  # Petrobras
    START_DATE = "2020-01-01"
    
    # Criar coletor
    collector = StockDataCollector(
        ticker=TICKER,
        start_date=START_DATE
    )
    
    # Obter informações da ação
    info = collector.get_stock_info()
    print("\n=== Informações da Ação ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Baixar dados
    data = collector.download_data()
    
    # Exibir resumo
    summary = collector.get_data_summary(data)
    print("\n=== Resumo dos Dados ===")
    print(f"Total de registros: {summary['total_records']}")
    print(f"Período: {summary['start_date']} até {summary['end_date']}")
    print(f"Preço mínimo: R$ {summary['price_range']['min']:.2f}")
    print(f"Preço máximo: R$ {summary['price_range']['max']:.2f}")
    print(f"Preço médio: R$ {summary['price_range']['mean']:.2f}")
    
    # Salvar dados
    filepath = collector.save_data(data, filename='stock_data_raw.csv')
    print(f"\nDados salvos em: {filepath}")
    
    # Exibir primeiras linhas
    print("\n=== Primeiras linhas dos dados ===")
    print(data.head())


if __name__ == "__main__":
    main()
