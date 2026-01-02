"""
Módulo de preprocessamento de dados para séries temporais.

Este módulo contém funções para preparar os dados para modelagem,
incluindo normalização, feature engineering e divisão em conjuntos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import logging
from typing import Tuple, Optional, Dict, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    Classe para preprocessar dados de séries temporais.
    
    Realiza normalização, feature engineering e preparação dos dados
    para treinamento de modelos.
    """
    
    def __init__(self, scaler_type: str = 'minmax'):
        """
        Inicializa o preprocessador.
        
        Args:
            scaler_type: Tipo de scaler ('minmax' ou 'standard')
        """
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de engenharia para séries temporais.
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame com features adicionais
        """
        df = df.copy()
        
        logger.info("Criando features de engenharia...")
        
        # 1. Retornos
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Médias Móveis
        for window in [7, 21, 50, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
        
        # 3. Médias Móveis Exponenciais
        for span in [12, 26]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        
        # 4. Volatilidade (desvio padrão móvel)
        for window in [7, 21, 30]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
        
        # 5. RSI (Relative Strength Index)
        df['RSI_14'] = self._calculate_rsi(df['Close'], window=14)
        
        # 6. MACD (Moving Average Convergence Divergence)
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 7. Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # 8. Features temporais
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfMonth'] = df.index.day
        df['WeekOfYear'] = df.index.isocalendar().week
        
        # 9. Lag features (valores passados)
        for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # 10. Diferenças
        df['Close_Diff_1'] = df['Close'].diff(1)
        df['Close_Diff_7'] = df['Close'].diff(7)
        
        # 11. Volume features
        df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
        df['Volume_MA_21'] = df['Volume'].rolling(window=21).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_7']
        
        # 12. High-Low range
        df['High_Low_Range'] = df['High'] - df['Low']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        logger.info(f"Features criadas. Total de colunas: {len(df.columns)}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calcula o Relative Strength Index (RSI).
        
        Args:
            prices: Série de preços
            window: Janela de tempo
            
        Returns:
            Série com valores RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calcula MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Série de preços
            fast: Período rápido
            slow: Período lento
            signal: Período do sinal
            
        Returns:
            Tupla (MACD, Signal Line)
        """
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Close',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepara dados para treinamento, validação e teste.
        
        Args:
            df: DataFrame com todas as features
            target_col: Nome da coluna alvo
            test_size: Proporção de dados para teste
            val_size: Proporção de treino para validação
            
        Returns:
            Dicionário com datasets de treino, validação e teste
        """
        logger.info("Preparando dados para modelagem...")
        
        # Remover linhas com NaN (gerados pelas operações de rolling)
        df_clean = df.dropna()
        logger.info(f"Dados após remoção de NaN: {len(df_clean)} registros")
        
        # Separar features e target
        feature_cols = [col for col in df_clean.columns if col != target_col]
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Dividir em treino e teste (respeitando ordem temporal)
        split_idx = int(len(df_clean) * (1 - test_size))
        
        X_temp = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_temp = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Dividir treino em treino e validação
        val_split_idx = int(len(X_temp) * (1 - val_size))
        
        X_train = X_temp.iloc[:val_split_idx]
        X_val = X_temp.iloc[val_split_idx:]
        y_train = y_temp.iloc[:val_split_idx]
        y_val = y_temp.iloc[val_split_idx:]
        
        logger.info(f"Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
    
    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """
        Ajusta o scaler aos dados de treino.
        
        Args:
            X_train: Dados de treino
        """
        logger.info(f"Ajustando scaler ({self.scaler_type})...")
        self.scaler.fit(X_train)
        self.feature_columns = X_train.columns.tolist()
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforma dados usando o scaler ajustado.
        
        Args:
            X: Dados a transformar
            
        Returns:
            Array numpy com dados normalizados
        """
        return self.scaler.transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverte a transformação do scaler.
        
        Args:
            X_scaled: Dados normalizados
            
        Returns:
            Array numpy com dados originais
        """
        return self.scaler.inverse_transform(X_scaled)
    
    def save_scaler(self, filepath: Path) -> None:
        """
        Salva o scaler em arquivo.
        
        Args:
            filepath: Caminho para salvar
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'scaler_type': self.scaler_type
        }, filepath)
        logger.info(f"Scaler salvo em: {filepath}")
    
    def load_scaler(self, filepath: Path) -> None:
        """
        Carrega scaler de arquivo.
        
        Args:
            filepath: Caminho do arquivo
        """
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.scaler_type = data['scaler_type']
        logger.info(f"Scaler carregado de: {filepath}")


def prepare_for_lstm(
    data: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara dados para modelo LSTM (sequências).
    
    Args:
        data: Array com dados
        sequence_length: Comprimento da sequência
        
    Returns:
        Tupla (X, y) com sequências e targets
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Sequências LSTM criadas: X shape {X.shape}, y shape {y.shape}")
    
    return X, y


def main():
    """
    Função principal para testar o preprocessamento.
    """
    from pathlib import Path
    import pandas as pd
    
    # Carregar dados brutos
    data_path = Path('data/raw/stock_data_raw.csv')
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    print(f"Dados carregados: {df.shape}")
    
    # Criar preprocessador
    preprocessor = TimeSeriesPreprocessor(scaler_type='minmax')
    
    # Criar features
    df_features = preprocessor.create_features(df)
    print(f"\nApós feature engineering: {df_features.shape}")
    print(f"Colunas: {df_features.columns.tolist()[:10]}...")
    
    # Preparar dados
    datasets = preprocessor.prepare_data(df_features, target_col='Close')
    
    print(f"\n=== Divisão dos Dados ===")
    print(f"Treino: {len(datasets['X_train'])} registros")
    print(f"Validação: {len(datasets['X_val'])} registros")
    print(f"Teste: {len(datasets['X_test'])} registros")
    
    # Ajustar e transformar
    preprocessor.fit_scaler(datasets['X_train'])
    X_train_scaled = preprocessor.transform(datasets['X_train'])
    
    print(f"\nDados normalizados: {X_train_scaled.shape}")
    print(f"Min: {X_train_scaled.min():.4f}, Max: {X_train_scaled.max():.4f}")
    
    # Salvar preprocessador
    scaler_path = Path('models/scaler.pkl')
    preprocessor.save_scaler(scaler_path)
    
    # Salvar dados processados
    output_path = Path('data/processed/stock_data_features.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path)
    print(f"\nDados processados salvos em: {output_path}")


if __name__ == "__main__":
    main()
