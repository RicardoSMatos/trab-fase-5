"""
Implementação do modelo LSTM para previsão de preços de ações.

Este módulo implementa uma arquitetura LSTM (Long Short-Term Memory)
otimizada para séries temporais financeiras.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMStockPredictor:
    """
    Modelo LSTM para previsão de preços de ações.
    
    Implementa uma arquitetura de rede neural recorrente com células LSTM
    para capturar dependências temporais em dados financeiros.
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        n_features: int = 1,
        lstm_units: list = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        bidirectional: bool = True
    ):
        """
        Inicializa o modelo LSTM.
        
        Args:
            sequence_length: Número de timesteps para input
            n_features: Número de features de entrada
            lstm_units: Lista com número de unidades em cada camada LSTM
            dropout_rate: Taxa de dropout para regularização
            learning_rate: Taxa de aprendizado do otimizador
            bidirectional: Usar LSTM bidirecional
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        
        self.model = None
        self.history = None
        
        logger.info(f"LSTM Predictor inicializado:")
        logger.info(f"  Sequence length: {sequence_length}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  LSTM units: {lstm_units}")
        logger.info(f"  Bidirectional: {bidirectional}")
    
    def build_model(self) -> Sequential:
        """
        Constrói a arquitetura do modelo LSTM.
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Construindo arquitetura do modelo...")
        
        model = Sequential(name='LSTM_Stock_Predictor')
        
        # Primeira camada LSTM
        if self.bidirectional:
            model.add(Bidirectional(
                LSTM(
                    self.lstm_units[0],
                    return_sequences=True if len(self.lstm_units) > 1 else False,
                    input_shape=(self.sequence_length, self.n_features)
                ),
                name='bidirectional_lstm_1'
            ))
        else:
            model.add(LSTM(
                self.lstm_units[0],
                return_sequences=True if len(self.lstm_units) > 1 else False,
                input_shape=(self.sequence_length, self.n_features),
                name='lstm_1'
            ))
        
        model.add(Dropout(self.dropout_rate, name='dropout_1'))
        
        # Camadas LSTM intermediárias
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)
            
            if self.bidirectional:
                model.add(Bidirectional(
                    LSTM(units, return_sequences=return_seq),
                    name=f'bidirectional_lstm_{i}'
                ))
            else:
                model.add(LSTM(
                    units,
                    return_sequences=return_seq,
                    name=f'lstm_{i}'
                ))
            
            model.add(Dropout(self.dropout_rate, name=f'dropout_{i}'))
        
        # Camadas densas finais
        model.add(Dense(32, activation='relu', name='dense_1'))
        model.add(Dropout(self.dropout_rate, name='dropout_final'))
        model.add(Dense(1, activation='linear', name='output'))
        
        # Compilar modelo
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        logger.info("\n" + "="*60)
        logger.info("ARQUITETURA DO MODELO")
        logger.info("="*60)
        model.summary(print_fn=logger.info)
        logger.info("="*60 + "\n")
        
        return model
    
    def create_sequences(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Cria sequências temporais para treinamento do LSTM.
        
        Args:
            data: Array com features
            target: Array com valores alvo (opcional)
            
        Returns:
            Tupla (X_sequences, y_sequences) ou apenas X_sequences
        """
        X = []
        y = [] if target is not None else None
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            if target is not None:
                y.append(target[i + self.sequence_length])
        
        X = np.array(X)
        
        if y is not None:
            y = np.array(y)
            logger.info(f"Sequências criadas: X={X.shape}, y={y.shape}")
            return X, y
        else:
            logger.info(f"Sequências criadas: X={X.shape}")
            return X
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """
        Treina o modelo LSTM.
        
        Args:
            X_train: Sequências de treino
            y_train: Targets de treino
            X_val: Sequências de validação
            y_val: Targets de validação
            epochs: Número máximo de épocas
            batch_size: Tamanho do batch
            verbose: Nível de verbosidade
            
        Returns:
            Histórico do treinamento
        """
        if self.model is None:
            self.build_model()
        
        logger.info("\n" + "="*60)
        logger.info("INICIANDO TREINAMENTO")
        logger.info("="*60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info("="*60 + "\n")
        
        # Callbacks
        callbacks = [
            # Early stopping - para se não melhorar
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduz learning rate se estagnar
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Salva melhor modelo
            ModelCheckpoint(
                'models/lstm_best_checkpoint.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Treinar
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        
        logger.info("\n" + "="*60)
        logger.info("TREINAMENTO CONCLUÍDO")
        logger.info("="*60)
        logger.info(f"Melhor val_loss: {min(history.history['val_loss']):.6f}")
        logger.info(f"Melhor val_mae: {min(history.history['val_mae']):.6f}")
        logger.info(f"Épocas treinadas: {len(history.history['loss'])}")
        logger.info("="*60 + "\n")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz previsões com o modelo treinado.
        
        Args:
            X: Sequências de entrada
            
        Returns:
            Previsões
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Avalia o modelo no conjunto de teste.
        
        Args:
            X_test: Sequências de teste
            y_test: Targets de teste
            
        Returns:
            Dicionário com métricas
        """
        logger.info("Avaliando modelo no conjunto de teste...")
        
        # Previsões
        y_pred = self.predict(X_test)
        
        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MSE': float(mse),
            'R2': float(r2),
            'MAPE': float(mape)
        }
        
        logger.info("\n" + "="*60)
        logger.info("MÉTRICAS DE AVALIAÇÃO")
        logger.info("="*60)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.6f}")
        logger.info("="*60 + "\n")
        
        return metrics
    
    def save_model(
        self,
        filepath: Path,
        save_metadata: bool = True
    ) -> None:
        """
        Salva o modelo treinado.
        
        Args:
            filepath: Caminho para salvar
            save_metadata: Salvar metadados do modelo
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo
        self.model.save(filepath)
        logger.info(f"Modelo salvo em: {filepath}")
        
        # Salvar metadados
        if save_metadata:
            metadata = {
                'sequence_length': self.sequence_length,
                'n_features': self.n_features,
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'bidirectional': self.bidirectional,
                'trained_at': datetime.now().isoformat(),
                'history': self.history if self.history else {}
            }
            
            metadata_path = filepath.parent / 'lstm_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Metadados salvos em: {metadata_path}")
    
    def load_model(self, filepath: Path) -> None:
        """
        Carrega modelo salvo.
        
        Args:
            filepath: Caminho do modelo
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        logger.info(f"Modelo carregado de: {filepath}")
        
        # Carregar metadados se existirem
        metadata_path = filepath.parent / 'lstm_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.sequence_length = metadata['sequence_length']
            self.n_features = metadata['n_features']
            self.lstm_units = metadata['lstm_units']
            self.dropout_rate = metadata['dropout_rate']
            self.learning_rate = metadata['learning_rate']
            self.bidirectional = metadata['bidirectional']
            self.history = metadata.get('history', {})
            
            logger.info("Metadados carregados")


def main():
    """
    Função de teste do modelo.
    """
    logger.info("Teste do modelo LSTM")
    
    # Dados sintéticos para teste
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    sequence_length = 30  # Atualizado para novo padrão
    
    # Criar dados de teste
    data = np.random.randn(n_samples, n_features)
    
    # Criar modelo
    predictor = LSTMStockPredictor(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=[64, 32],
        dropout_rate=0.2,
        bidirectional=True
    )
    
    # Construir modelo
    predictor.build_model()
    
    logger.info("Modelo criado com sucesso!")


if __name__ == "__main__":
    main()
