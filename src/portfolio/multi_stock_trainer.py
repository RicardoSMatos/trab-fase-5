"""
Orquestrador de treinamento de modelos para múltiplas ações.

Gerencia o processo de treinamento paralelo/sequencial de modelos LSTM
para todas as ações na carteira.
"""

# IMPORTANTE: Importar yfinance ANTES de TensorFlow
# Ref: https://github.com/ranaroussi/yfinance/issues/2528
import os
os.environ['YF_FORCE_URLLIB'] = '1'
import yfinance as yf

import logging
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data.collect_data import StockDataCollector
from src.data.preprocessing import TimeSeriesPreprocessor
from src.models.lstm_model import LSTMStockPredictor
from src.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class MultiStockTrainer:
    """
    Orquestrador de treinamento para múltiplas ações.
    
    Coleta dados, pré-processa e treina modelos LSTM
    para cada ação na carteira.
    """
    
    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        models_dir: str = 'models',
        data_dir: str = 'data'
    ):
        """
        Inicializa o orquestrador.
        
        Args:
            portfolio_manager: Gerenciador de carteira
            models_dir: Diretório para salvar modelos
            data_dir: Diretório para salvar dados
        """
        self.portfolio = portfolio_manager
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def train_single_stock(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        sequence_length: int = 30,
        epochs: int = 150,
        batch_size: int = 16
    ) -> Dict:
        """
        Treina modelo para uma única ação.
        
        Args:
            symbol: Símbolo da ação
            start_date: Data inicial (None = 5 anos atrás)
            end_date: Data final (None = hoje)
            sequence_length: Janela de tempo
            epochs: Épocas de treinamento
            batch_size: Tamanho do batch
            
        Returns:
            Dicionário com resultados do treinamento
        """
        try:
            logger.info(f"Iniciando treinamento para {symbol}")
            self.portfolio.update_training_status(symbol, 'training')
                        # Limpar artefatos antigos para garantir consistência
            old_model = self.models_dir / f'{symbol}_model.keras'
            old_scaler = self.models_dir / f'{symbol}_scaler.pkl'
            old_target_scaler = self.models_dir / f'{symbol}_target_scaler.pkl'
            
            for old_file in [old_model, old_scaler, old_target_scaler]:
                if old_file.exists():
                    old_file.unlink()
                    logger.info(f"Removido artefato antigo: {old_file.name}")
                        # 1. Coletar dados
            if not start_date:
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"{symbol}: Coletando dados de {start_date} a {end_date}")
            
            collector = StockDataCollector(symbol)
            df = collector.collect(start_date, end_date)
            
            if df is None:
                error = f"Não foi possível coletar dados para {symbol}. Verifique se o símbolo está correto."
                logger.error(error)
                self.portfolio.update_training_status(symbol, 'failed', error)
                return {'success': False, 'error': error, 'symbol': symbol}
            
            if len(df) < 200:
                error = f"Dados insuficientes para {symbol}: {len(df)} dias (mínimo 200 dias)"
                logger.error(error)
                self.portfolio.update_training_status(symbol, 'failed', error)
                return {'success': False, 'error': error, 'symbol': symbol}
            
            logger.info(f"{symbol}: {len(df)} dias de dados coletados")
            
            # Salvar dados brutos
            raw_path = self.data_dir / f'{symbol}_raw.csv'
            df.to_csv(raw_path)
            
            # 2. Pré-processar dados
            preprocessor = TimeSeriesPreprocessor()
            
            # Criar features
            df_features = preprocessor.create_features(df)
            
            # Preparar dados (split train/val/test)
            data_splits = preprocessor.prepare_data(df_features, target_col='Close')
            
            X_train = data_splits['X_train']
            X_val = data_splits['X_val']
            X_test = data_splits['X_test']
            y_train = data_splits['y_train']
            y_val = data_splits['y_val']
            y_test = data_splits['y_test']
            
            # Normalizar dados (features E target)
            preprocessor.fit_scaler(X_train)
            X_train_scaled = preprocessor.transform(X_train)
            X_val_scaled = preprocessor.transform(X_val)
            X_test_scaled = preprocessor.transform(X_test)
            
            # Normalizar target também (CRÍTICO!)
            from sklearn.preprocessing import MinMaxScaler
            target_scaler = MinMaxScaler()
            y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
            y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
            
            logger.info(f"{symbol}: Target normalizado - y_train range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
            
            # Criar sequências LSTM
            from src.data.preprocessing import prepare_for_lstm
            
            # Empilhar features normalizadas + target NORMALIZADO
            # Agora tudo está na mesma escala [0-1]
            train_data = np.column_stack([X_train_scaled, y_train_scaled.reshape(-1, 1)])
            test_data = np.column_stack([X_test_scaled, y_test_scaled.reshape(-1, 1)])
            
            X_train_seq, y_train_seq = prepare_for_lstm(
                train_data,
                sequence_length=sequence_length
            )
            X_test_seq, y_test_seq = prepare_for_lstm(
                test_data,
                sequence_length=sequence_length
            )
            
            # Separar features e target
            # X: todas as features exceto a última (Close) em todos os timesteps
            # y: Close NORMALIZADO (última coluna) do último timestep da sequência de saída
            X_train_final = X_train_seq[:, :, :-1]  # (samples, timesteps, features-1)
            y_train_final = y_train_seq[:, -1]      # (samples,) - Close NORMALIZADO [0-1]
            X_test_final = X_test_seq[:, :, :-1]
            y_test_final = y_test_seq[:, -1]
            
            logger.info(f"{symbol}: Shapes - X_train: {X_train_final.shape}, y_train: {y_train_final.shape}")
            logger.info(f"{symbol}: y_train_final range (normalizado): [{y_train_final.min():.4f}, {y_train_final.max():.4f}]")
            
            if X_train_final is None or len(X_train_final) == 0:
                error = f"Falha no pré-processamento de {symbol}"
                logger.error(error)
                self.portfolio.update_training_status(symbol, 'failed', error)
                return {'success': False, 'error': error, 'symbol': symbol}
            
            logger.info(f"{symbol}: Dados processados - Train: {len(X_train_final)}, Test: {len(X_test_final)}")
            
            # Salvar dados processados
            processed_path = self.data_dir / f'{symbol}_processed.csv'
            df_features.to_csv(processed_path)
            
            # 3. Treinar modelo
            model = LSTMStockPredictor(
                sequence_length=sequence_length,
                n_features=X_train_final.shape[2]
            )
            
            model.build_model()
            history = model.train(
                X_train_final, y_train_final,
                X_test_final, y_test_final,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # 4. Avaliar modelo com métricas em escala REAL
            # Fazer predições normalizadas
            y_pred_norm = model.model.predict(X_test_final, verbose=0).flatten()
            
            # Desnormalizar predições e targets
            y_pred_real = target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
            y_test_real = target_scaler.inverse_transform(y_test_final.reshape(-1, 1)).flatten()
            
            # Calcular métricas em escala real
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_test_real, y_pred_real)
            rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
            mse = mean_squared_error(y_test_real, y_pred_real)
            r2 = r2_score(y_test_real, y_pred_real)
            mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100
            
            metrics = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MSE': float(mse),
                'R2': float(r2),
                'MAPE': float(mape)
            }
            
            logger.info(f"{symbol}: Métricas (escala real) - MAE: R$ {metrics['MAE']:.2f}, RMSE: R$ {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")
            
            # 5. Salvar modelo, scalers e metadados
            model_path = self.models_dir / f'{symbol}_model.keras'
            scaler_path = self.models_dir / f'{symbol}_scaler.pkl'
            target_scaler_path = self.models_dir / f'{symbol}_target_scaler.pkl'
            metadata_path = self.models_dir / f'{symbol}_metadata.json'
            
            model.save_model(str(model_path), save_metadata=True)
            
            # Salvar scalers para reutilizar na predição
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(preprocessor.scaler, f)
            with open(target_scaler_path, 'wb') as f:
                pickle.dump(target_scaler, f)
            logger.info(f"Scalers salvos: {scaler_path}, {target_scaler_path}")
            
            # 6. Atualizar carteira com informações do modelo
            stock = self.portfolio.portfolio["stocks"][symbol]
            stock.update({
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "target_scaler_path": str(target_scaler_path),
                "last_trained": datetime.now().isoformat(),
                "metrics": metrics,
                "sequence_length": sequence_length,
                "n_features": X_train_final.shape[2],
                "status": "trained"
            })
            self.portfolio._save_portfolio()
            
            self.portfolio.update_training_status(symbol, 'completed')
            
            logger.info(f"{symbol}: Treinamento concluído com sucesso")
            
            return {
                'success': True,
                'symbol': symbol,
                'metrics': metrics,
                'model_path': str(model_path),
                'train_samples': len(X_train_final),
                'test_samples': len(X_test_final)
            }
            
        except Exception as e:
            error = f"Erro no treinamento de {symbol}: {str(e)}"
            logger.error(error, exc_info=True)
            self.portfolio.update_training_status(symbol, 'failed', error)
            return {'success': False, 'error': error, 'symbol': symbol}
    
    def train_multiple_stocks(
        self,
        symbols: List[str] = None,
        parallel: bool = False,
        max_workers: int = 3,
        **training_params
    ) -> Dict[str, Dict]:
        """
        Treina modelos para múltiplas ações.
        
        Args:
            symbols: Lista de símbolos (None = todas na carteira)
            parallel: Se True, treina em paralelo
            max_workers: Número máximo de threads paralelas
            **training_params: Parâmetros de treinamento
            
        Returns:
            Dicionário com resultados por símbolo
        """
        if symbols is None:
            symbols = [stock['symbol'] for stock in self.portfolio.list_stocks()]
        
        logger.info(f"Iniciando treinamento de {len(symbols)} ações")
        logger.info(f"Modo: {'paralelo' if parallel else 'sequencial'}")
        
        results = {}
        
        if parallel:
            # Treinamento paralelo
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(
                        self.train_single_stock,
                        symbol,
                        **training_params
                    ): symbol
                    for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                        logger.info(f"Concluído: {symbol}")
                    except Exception as e:
                        logger.error(f"Erro em {symbol}: {str(e)}")
                        results[symbol] = {
                            'success': False,
                            'error': str(e),
                            'symbol': symbol
                        }
        else:
            # Treinamento sequencial
            for symbol in symbols:
                result = self.train_single_stock(symbol, **training_params)
                results[symbol] = result
        
        # Resumo
        successful = sum(1 for r in results.values() if r.get('success', False))
        failed = len(results) - successful
        
        logger.info(f"Treinamento finalizado: {successful} sucesso, {failed} falhas")
        
        return {
            'results': results,
            'summary': {
                'total': len(results),
                'successful': successful,
                'failed': failed
            }
        }
    
    def retrain_failed(self, **training_params) -> Dict[str, Dict]:
        """
        Re-treina modelos que falharam.
        
        Args:
            **training_params: Parâmetros de treinamento
            
        Returns:
            Dicionário com resultados
        """
        failed_symbols = [
            symbol for symbol, info in self.portfolio.portfolio['stocks'].items()
            if info.get('training_status') == 'failed'
        ]
        
        if not failed_symbols:
            logger.info("Nenhum modelo falho para re-treinar")
            return {'results': {}, 'summary': {'total': 0, 'successful': 0, 'failed': 0}}
        
        logger.info(f"Re-treinando {len(failed_symbols)} modelos falhos")
        return self.train_multiple_stocks(failed_symbols, **training_params)


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Teste do orquestrador
    portfolio = PortfolioManager()
    trainer = MultiStockTrainer(portfolio)
    
    # Treinar uma ação
    result = trainer.train_single_stock(
        'PETR4.SA',
        epochs=150,
        batch_size=16
    )
    
    print(f"\nResultado do treinamento:")
    print(f"  Sucesso: {result['success']}")
    if result['success']:
        print(f"  Métricas: {result['metrics']}")
