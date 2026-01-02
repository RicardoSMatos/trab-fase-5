"""
Preditor para múltiplas ações da carteira.

Gera predições para todas as ações com modelos treinados.
"""

# IMPORTANTE: Importar yfinance ANTES de TensorFlow
# Ref: https://github.com/ranaroussi/yfinance/issues/2528
import os
os.environ['YF_FORCE_URLLIB'] = '1'
import yfinance as yf

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data.collect_data import StockDataCollector
from src.data.preprocessing import TimeSeriesPreprocessor
from src.models.lstm_model import LSTMStockPredictor
from src.portfolio.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class MultiStockPredictor:
    """
    Preditor para múltiplas ações.
    
    Carrega modelos treinados e gera predições
    para ações na carteira.
    """
    
    def __init__(self, portfolio_manager: PortfolioManager):
        """
        Inicializa o preditor.
        
        Args:
            portfolio_manager: Gerenciador de carteira
        """
        self.portfolio = portfolio_manager
        self._models_cache = {}
    
    def _load_model(self, symbol: str) -> Optional[LSTMStockPredictor]:
        """
        Carrega modelo de uma ação (com cache).
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            Modelo carregado ou None
        """
        if symbol in self._models_cache:
            return self._models_cache[symbol]
        
        stock_info = self.portfolio.get_stock(symbol)
        # Verificar se modelo está treinado usando status ou training_status
        is_trained = (
            stock_info.get('status') == 'trained' or 
            stock_info.get('training_status') == 'completed'
        )
        if not stock_info or not is_trained:
            logger.warning(f"Modelo não treinado para {symbol}")
            return None
        
        # Normalizar caminho (substituir barras invertidas por normais)
        model_path = stock_info.get('model_path')
        if model_path:
            model_path = str(model_path).replace('\\', '/')
        
        if not model_path or not Path(model_path).exists():
            logger.warning(f"Modelo não encontrado para {symbol}: {model_path}")
            return None
        
        try:
            # Obter metadados
            metadata_path = str(model_path).replace('_model.keras', '_metadata.json')
            
            model = LSTMStockPredictor(
                sequence_length=stock_info.get('sequence_length', 60),
                n_features=stock_info.get('n_features', 7)
            )
            model.load_model(model_path)
            
            self._models_cache[symbol] = model
            logger.info(f"Modelo carregado para {symbol}")
            return model
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de {symbol}: {str(e)}")
            return None
    
    def predict_single_stock(
        self,
        symbol: str,
        days_ahead: int = 5,
        start_date: str = None,
        end_date: str = None
    ) -> Optional[Dict]:
        """
        Gera predição para uma única ação.
        
        Args:
            symbol: Símbolo da ação
            days_ahead: Número de dias para prever
            start_date: Data inicial dos dados
            end_date: Data final dos dados
            
        Returns:
            Dicionário com predições e métricas
        """
        try:
            # Carregar modelo
            model = self._load_model(symbol)
            if model is None:
                return None
            
            stock_info = self.portfolio.get_stock(symbol)
            sequence_length = stock_info.get('sequence_length', 30)
            
            # Normalizar caminhos dos scalers
            scaler_path = stock_info.get('scaler_path')
            if scaler_path:
                scaler_path = str(scaler_path).replace('\\', '/')
            
            target_scaler_path = stock_info.get('target_scaler_path')
            if target_scaler_path:
                target_scaler_path = str(target_scaler_path).replace('\\', '/')
            
            # Verificar se temos os scalers salvos
            if not scaler_path or not Path(scaler_path).exists():
                logger.error(f"Feature scaler não encontrado para {symbol}. Retreine o modelo.")
                return None
            if not target_scaler_path or not Path(target_scaler_path).exists():
                logger.error(f"Target scaler não encontrado para {symbol}. Retreine o modelo.")
                return None
            
            # Coletar dados recentes (2 anos para garantir sequências suficientes)
            if not start_date:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            collector = StockDataCollector(symbol)
            df = collector.collect(start_date, end_date)
            
            if df is None or len(df) < sequence_length + 100:
                logger.error(f"Dados insuficientes para {symbol}: {len(df) if df is not None else 0} registros")
                return None
            
            # Pré-processar manualmente (pipeline completo - igual ao trainer)
            preprocessor = TimeSeriesPreprocessor()
            
            # 1. Criar features técnicas
            df_features = preprocessor.create_features(df)
            
            # 2. Carregar scalers do treino (CRÍTICO: usar os mesmos scalers!)
            import pickle
            with open(scaler_path, 'rb') as f:
                preprocessor.scaler = pickle.load(f)
            with open(target_scaler_path, 'rb') as f:
                target_scaler = pickle.load(f)
            logger.info(f"Scalers carregados: {scaler_path}, {target_scaler_path}")
            
            # 3. Preparar dados (split train/val/test) - retorna dicionário
            data_splits = preprocessor.prepare_data(df_features, target_col='Close')
            
            X_train = data_splits['X_train']
            X_test = data_splits['X_test']
            y_train = data_splits['y_train']
            y_test = data_splits['y_test']
            
            # 4. Normalizar dados com os scalers do treino (não fit novamente!)
            X_train_scaled = preprocessor.transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            
            # Normalizar target também
            y_train_scaled = target_scaler.transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
            
            # 4. Criar sequências LSTM
            from src.data.preprocessing import prepare_for_lstm
            
            # Empilhar features + target NORMALIZADO (mesma estrutura do treino)
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
            
            # Verificar se sequências foram criadas
            if X_test_seq.shape[0] == 0 or y_test_seq.shape[0] == 0:
                logger.error(f"Sequências vazias para {symbol}. Dados insuficientes após preprocessing.")
                return None
            
            # Separar features e target (mesma lógica do treino)
            X_test_final = X_test_seq[:, :, :-1]  # Remove Close das features
            y_test_final = y_test_seq[:, -1]      # Pega Close do timestep seguinte
            
            logger.info(f"Sequências criadas para {symbol}: X_test shape {X_test_final.shape}, "
                       f"y_test shape {y_test_final.shape}")
            logger.info(f"Close atual (último valor histórico): {df['Close'].iloc[-1]:.2f}")
            logger.info(f"y_test_final (valores reais): min={y_test_final.min():.2f}, max={y_test_final.max():.2f}, "
                       f"sample={y_test_final[:3]}")
            
            # Predições para dados de teste (normalizadas)
            test_predictions_norm = model.predict(X_test_final)
            
            # Desnormalizar predições e targets para escala real
            test_predictions = target_scaler.inverse_transform(test_predictions_norm.reshape(-1, 1)).flatten()
            y_test_real = target_scaler.inverse_transform(y_test_final.reshape(-1, 1)).flatten()
            
            logger.info(f"Test predictions: min={test_predictions.min():.2f}, max={test_predictions.max():.2f}, "
                       f"sample={test_predictions[:3]}")
            logger.info(f"y_test real: min={y_test_real.min():.2f}, max={y_test_real.max():.2f}")
            
            # Métricas de teste (usando valores REAIS, não normalizados!)
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_test_real, test_predictions)
            rmse = np.sqrt(mean_squared_error(y_test_real, test_predictions))
            mse = mean_squared_error(y_test_real, test_predictions)
            r2 = r2_score(y_test_real, test_predictions)
            mape = np.mean(np.abs((y_test_real - test_predictions) / y_test_real)) * 100
            
            test_metrics = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MSE': float(mse),
                'R2': float(r2),
                'MAPE': float(mape)
            }
            
            logger.info(f"Métricas (escala real): MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            
            # Converter NaN para None (JSON compliant)
            test_metrics_clean = {
                k: (None if np.isnan(v) else float(v))
                for k, v in test_metrics.items()
            }
            
            # Predições futuras - usar os ÚLTIMOS 30 dias mais recentes de TODO o dataset
            # Não usar apenas a última sequência de teste (que pode ser antiga)
            
            # Pegar últimos sequence_length registros de TODOS os dados (NORMALIZADOS)
            # Usar dados de treino já normalizados
            if len(X_train_scaled) >= sequence_length:
                recent_features = X_train_scaled[-sequence_length:]
                recent_target = y_train_scaled[-sequence_length:]
            else:
                # Se não houver dados suficientes, usar todo o dataset disponível
                all_X = df_features.drop(columns=['Close'])
                all_y = df_features['Close']
                
                recent_features = preprocessor.transform(all_X.tail(sequence_length))
                recent_target_raw = all_y.tail(sequence_length).values
                recent_target = target_scaler.transform(recent_target_raw.reshape(-1, 1)).flatten()
            
            # Criar sequência inicial com os dados mais recentes (TUDO NORMALIZADO)
            recent_data = np.column_stack([recent_features, recent_target.reshape(-1, 1)])
            
            # Sequência inicial: últimos sequence_length dias
            initial_sequence = recent_data[-sequence_length:].reshape(1, sequence_length, -1)
            
            # Desnormalizar último Close para log
            last_close_real = target_scaler.inverse_transform(recent_target[-1].reshape(-1, 1))[0, 0]
            logger.info(f"Sequência inicial para predições futuras: shape={initial_sequence.shape}, "
                       f"último Close={last_close_real:.2f} (normalizado: {recent_target[-1]:.4f})")
            
            future_predictions = []
            current_sequence = initial_sequence.copy()
            
            # Pegar features do último dia real para reutilizar
            last_real_features = current_sequence[0, -1, :-1].copy()
            
            for i in range(days_ahead):
                # Extrair features (sem Close) para predição
                current_features = current_sequence[:, :, :-1]
                
                # Fazer predição (retorna valor normalizado)
                pred_norm = model.predict(current_features)
                
                # Desnormalizar para escala real
                pred_real = target_scaler.inverse_transform(pred_norm.reshape(-1, 1))[0, 0]
                future_predictions.append(pred_real)
                
                # Atualizar sequência: MANTER features do último dia real + novo Close previsto
                # Isso evita erro composto nas features técnicas mas permite variação no Close
                new_timestep = np.append(last_real_features, pred_norm[0])
                
                # Deslocar sequência e adicionar novo timestep
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_timestep
            
            future_predictions_array = np.array(future_predictions)
            
            logger.info(f"Predições futuras (escala real) para {symbol}: {future_predictions_array}")
            logger.info(f"NOTA: Features técnicas mantidas constantes. Apenas Close é atualizado iterativamente.")
            
            # Preço atual
            current_price = float(df['Close'].iloc[-1])
            
            # Datas das predições
            last_date = pd.to_datetime(df.index[-1])
            prediction_dates = [
                (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                for i in range(days_ahead)
            ]
            
            # Criar resultado
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'prediction_date': datetime.now().isoformat(),
                'test_metrics': test_metrics_clean,
                'predictions': [
                    {
                        'date': date,
                        'predicted_price': float(price),
                        'change': float(price - current_price),
                        'change_pct': float((price - current_price) / current_price * 100)
                    }
                    for date, price in zip(prediction_dates, future_predictions_array)
                ],
                'historical': {
                    'dates': [d.strftime('%Y-%m-%d') for d in df.index[-30:]],
                    'prices': df['Close'].iloc[-30:].tolist()
                }
            }
            
            # Atualizar carteira
            self.portfolio.update_prediction(
                symbol,
                predictions=future_predictions_array.tolist(),
                confidence=test_metrics_clean.get('RMSE', 0.0) or 0.0
            )
            
            logger.info(f"Predições geradas para {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Erro ao prever {symbol}: {str(e)}", exc_info=True)
            return None
    
    def predict_multiple_stocks(
        self,
        symbols: List[str] = None,
        days_ahead: int = 5
    ) -> Dict[str, Dict]:
        """
        Gera predições para múltiplas ações.
        
        Args:
            symbols: Lista de símbolos (None = todas treinadas)
            days_ahead: Número de dias para prever
            
        Returns:
            Dicionário com predições por símbolo
        """
        if symbols is None:
            symbols = self.portfolio.get_trained_stocks()
        
        logger.info(f"Gerando predições para {len(symbols)} ações")
        
        predictions = {}
        successful = 0
        failed = 0
        
        for symbol in symbols:
            result = self.predict_single_stock(symbol, days_ahead=days_ahead)
            if result:
                predictions[symbol] = result
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Predições concluídas: {successful} sucesso, {failed} falhas")
        
        return {
            'predictions': predictions,
            'summary': {
                'total': len(symbols),
                'successful': successful,
                'failed': failed,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def get_portfolio_outlook(self, days_ahead: int = 5) -> Dict:
        """
        Gera visão geral da carteira com predições.
        
        Args:
            days_ahead: Número de dias para prever
            
        Returns:
            Dicionário com análise consolidada
        """
        predictions = self.predict_multiple_stocks(days_ahead=days_ahead)
        
        if not predictions['predictions']:
            return {
                'outlook': 'No predictions available',
                'total_stocks': 0,
                'predictions': {}
            }
        
        # Análise consolidada (sem considerar quantidade investida)
        positive_outlook = 0
        negative_outlook = 0
        avg_change_pct = 0
        
        for symbol, pred in predictions['predictions'].items():
            current_price = pred['current_price']
            last_prediction = pred['predictions'][-1]['predicted_price']
            change_pct = pred['predictions'][-1]['change_pct']
            
            avg_change_pct += change_pct
            
            if last_prediction > current_price:
                positive_outlook += 1
            else:
                negative_outlook += 1
        
        avg_change_pct = avg_change_pct / len(predictions['predictions']) if predictions['predictions'] else 0
        
        # Determinar outlook baseado na m\u00e9dia de mudan\u00e7a percentual
        outlook_sentiment = 'bullish' if avg_change_pct > 0 else 'bearish' if avg_change_pct < 0 else 'neutral'
        
        return {
            'outlook': outlook_sentiment,
            'total_stocks': len(predictions['predictions']),
            'positive_outlook': positive_outlook,
            'negative_outlook': negative_outlook,
            'avg_change_pct': avg_change_pct,
            'predictions': predictions['predictions'],
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Teste do preditor
    portfolio = PortfolioManager()
    predictor = MultiStockPredictor(portfolio)
    
    # Predições
    outlook = predictor.get_portfolio_outlook(days_ahead=5)
    
    print(f"\nVisão da carteira:")
    print(f"  Outlook: {outlook['outlook']}")
    print(f"  Total de ações: {outlook.get('total_stocks', 0)}")
    print(f"  Perspectiva positiva: {outlook.get('positive_outlook', 0)}")
    print(f"  Mudança média esperada: {outlook.get('avg_change_pct', 0):.2f}%")
