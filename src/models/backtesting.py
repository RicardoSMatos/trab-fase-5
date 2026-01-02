"""
Sistema de backtesting para validação histórica do modelo.

Este módulo implementa walk-forward validation para avaliar
a performance do modelo ao longo do tempo com dados históricos.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


class Backtester:
    """
    Classe para executar backtesting de modelos de séries temporais.
    
    Implementa walk-forward validation onde o modelo é treinado
    em uma janela móvel de dados históricos e testado em períodos futuros.
    """
    
    def __init__(self, model, preprocessor, data: pd.DataFrame):
        """
        Inicializa o backtester.
        
        Args:
            model: Modelo de ML a ser testado (LSTMStockPredictor)
            preprocessor: Preprocessador de dados (TimeSeriesPreprocessor)
            data: DataFrame com dados históricos completos
        """
        self.model = model
        self.preprocessor = preprocessor
        self.data = data.copy()
        self.results = []
        
        logger.info(f"Backtester inicializado com {len(data)} registros")
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        train_window: int = 365,
        prediction_horizon: int = 5,
        step_days: int = 5,
        lookback: int = 30
    ) -> pd.DataFrame:
        """
        Executa backtesting ao longo do tempo.
        
        Args:
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)
            train_window: Janela de treino em dias
            prediction_horizon: Dias à frente para prever
            step_days: Passo entre backtests
            lookback: Janela de lookback para sequências
            
        Returns:
            DataFrame com resultados do backtesting
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        logger.info(f"Iniciando backtesting de {start_date.date()} até {end_date.date()}")
        logger.info(f"Parâmetros: train_window={train_window}, horizon={prediction_horizon}, step={step_days}")
        
        self.results = []
        current_date = start_date
        iteration = 0
        skipped_reasons = []
        
        logger.info(f"Total de dados disponíveis: {len(self.data)} registros")
        logger.info(f"Período dos dados: {self.data.index.min().date()} até {self.data.index.max().date()}")
        
        while current_date <= end_date:
            try:
                iteration += 1
                
                # Dados de treino: train_window dias antes de current_date
                train_start = current_date - timedelta(days=train_window)
                train_data = self.data[
                    (self.data.index >= train_start) &
                    (self.data.index < current_date)
                ]
                
                # Dados de teste: prediction_horizon dias após current_date
                # IMPORTANTE: Precisamos incluir dados históricos para criar features (MA de 200 dias, etc.)
                test_start_with_history = current_date - timedelta(days=250)  # 250 dias de história para features
                test_end = current_date + timedelta(days=prediction_horizon)
                test_data_full = self.data[
                    (self.data.index >= test_start_with_history) &
                    (self.data.index < test_end)
                ]
                
                # Separar dados reais de teste (apenas para referência)
                test_data_actual = self.data[
                    (self.data.index >= current_date) &
                    (self.data.index < test_end)
                ]
                
                # Validar se temos dados suficientes
                if len(train_data) < lookback + 10:
                    reason = f"Treino insuficiente: {len(train_data)} registros (mínimo {lookback + 10})"
                    skipped_reasons.append(f"{current_date.date()}: {reason}")
                    logger.debug(f"{reason} em {current_date.date()}, pulando...")
                    current_date += timedelta(days=step_days)
                    continue
                    
                if len(test_data_actual) == 0:
                    reason = f"Sem dados de teste (test_end={test_end.date()} > data_max={self.data.index.max().date()})"
                    skipped_reasons.append(f"{current_date.date()}: {reason}")
                    logger.debug(f"{reason} em {current_date.date()}, pulando...")
                    current_date += timedelta(days=step_days)
                    continue
                    current_date += timedelta(days=step_days)
                    continue
                
                # Preparar dados de treino
                train_features = self.preprocessor.create_features(train_data)
                train_features_clean = train_features.dropna()
                
                # Verificar se sobrou dados após dropna
                if len(train_features_clean) < lookback + 50:  # Mínimo 50 registros após lookback
                    logger.debug(f"Dados insuficientes após feature engineering em {current_date.date()} (apenas {len(train_features_clean)} registros), pulando...")
                    current_date += timedelta(days=step_days)
                    continue
                
                # Ajustar scaler aos dados de treino
                feature_cols = [col for col in train_features_clean.columns if col != 'Close']
                
                # Verificar se temos features válidas
                if len(feature_cols) == 0:
                    logger.error(f"Nenhuma feature disponível em {current_date.date()}, pulando...")
                    current_date += timedelta(days=step_days)
                    continue
                
                self.preprocessor.fit_scaler(train_features_clean[feature_cols])
                
                # Normalizar dados
                train_normalized = self.preprocessor.transform(train_features_clean[feature_cols])
                
                # Criar sequências para LSTM (importar função diretamente do módulo)
                X_train, y_train = [], []
                for i in range(len(train_normalized) - lookback):
                    X_train.append(train_normalized[i:i + lookback])
                    y_train.append(train_normalized[i + lookback, 0])  # Predizer primeira feature (Close)
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Treinar modelo
                logger.debug(f"[{iteration}] Treinando modelo em {current_date.date()}")
                
                # Atualizar configurações do modelo para este fold
                self.model.sequence_length = lookback
                self.model.n_features = X_train.shape[2]
                self.model.build_model()
                
                history = self.model.model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[
                        # Early stopping para não overfit
                        EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True,
                            verbose=0
                        )
                    ]
                )
                
                # Preparar dados de teste
                # Usar test_data_full (com histórico) para criar features
                test_features = self.preprocessor.create_features(test_data_full)
                
                # IMPORTANTE: NÃO fazer dropna ainda! Filtrar primeiro o período de teste
                # Filtrar apenas o período real de teste (após current_date)
                test_features_period = test_features[test_features.index >= current_date]
                
                # AGORA fazer dropna apenas no período de teste
                test_features_clean = test_features_period.dropna()
                
                # Verificar se sobrou dados de teste após dropna
                if len(test_features_clean) == 0:
                    reason = "Nenhum dado de teste válido após feature engineering e filtro de data"
                    skipped_reasons.append(f"{current_date.date()}: {reason}")
                    logger.debug(f"{reason} em {current_date.date()}, pulando...")
                    current_date += timedelta(days=step_days)
                    continue
                
                logger.info(f"[{iteration}] {current_date.date()}: Predizendo com {len(test_features_clean)} dias de teste...")
                
                # Normalizar dados de teste usando o scaler já ajustado
                test_normalized = self.preprocessor.transform(test_features_clean[feature_cols])
                
                # Criar sequências de teste
                # Combinar últimos lookback dias do treino com dados de teste
                combined_normalized = np.vstack([train_normalized[-lookback:], test_normalized])
                
                X_test, y_test = [], []
                for i in range(len(combined_normalized) - lookback):
                    X_test.append(combined_normalized[i:i + lookback])
                    y_test.append(combined_normalized[i + lookback, 0])
                
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                
                # Fazer predições
                if len(X_test) > 0:
                    logger.info(f"[{iteration}] Fazendo predições com {len(X_test)} sequências...")
                    predictions_scaled = self.model.model.predict(X_test, verbose=0)
                    
                    logger.info(f"[{iteration}] Desnormalizando {len(predictions_scaled)} predições...")
                    # Desnormalizar predições (apenas a primeira feature que é Close)
                    # Criar array com shape correto para inverse_transform
                    dummy_features = np.zeros((len(predictions_scaled), len(feature_cols)))
                    dummy_features[:, 0] = predictions_scaled.flatten()  # Assumindo Close é a primeira feature
                    predictions_denorm = self.preprocessor.inverse_transform(dummy_features)
                    predictions = predictions_denorm[:, 0]  # Pegar apenas a coluna Close
                    
                    actual_values = test_features_clean['Close'].values[:len(predictions)]
                    
                    logger.info(f"[{iteration}] Calculando métricas para {len(actual_values)} valores...")
                    # Calcular métricas
                    if len(actual_values) > 0 and len(predictions) > 0:
                        mae = mean_absolute_error(actual_values, predictions)
                        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
                        mape = mean_absolute_percentage_error(actual_values, predictions) * 100
                        
                        # Calcular acurácia direcional (previu corretamente subida/descida?)
                        if len(actual_values) > 1:
                            actual_direction = np.sign(np.diff(actual_values))
                            pred_direction = np.sign(np.diff(predictions))
                            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                        else:
                            directional_accuracy = 0.0
                        
                        self.results.append({
                            'date': current_date,
                            'train_size': len(train_data),
                            'test_size': len(actual_values),
                            'mae': mae,
                            'rmse': rmse,
                            'mape': mape,
                            'directional_accuracy': directional_accuracy,
                            'prediction_first': predictions[0] if len(predictions) > 0 else None,
                            'actual_first': actual_values[0] if len(actual_values) > 0 else None,
                            'predictions': predictions.tolist(),
                            'actual': actual_values.tolist()
                        })
                        
                        logger.info(
                            f"[{iteration}] {current_date.date()}: "
                            f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, "
                            f"Dir_Acc={directional_accuracy:.1f}%"
                        )
                
            except Exception as e:
                logger.error(f"Erro no backtest em {current_date.date()}: {e}")
                skipped_reasons.append(f"{current_date.date()}: ERRO - {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
            
            current_date += timedelta(days=step_days)
        
        logger.info(f"Backtesting concluído: {len(self.results)} iterações")
        
        # Se não houve resultados, mostrar razões
        if len(self.results) == 0:
            logger.warning("⚠️  Nenhuma iteração válida foi executada!")
            logger.warning(f"Total de tentativas puladas: {len(skipped_reasons)}")
            if len(skipped_reasons) > 0:
                logger.warning("Primeiras 5 razões:")
                for reason in skipped_reasons[:5]:
                    logger.warning(f"  - {reason}")
        
        return pd.DataFrame(self.results)
    
    def get_summary(self) -> Dict:
        """
        Retorna resumo estatístico dos resultados.
        
        Returns:
            Dict com estatísticas do backtesting
        """
        if not self.results:
            raise ValueError("Execute run_backtest() primeiro")
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'total_iterations': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'metrics': {
                'mae': {
                    'mean': float(df['mae'].mean()),
                    'std': float(df['mae'].std()),
                    'min': float(df['mae'].min()),
                    'max': float(df['mae'].max()),
                    'median': float(df['mae'].median())
                },
                'rmse': {
                    'mean': float(df['rmse'].mean()),
                    'std': float(df['rmse'].std()),
                    'min': float(df['rmse'].min()),
                    'max': float(df['rmse'].max()),
                    'median': float(df['rmse'].median())
                },
                'mape': {
                    'mean': float(df['mape'].mean()),
                    'std': float(df['mape'].std()),
                    'min': float(df['mape'].min()),
                    'max': float(df['mape'].max()),
                    'median': float(df['mape'].median())
                },
                'directional_accuracy': {
                    'mean': float(df['directional_accuracy'].mean()),
                    'std': float(df['directional_accuracy'].std()),
                    'min': float(df['directional_accuracy'].min()),
                    'max': float(df['directional_accuracy'].max())
                }
            },
            'total_predictions': int(df['test_size'].sum())
        }
        
        return summary
    
    def save_results(self, filepath: str = "logs/backtest_results.json"):
        """
        Salva resultados do backtesting em arquivo JSON.
        
        Args:
            filepath: Caminho do arquivo para salvar
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        # Converter para formato serializável
        results_dict = {
            'summary': self.get_summary(),
            'detailed_results': [
                {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'train_size': int(row['train_size']),
                    'test_size': int(row['test_size']),
                    'mae': float(row['mae']),
                    'rmse': float(row['rmse']),
                    'mape': float(row['mape']),
                    'directional_accuracy': float(row['directional_accuracy']),
                    'prediction_first': float(row['prediction_first']) if row['prediction_first'] else None,
                    'actual_first': float(row['actual_first']) if row['actual_first'] else None
                }
                for _, row in df.iterrows()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Resultados salvos em {filepath}")
    
    def visualize_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Cria visualizações dos resultados do backtesting.
        
        Args:
            save_path: Caminho para salvar a figura (opcional)
            
        Returns:
            Figure matplotlib
        """
        if not self.results:
            raise ValueError("Execute run_backtest() primeiro")
        
        df = pd.DataFrame(self.results)
        
        # Criar figura com 6 subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. MAE ao longo do tempo
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(df['date'], df['mae'], marker='o', linestyle='-', alpha=0.7)
        ax1.axhline(y=df['mae'].mean(), color='r', linestyle='--', 
                   label=f'Média: {df["mae"].mean():.2f}')
        ax1.set_xlabel('Data')
        ax1.set_ylabel('MAE (R$)')
        ax1.set_title('Mean Absolute Error ao Longo do Tempo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. RMSE ao longo do tempo
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(df['date'], df['rmse'], marker='o', linestyle='-', 
                alpha=0.7, color='orange')
        ax2.axhline(y=df['rmse'].mean(), color='r', linestyle='--',
                   label=f'Média: {df["rmse"].mean():.2f}')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('RMSE (R$)')
        ax2.set_title('Root Mean Square Error ao Longo do Tempo')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. MAPE ao longo do tempo
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(df['date'], df['mape'], marker='o', linestyle='-', 
                alpha=0.7, color='green')
        ax3.axhline(y=df['mape'].mean(), color='r', linestyle='--',
                   label=f'Média: {df["mape"].mean():.2f}%')
        ax3.set_xlabel('Data')
        ax3.set_ylabel('MAPE (%)')
        ax3.set_title('Mean Absolute Percentage Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Predições vs Valores Reais (scatter)
        ax4 = plt.subplot(2, 3, 4)
        all_predictions = []
        all_actual = []
        for _, row in df.iterrows():
            all_predictions.extend(row['predictions'])
            all_actual.extend(row['actual'])
        
        ax4.scatter(all_actual, all_predictions, alpha=0.5)
        
        # Linha de predição perfeita
        min_val = min(min(all_actual), min(all_predictions))
        max_val = max(max(all_actual), max(all_predictions))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Predição Perfeita')
        
        ax4.set_xlabel('Valor Real (R$)')
        ax4.set_ylabel('Predição (R$)')
        ax4.set_title('Predições vs Valores Reais')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Distribuição de erros
        ax5 = plt.subplot(2, 3, 5)
        errors = np.array(all_predictions) - np.array(all_actual)
        ax5.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.set_xlabel('Erro (R$)')
        ax5.set_ylabel('Frequência')
        ax5.set_title('Distribuição de Erros de Predição')
        ax5.grid(True, alpha=0.3)
        
        # 6. Acurácia direcional
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(df['date'], df['directional_accuracy'], marker='o', 
                linestyle='-', alpha=0.7, color='purple')
        ax6.axhline(y=50, color='gray', linestyle='--', 
                   label='Random (50%)')
        ax6.axhline(y=df['directional_accuracy'].mean(), color='r', 
                   linestyle='--', label=f'Média: {df["directional_accuracy"].mean():.1f}%')
        ax6.set_xlabel('Data')
        ax6.set_ylabel('Acurácia (%)')
        ax6.set_title('Acurácia Direcional (Previu Subida/Descida)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
        ax6.set_ylim([0, 100])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em {save_path}")
        
        return fig
    
    def get_best_worst_predictions(self, n: int = 5) -> Dict:
        """
        Retorna as melhores e piores predições.
        
        Args:
            n: Número de predições a retornar
            
        Returns:
            Dict com melhores e piores predições
        """
        if not self.results:
            raise ValueError("Execute run_backtest() primeiro")
        
        df = pd.DataFrame(self.results)
        
        # Ordenar por MAE
        df_sorted_best = df.nsmallest(n, 'mae')
        df_sorted_worst = df.nlargest(n, 'mae')
        
        return {
            'best_predictions': [
                {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'mae': float(row['mae']),
                    'prediction': float(row['prediction_first']),
                    'actual': float(row['actual_first'])
                }
                for _, row in df_sorted_best.iterrows()
            ],
            'worst_predictions': [
                {
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'mae': float(row['mae']),
                    'prediction': float(row['prediction_first']),
                    'actual': float(row['actual_first'])
                }
                for _, row in df_sorted_worst.iterrows()
            ]
        }


if __name__ == "__main__":
    """
    Exemplo de uso do Backtester.
    """
    import sys
    from pathlib import Path
    
    # Adicionar src ao path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.preprocessing import TimeSeriesPreprocessor
    from models.lstm_model import LSTMStockPredictor
    import pandas as pd
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Exemplo de uso do Backtester")
    print("=" * 60)
    
    # Carregar dados
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "stock_data_processed.csv"
    
    if not data_path.exists():
        print(f"Arquivo não encontrado: {data_path}")
        print("Execute primeiro o preprocessamento dos dados")
        sys.exit(1)
    
    data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    print(f"Dados carregados: {len(data)} registros")
    print(f"Período: {data.index.min().date()} até {data.index.max().date()}")
    
    # Verificar se há dados suficientes
    if len(data) < 500:
        print(f"\n⚠️  AVISO: Apenas {len(data)} registros disponíveis")
        print("Para backtesting robusto, recomenda-se pelo menos 500 registros")
    
    # Calcular período de teste (últimos 30% dos dados)
    total_days = (data.index.max() - data.index.min()).days
    test_days = int(total_days * 0.3)
    test_start = data.index.max() - pd.Timedelta(days=test_days)
    
    # Garantir que há dados suficientes para treino (pelo menos 18 meses antes para acomodar features de 200 dias)
    train_start = test_start - pd.Timedelta(days=540)  # 18 meses
    if train_start < data.index.min():
        train_start = data.index.min() + pd.Timedelta(days=250)  # Mínimo de 250 dias para criar features de 200 dias
        test_start = train_start + pd.Timedelta(days=540)
    
    print(f"\nPeríodo de backtesting:")
    print(f"  Início: {test_start.date()}")
    print(f"  Fim: {data.index.max().date()}")
    print(f"  Duração: ~{test_days} dias")
    
    # Inicializar componentes
    preprocessor = TimeSeriesPreprocessor()
    model = LSTMStockPredictor()
    
    # Criar backtester
    backtester = Backtester(model, preprocessor, data)
    
    # Executar backtesting
    print("\nExecutando backtesting...")
    print("⏳ Isso pode levar alguns minutos...")
    
    try:
        results_df = backtester.run_backtest(
            start_date=test_start.strftime('%Y-%m-%d'),
            end_date=data.index.max().strftime('%Y-%m-%d'),
            train_window=540,  # 18 meses para acomodar features longas
            prediction_horizon=5,
            step_days=60,  # A cada 2 meses (menos iterações para economizar tempo)
            lookback=30
        )
    except Exception as e:
        print(f"\n❌ Erro durante backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verificar se há resultados
    if results_df is None or len(results_df) == 0:
        print("\n❌ Nenhum resultado gerado. Verifique os parâmetros e dados.")
        sys.exit(1)
    
    print(f"\n✅ Backtesting concluído: {len(results_df)} iterações")
    
    # Exibir sumário
    print("\n" + "=" * 60)
    print("SUMÁRIO DO BACKTESTING")
    print("=" * 60)
    
    try:
        summary = backtester.get_summary()
    except ValueError as e:
        print(f"\n❌ Erro ao obter sumário: {str(e)}")
        print("Os resultados não foram salvos corretamente durante o backtesting")
        sys.exit(1)
    
    print(f"\nPeríodo: {summary['date_range']['start']} até {summary['date_range']['end']}")
    print(f"Total de iterações: {summary['total_iterations']}")
    print(f"Total de predições: {summary['total_predictions']}")
    
    print(f"\nMAE:")
    print(f"  Média: R$ {summary['metrics']['mae']['mean']:.2f}")
    print(f"  Desvio: R$ {summary['metrics']['mae']['std']:.2f}")
    print(f"  Min/Max: R$ {summary['metrics']['mae']['min']:.2f} / R$ {summary['metrics']['mae']['max']:.2f}")
    
    print(f"\nMAPE:")
    print(f"  Média: {summary['metrics']['mape']['mean']:.2f}%")
    
    print(f"\nAcurácia Direcional:")
    print(f"  Média: {summary['metrics']['directional_accuracy']['mean']:.1f}%")
    
    # Salvar resultados
    backtester.save_results()
    
    # Criar visualizações
    print("\nGerando visualizações...")
    backtester.visualize_results(save_path='logs/backtest_visualization.png')
    
    print("\n✅ Backtesting concluído!")
    print(f"Resultados salvos em: logs/backtest_results.json")
    print(f"Visualização salva em: logs/backtest_visualization.png")
