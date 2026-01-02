"""
API FastAPI para o modelo LSTM de previsão de preços de ações.

Este módulo implementa uma API REST para servir o modelo LSTM treinado,
permitindo fazer previsões, obter informações do modelo e verificar o status.
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# IMPORTANTE: Importar yfinance ANTES de TensorFlow para evitar erro
# "Impersonating chrome136 is not supported"
# Ref: https://github.com/ranaroussi/yfinance/issues/2528
import os
os.environ['YF_FORCE_URLLIB'] = '1'
import yfinance as yf  # Deve ser importado antes de qualquer módulo que use TensorFlow

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional
import json
from contextlib import asynccontextmanager

# Agora podemos importar módulos que usam TensorFlow
from models.lstm_model import LSTMStockPredictor
from data.preprocessing import TimeSeriesPreprocessor
from data.collect_data import StockDataCollector
from monitoring.metrics import APIMetrics, ResponseTimer
from monitoring.logging_config import setup_logging
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse
)

# Importar novos módulos
try:
    from models.backtesting import Backtester
    from models.explainability import ModelExplainer, create_feature_names
    from portfolio.portfolio_manager import PortfolioManager
    from portfolio.multi_stock_trainer import MultiStockTrainer
    from portfolio.multi_stock_predictor import MultiStockPredictor
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    PortfolioManager = None
    MultiStockTrainer = None
    MultiStockPredictor = None
    import logging
    logging.warning(f"Módulos avançados não disponíveis: {e}")

# Configurar logging avançado
setup_logging(log_level="INFO", log_to_file=True)
logger = logging.getLogger(__name__)

# Variáveis globais para modelo e preprocessador
model: Optional[LSTMStockPredictor] = None
preprocessor: Optional[TimeSeriesPreprocessor] = None
data_collector: Optional[StockDataCollector] = None
model_metadata: Dict[str, Any] = {}
api_metrics: Optional[APIMetrics] = None
portfolio_manager: Optional[PortfolioManager] = None
multi_trainer: Optional[MultiStockTrainer] = None
multi_predictor: Optional[MultiStockPredictor] = None

# Caminhos dos arquivos
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
METADATA_PATH = BASE_DIR / "models" / "lstm_metadata.json"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação (startup e shutdown).
    """
    global model, preprocessor, data_collector, model_metadata, api_metrics
    global portfolio_manager, multi_trainer, multi_predictor
    
    # Startup
    try:
        logger.info("Iniciando API...")
        
        # Inicializar sistema de métricas
        api_metrics = APIMetrics()
        logger.info("Sistema de métricas inicializado")
        
        # Inicializar gerenciador de carteira
        if ADVANCED_FEATURES_AVAILABLE:
            portfolio_manager = PortfolioManager()
            multi_trainer = MultiStockTrainer(portfolio_manager)
            multi_predictor = MultiStockPredictor(portfolio_manager)
            logger.info("Sistema de carteira inicializado")
        
        # Carregar metadata
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            logger.info("Metadata carregada com sucesso")
        
        # Carregar modelo
        logger.info(f"Carregando modelo de {MODEL_PATH}...")
        model = LSTMStockPredictor()
        model.load_model(MODEL_PATH)
        logger.info("Modelo carregado com sucesso")
        
        # Carregar preprocessador
        preprocessor = TimeSeriesPreprocessor()
        if SCALER_PATH.exists():
            logger.info(f"Carregando scaler de {SCALER_PATH}...")
            preprocessor.load_scaler(SCALER_PATH)
            logger.info("Scaler carregado com sucesso")
        else:
            logger.warning(f"Scaler não encontrado em {SCALER_PATH}. Usando scaler padrão.")
            logger.warning("Para melhor desempenho, treine o modelo novamente para gerar o scaler.")
        
        # Inicializar coletor de dados
        data_collector = StockDataCollector(ticker="PETR4.SA")
        logger.info("Coletor de dados inicializado")
        
        logger.info("API iniciada com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise
    
    yield  # Aplicação está rodando
    
    # Shutdown
    logger.info("Encerrando API...")
    if api_metrics:
        api_metrics.save_to_file()
        logger.info("Métricas salvas com sucesso")


# Criar aplicação FastAPI com lifespan
app = FastAPI(
    title="LSTM Stock Predictor API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint raiz da API.
    """
    return {
        "message": "LSTM Stock Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Verifica o status de saúde da API.
    
    Returns:
        HealthResponse: Status da API e informações do modelo
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Retorna métricas de uso da API.
    
    Returns:
        Dict com estatísticas de uso, performance e erros
    """
    if api_metrics is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sistema de métricas não inicializado"
        )
    
    return api_metrics.get_summary()



@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Retorna informações sobre o modelo carregado.
    
    Returns:
        ModelInfo: Informações detalhadas do modelo
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado"
        )
    
    return ModelInfo(
        model_name="LSTM Stock Predictor",
        version=model_metadata.get("version", "1.0.0"),
        training_date=model_metadata.get("training_date", "unknown"),
        metrics=model_metadata.get("metrics", {}),
        architecture=model_metadata.get("architecture", {})
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Faz previsões de preços para os próximos N dias.
    
    Args:
        request: Requisição com número de dias para prever
        
    Returns:
        PredictionResponse: Preços previstos e informações adicionais
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado"
        )
    
    try:
        logger.info(f"Fazendo previsão para {request.days} dias...")
        
        # Coletar dados recentes
        logger.info("Coletando dados recentes...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 anos de histórico (garante dados após features)
        
        # Criar novo coletor com datas específicas
        recent_collector = StockDataCollector(
            ticker="PETR4.SA",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        df = recent_collector.download_data()
        logger.info(f"Dados coletados: {len(df)} registros")
        
        if df.empty:
            raise ValueError("Não foi possível coletar dados")
        
        # Preprocessar dados
        logger.info("Preprocessando dados...")
        df_features = preprocessor.create_features(df)
        logger.info(f"Features criadas: {len(df_features)} registros")
        df_clean = df_features.dropna()
        logger.info(f"Após limpeza: {len(df_clean)} registros")
        
        # Normalizar
        X = df_clean.drop('Close', axis=1)
        
        # Se o scaler não foi carregado, treinar com os dados atuais
        if not hasattr(preprocessor.scaler, 'n_features_in_'):
            logger.warning("Scaler não foi treinado previamente. Treinando com dados atuais...")
            preprocessor.fit(X)
        
        X_scaled = preprocessor.transform(X)
        logger.info(f"Dados normalizados: {len(X_scaled)} registros")
        
        # Pegar últimas sequências
        sequence_length = model_metadata.get("architecture", {}).get("sequence_length", 60)
        if len(X_scaled) < sequence_length:
            raise ValueError(
                f"Dados insuficientes após preprocessamento. "
                f"Necessário: {sequence_length} dias, Disponível: {len(X_scaled)} dias"
            )
        
        # Criar sequência
        X_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)
        logger.info(f"Sequência criada com shape: {X_sequence.shape}")
        
        # Fazer previsões iterativas
        predictions = []
        current_sequence = X_sequence.copy()
        
        for i in range(request.days):
            # Prever próximo valor
            pred = model.predict(current_sequence)
            logger.info(f"Predição {i+1}: shape={pred.shape}, valor={pred}")
            
            # Extrair valor da predição
            pred_value = float(pred.flatten()[0])
            predictions.append(pred_value)
            
            # Atualizar sequência (simplificado - idealmente recalcular features)
            # Aqui estamos apenas movendo a janela
            # Em produção, você recalcularia as features com o novo valor
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred_value  # Atualizar último valor
        
        # Gerar datas futuras
        last_date = df.index[-1]
        future_dates = []
        for i in range(1, request.days + 1):
            future_date = last_date + timedelta(days=i)
            # Pular fins de semana
            while future_date.weekday() >= 5:  # 5=sábado, 6=domingo
                future_date += timedelta(days=1)
            future_dates.append(future_date.strftime("%Y-%m-%d"))
        
        # Determinar confiança baseado nas métricas
        mape = model_metadata.get("metrics", {}).get("MAPE", 100)
        if mape < 5:
            confidence = "high"
        elif mape < 10:
            confidence = "medium"
        else:
            confidence = "low"
        
        last_price = float(df['Close'].iloc[-1])
        
        logger.info("Previsão realizada com sucesso")
        
        # Registrar predição nas métricas
        if api_metrics:
            api_metrics.record_prediction(request.days, confidence)
        
        return PredictionResponse(
            predictions=predictions,
            dates=future_dates,
            last_price=last_price,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Erro ao fazer previsão: {e}")
        if api_metrics:
            api_metrics.record_error("PredictionError", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao fazer previsão: {str(e)}"
        )


@app.get("/backtest", tags=["Advanced"])
async def run_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    train_window: int = 365,
    prediction_horizon: int = 5,
    step_days: int = 30
):
    """
    Executa backtesting do modelo.
    
    Args:
        start_date: Data inicial (YYYY-MM-DD)
        end_date: Data final (YYYY-MM-DD)
        train_window: Janela de treino em dias
        prediction_horizon: Dias à frente para prever
        step_days: Passo entre backtests
        
    Returns:
        Resultados do backtesting
    """
    if not ADVANCED_FEATURES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Funcionalidade de backtesting não disponível"
        )
    
    try:
        logger.info(f"Iniciando backtesting: {start_date} até {end_date}")
        
        # Carregar dados completos
        data_path = BASE_DIR / "data" / "processed" / "stock_data_processed.csv"
        if not data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dados processados não encontrados"
            )
        
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        
        # Criar backtester
        backtester = Backtester(model, preprocessor, data)
        
        # Executar backtesting
        results_df = backtester.run_backtest(
            start_date=start_date,
            end_date=end_date,
            train_window=train_window,
            prediction_horizon=prediction_horizon,
            step_days=step_days
        )
        
        # Obter sumário
        summary = backtester.get_summary()
        
        # Salvar resultados
        backtester.save_results()
        
        logger.info(f"Backtesting concluído: {len(results_df)} iterações")
        
        return {
            "status": "success",
            "summary": summary,
            "results_saved": "logs/backtest_results.json",
            "message": f"Backtesting executado com {summary['total_iterations']} iterações"
        }
    
    except Exception as e:
        logger.error(f"Erro no backtesting: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao executar backtesting: {str(e)}"
        )


@app.get("/feature-importance", tags=["Advanced"])
async def get_feature_importance(
    n_samples: int = 100
):
    """
    Calcula feature importance usando SHAP.
    
    Args:
        n_samples: Número de amostras para análise
        
    Returns:
        Importância das features
    """
    if not ADVANCED_FEATURES_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Funcionalidade de feature importance não disponível"
        )
    
    try:
        logger.info("Calculando feature importance...")
        
        # Carregar dados com features
        data_path = BASE_DIR / "data" / "processed" / "stock_data_features.csv"
        if not data_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dados com features não encontrados"
            )
        
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        
        # Preparar dados
        normalized = preprocessor.normalize_data(data)
        X, y = preprocessor.create_sequences(normalized, lookback=30)
        
        # Limitar amostras
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Criar explainer
        feature_names = create_feature_names()
        explainer = ModelExplainer(model.model, feature_names)
        
        # Criar e calcular SHAP
        explainer.create_explainer(X_sample[:50], max_samples=50)
        explainer.explain(X_sample)
        
        # Obter importância
        importance_df = explainer.get_feature_importance()
        
        # Salvar resultados
        explainer.save_explanation()
        
        # Criar visualizações
        explainer.plot_feature_importance(save_path='logs/feature_importance.png')
        
        logger.info("Feature importance calculada com sucesso")
        
        return {
            "status": "success",
            "feature_importance": importance_df.to_dict('records'),
            "top_5_features": importance_df.head(5)['feature'].tolist(),
            "results_saved": "logs/feature_importance.json",
            "visualization_saved": "logs/feature_importance.png"
        }
    
    except Exception as e:
        logger.error(f"Erro ao calcular feature importance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao calcular feature importance: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Handler customizado para exceções HTTP.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            detail=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handler customizado para exceções gerais.
    """
    logger.error(f"Erro não tratado: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# ==================== ENDPOINTS DE CARTEIRA ====================

@app.post("/portfolio/stocks", tags=["Portfolio"])
async def add_stock_to_portfolio(
    symbol: str,
    name: str,
    quantity: float = 0,
    avg_price: float = 0
):
    """
    Adiciona uma ação à carteira.
    
    Args:
        symbol: Símbolo da ação (ex: PETR4.SA)
        name: Nome da empresa
        quantity: Quantidade de ações
        avg_price: Preço médio de compra
    """
    if not ADVANCED_FEATURES_AVAILABLE or not portfolio_manager:
        raise HTTPException(
            status_code=503,
            detail="Sistema de carteira não disponível"
        )
    
    try:
        success = portfolio_manager.add_stock(symbol, name, quantity, avg_price)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Ação {symbol} já existe na carteira"
            )
        
        return {
            "message": f"Ação {symbol} adicionada com sucesso",
            "stock": portfolio_manager.get_stock(symbol)
        }
    except Exception as e:
        logger.error(f"Erro ao adicionar ação: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/portfolio/stocks/{symbol}", tags=["Portfolio"])
async def remove_stock_from_portfolio(symbol: str):
    """Remove uma ação da carteira."""
    if not ADVANCED_FEATURES_AVAILABLE or not portfolio_manager:
        raise HTTPException(
            status_code=503,
            detail="Sistema de carteira não disponível"
        )
    
    try:
        success = portfolio_manager.remove_stock(symbol)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Ação {symbol} não encontrada na carteira"
            )
        
        return {"message": f"Ação {symbol} removida com sucesso"}
    except Exception as e:
        logger.error(f"Erro ao remover ação: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/stocks", tags=["Portfolio"])
async def list_portfolio_stocks():
    """Lista todas as ações na carteira."""
    if not ADVANCED_FEATURES_AVAILABLE or not portfolio_manager:
        raise HTTPException(
            status_code=503,
            detail="Sistema de carteira não disponível"
        )
    
    try:
        stocks = portfolio_manager.list_stocks()
        return {"stocks": stocks, "total": len(stocks)}
    except Exception as e:
        logger.error(f"Erro ao listar ações: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/summary", tags=["Portfolio"])
async def get_portfolio_summary():
    """Obtém resumo da carteira."""
    if not ADVANCED_FEATURES_AVAILABLE or not portfolio_manager:
        raise HTTPException(
            status_code=503,
            detail="Sistema de carteira não disponível"
        )
    
    try:
        summary = portfolio_manager.get_summary()
        return summary
    except Exception as e:
        logger.error(f"Erro ao obter resumo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/train", tags=["Portfolio"])
async def train_portfolio_stocks(
    symbols: list[str] = None,
    parallel: bool = False,
    epochs: int = 150,
    batch_size: int = 16
):
    """
    Treina modelos para ações na carteira.
    
    Args:
        symbols: Lista de símbolos (None = todas)
        parallel: Treinamento paralelo
        epochs: Épocas de treinamento
        batch_size: Tamanho do batch
    """
    if not ADVANCED_FEATURES_AVAILABLE or not multi_trainer:
        raise HTTPException(
            status_code=503,
            detail="Sistema de treinamento não disponível"
        )
    
    try:
        logger.info(f"Iniciando treinamento de {len(symbols) if symbols else 'todas as'} ações")
        
        results = multi_trainer.train_multiple_stocks(
            symbols=symbols,
            parallel=parallel,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return results
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/train/{symbol}", tags=["Portfolio"])
async def train_single_stock(
    symbol: str,
    epochs: int = 150,
    batch_size: int = 16,
    sequence_length: int = 30
):
    """
    Treina modelo para uma única ação.
    
    Args:
        symbol: Símbolo da ação
        epochs: Épocas de treinamento
        batch_size: Tamanho do batch
        sequence_length: Janela de tempo
    """
    if not ADVANCED_FEATURES_AVAILABLE or not multi_trainer:
        raise HTTPException(
            status_code=503,
            detail="Sistema de treinamento não disponível"
        )
    
    try:
        logger.info(f"Iniciando treinamento de {symbol}")
        
        result = multi_trainer.train_single_stock(
            symbol=symbol,
            epochs=epochs,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        if not result:
            return {
                'success': False,
                'error': 'Falha no treinamento: resultado vazio',
                'symbol': symbol
            }
        
        # Retornar resultado mesmo se falhou (com success=False)
        # Isso permite que o frontend mostre a mensagem de erro correta
        return result
        
    except Exception as e:
        logger.error(f"Erro no treinamento de {symbol}: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'symbol': symbol
        }


@app.post("/portfolio/predict", tags=["Portfolio"])
async def predict_portfolio_stocks(
    symbols: list[str] = None,
    days_ahead: int = 5
):
    """
    Gera predições para ações na carteira.
    
    Args:
        symbols: Lista de símbolos (None = todas treinadas)
        days_ahead: Número de dias para prever
    """
    if not ADVANCED_FEATURES_AVAILABLE or not multi_predictor:
        raise HTTPException(
            status_code=503,
            detail="Sistema de predição não disponível"
        )
    
    try:
        logger.info(f"Gerando predições para {len(symbols) if symbols else 'todas as'} ações")
        
        results = multi_predictor.predict_multiple_stocks(
            symbols=symbols,
            days_ahead=days_ahead
        )
        
        return results
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/outlook", tags=["Portfolio"])
async def get_portfolio_outlook(days_ahead: int = 5):
    """
    Obtém visão geral da carteira com predições.
    
    Args:
        days_ahead: Número de dias para prever
    """
    if not ADVANCED_FEATURES_AVAILABLE or not multi_predictor:
        raise HTTPException(
            status_code=503,
            detail="Sistema de predição não disponível"
        )
    
    try:
        outlook = multi_predictor.get_portfolio_outlook(days_ahead=days_ahead)
        return outlook
    except Exception as e:
        logger.error(f"Erro ao gerar outlook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
