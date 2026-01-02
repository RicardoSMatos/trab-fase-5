"""
Dashboard Streamlit para gestão de carteira de ações com LSTM.

Interface completa para adicionar ações, treinar modelos e gerar predições.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import json
import os

# Configuração da página
st.set_page_config(
    page_title="Portfolio Manager - LSTM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL da API - Detectar ambiente Docker ou local
DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("URL da API", value=DEFAULT_API_URL)

# Estilo customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==================== FUNÇÕES DE API ====================

def api_request(endpoint: str, method: str = "GET", **kwargs):
    """Faz requisição para a API."""
    try:
        url = f"{API_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=kwargs.get('params', {}), timeout=10)
        elif method == "POST":
            response = requests.post(url, json=kwargs.get('json', {}), params=kwargs.get('params', {}), timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        else:
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na API: {str(e)}")
        return None


def get_portfolio_summary():
    """Obtém resumo da carteira."""
    return api_request("/portfolio/summary")


def list_stocks():
    """Lista ações na carteira."""
    return api_request("/portfolio/stocks")


def add_stock(symbol: str, name: str, quantity: float, avg_price: float):
    """Adiciona ação à carteira."""
    return api_request(
        "/portfolio/stocks",
        method="POST",
        params={
            "symbol": symbol,
            "name": name,
            "quantity": quantity,
            "avg_price": avg_price
        }
    )


def remove_stock(symbol: str):
    """Remove ação da carteira."""
    return api_request(f"/portfolio/stocks/{symbol}", method="DELETE")


def train_single_stock(symbol: str, epochs: int, batch_size: int, sequence_length: int):
    """Treina modelo para uma ação."""
    return api_request(
        f"/portfolio/train/{symbol}",
        method="POST",
        params={
            "epochs": epochs,
            "batch_size": batch_size,
            "sequence_length": sequence_length
        }
    )


def train_multiple_stocks(symbols: list, parallel: bool, epochs: int, batch_size: int):
    """Treina modelos para múltiplas ações."""
    return api_request(
        "/portfolio/train",
        method="POST",
        params={
            "parallel": parallel,
            "epochs": epochs,
            "batch_size": batch_size
        },
        json={"symbols": symbols} if symbols else {}
    )


def predict_portfolio(symbols: list = None, days_ahead: int = 5):
    """Gera predições para a carteira."""
    return api_request(
        "/portfolio/predict",
        method="POST",
        params={"days_ahead": days_ahead},
        json=symbols  # Lista direta
    )


def get_portfolio_outlook(days_ahead: int = 5):
    """Obtém visão geral da carteira."""
    return api_request("/portfolio/outlook", params={"days_ahead": days_ahead})


# ==================== PÁGINAS ====================

def page_portfolio_overview():
    """Página de visão geral da carteira."""
    st.markdown('<h1 class="main-header">📊 Visão Geral da Carteira</h1>', unsafe_allow_html=True)
    
    # Obter resumo e lista de ações
    summary = get_portfolio_summary()
    stocks_response = list_stocks()
    
    if not summary:
        st.warning("Não foi possível carregar o resumo da carteira")
        return
    
    # Extrair lista de ações
    stocks = stocks_response.get('stocks', []) if stocks_response else []
    
    # Calcular métricas
    by_status = summary.get('by_status', {})
    trained_stocks = by_status.get('trained', 0)
    pending_stocks = summary.get('total_stocks', 0) - trained_stocks
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total de Ações",
            summary.get('total_stocks', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            "Modelos Treinados",
            trained_stocks,
            delta=None
        )
    
    with col3:
        st.metric(
            "Modelos Pendentes",
            pending_stocks,
            delta=None
        )
    
    with col4:
        # Como quantity e avg_price não são obrigatórios, total investido será sempre 0
        # Mantém por enquanto para compatibilidade futura
        total_invested = sum(stock.get('quantity', 0) * stock.get('avg_price', 0) for stock in stocks)
        st.metric(
            "Total Investido",
            f"R$ {total_invested:,.2f}",
            delta=None,
            help="Feature de investimento ainda não implementada"
        )
    
    st.divider()
    
    # Debug info (expandir para ver detalhes)
    with st.expander("🔍 Debug Info"):
        st.write(f"**API URL:** {API_URL}")
        st.write(f"**Summary response:** {summary}")
        st.write(f"**Stocks response:** {stocks_response}")
        st.write(f"**Stocks list length:** {len(stocks)}")
    
    # Lista de ações
    
    if not stocks:
        st.info("📝 Sua carteira está vazia. Adicione ações na página 'Gerenciar Ações'.")
        st.warning("💡 **Dica:** Verifique se a URL da API está correta no sidebar.")
        return
    
    # Tabela de ações
    st.subheader("Ações na Carteira")
    
    df_stocks = pd.DataFrame(stocks)
    
    # Adicionar coluna de status formatada
    df_stocks['Status'] = df_stocks.apply(
        lambda row: '✅ Treinado' if row.get('status') == 'trained' or row.get('training_status') == 'completed' 
        else '⏳ Pendente',
        axis=1
    )
    
    # Adicionar data de treinamento formatada
    df_stocks['last_trained_formatted'] = df_stocks['last_trained'].apply(
        lambda x: pd.to_datetime(x).strftime('%d/%m/%Y %H:%M') if pd.notna(x) else 'N/A'
    )
    
    # Selecionar colunas para exibir
    display_cols = ['symbol', 'name', 'Status', 'last_trained_formatted']
    df_display = df_stocks[display_cols].copy()
    df_display.columns = ['Símbolo', 'Nome', 'Status', 'Último Treinamento']
    
    st.dataframe(df_display, use_container_width=True)
    
    # Métricas de qualidade dos modelos
    if len(stocks) > 0:
        trained_stocks = [s for s in stocks if s.get('status') == 'trained' or s.get('training_status') == 'completed']
        
        if trained_stocks:
            st.subheader("📈 Métricas dos Modelos Treinados")
            
            for stock in trained_stocks:
                with st.expander(f"📊 {stock['symbol']} - {stock['name']}"):
                    metrics = stock.get('metrics', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
                    with col2:
                        st.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
                    with col3:
                        st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
                    with col4:
                        st.metric("R²", f"{metrics.get('R2', 0):.4f}")


def page_manage_stocks():
    """Página de gerenciamento de ações."""
    st.markdown('<h1 class="main-header">🔧 Gerenciar Ações</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["➕ Adicionar Ação", "🗑️ Remover Ação"])
    
    with tab1:
        st.subheader("Adicionar Nova Ação à Carteira")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input(
                "Símbolo da Ação",
                placeholder="Ex: PETR4.SA",
                help="Símbolo da ação no formato do Yahoo Finance"
            )
            
            quantity = st.number_input(
                "Quantidade de Ações",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Número de ações que você possui"
            )
        
        with col2:
            name = st.text_input(
                "Nome da Empresa",
                placeholder="Ex: Petrobras",
                help="Nome da empresa"
            )
            
            avg_price = st.number_input(
                "Preço Médio de Compra (R$)",
                min_value=0.0,
                value=0.0,
                step=0.01,
                help="Preço médio pago por ação"
            )
        
        if st.button("➕ Adicionar à Carteira", type="primary", use_container_width=True):
            if not symbol or not name:
                st.error("Preencha o símbolo e o nome da ação")
            else:
                with st.spinner(f"Adicionando {symbol}..."):
                    result = add_stock(symbol, name, quantity, avg_price)
                    
                    if result:
                        st.success(f"✅ {symbol} adicionado com sucesso!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
    
    with tab2:
        st.subheader("Remover Ação da Carteira")
        
        # Listar ações
        stocks_data = list_stocks()
        
        if not stocks_data or not stocks_data.get('stocks'):
            st.info("Nenhuma ação na carteira")
            return
        
        stocks = stocks_data['stocks']
        stock_options = {f"{s['symbol']} - {s['name']}": s['symbol'] for s in stocks}
        
        selected_stock = st.selectbox(
            "Selecione a ação para remover",
            options=list(stock_options.keys())
        )
        
        if selected_stock:
            symbol = stock_options[selected_stock]
            
            st.warning(f"⚠️ Você está prestes a remover **{selected_stock}** da carteira.")
            st.info("O modelo treinado também será excluído.")
            
            if st.button("🗑️ Confirmar Remoção", type="secondary", use_container_width=True):
                with st.spinner(f"Removendo {symbol}..."):
                    result = remove_stock(symbol)
                    
                    if result:
                        st.success(f"✅ {symbol} removido com sucesso!")
                        time.sleep(1)
                        st.rerun()


def page_train_models():
    """Página de treinamento de modelos."""
    st.markdown('<h1 class="main-header">🎯 Treinar Modelos</h1>', unsafe_allow_html=True)
    
    # Listar ações
    stocks_data = list_stocks()
    
    if not stocks_data or not stocks_data.get('stocks'):
        st.info("📝 Adicione ações à carteira primeiro")
        return
    
    stocks = stocks_data['stocks']
    
    # Filtrar ações não treinadas
    untrained = [s for s in stocks if not s.get('model_trained')]
    trained = [s for s in stocks if s.get('model_trained')]
    
    # Estatísticas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Ações", len(stocks))
    with col2:
        st.metric("Modelos Treinados", len(trained))
    with col3:
        st.metric("Pendentes", len(untrained))
    
    st.divider()
    
    # Tabs para treinamento
    tab1, tab2 = st.tabs(["🎯 Treinar Individual", "🚀 Treinar em Lote"])
    
    with tab1:
        st.subheader("Treinar Modelo Individual")
        
        if not stocks:
            st.info("Nenhuma ação disponível")
            return
        
        # Selecionar ação
        stock_options = {f"{s['symbol']} - {s['name']}": s for s in stocks}
        selected_stock_name = st.selectbox(
            "Selecione a ação",
            options=list(stock_options.keys())
        )
        
        selected_stock = stock_options[selected_stock_name]
        symbol = selected_stock['symbol']
        
        # Exibir status
        if selected_stock.get('model_trained'):
            st.info(f"✅ Modelo já treinado para {symbol}")
            last_trained = selected_stock.get('last_trained')
            if last_trained:
                st.caption(f"Último treinamento: {last_trained}")
        
        # Parâmetros de treinamento
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.number_input("Épocas", min_value=10, max_value=500, value=150, step=10,
                                   help="Mais épocas com Early Stopping para melhor convergência")
        with col2:
            batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=16, step=8,
                                       help="Menor batch size para datasets pequenos")
        with col3:
            sequence_length = st.number_input("Janela Temporal (dias)", min_value=20, max_value=120, value=30, step=10,
                                            help="30 dias = padrões de médio prazo com mais amostras")
        
        if st.button(f"🎯 Treinar {symbol}", type="primary", use_container_width=True):
            # Validação de volatilidade
            with st.spinner("Validando ação e preparando dados..."):
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period='2y')
                    
                    if df.empty:
                        st.error("❌ Não foi possível obter dados históricos da ação.")
                        st.stop()
                    
                    # Calcular métricas de volatilidade
                    price_mean = df['Close'].mean()
                    price_std = df['Close'].std()
                    price_min = df['Close'].min()
                    price_max = df['Close'].max()
                    volatility_pct = (price_std / price_mean) * 100
                    variation_pct = ((price_max - price_min) / price_min) * 100
                    
                    # Mostrar análise de volatilidade
                    st.subheader("📊 Análise de Volatilidade")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Preço Médio", f"R$ {price_mean:.2f}")
                    with col2:
                        st.metric("Volatilidade", f"{volatility_pct:.1f}%", 
                                 delta="Alta" if volatility_pct > 25 else "Moderada",
                                 delta_color="inverse" if volatility_pct > 25 else "normal")
                    with col3:
                        st.metric("Range", f"R$ {price_min:.2f} - R$ {price_max:.2f}")
                    with col4:
                        st.metric("Variação", f"{variation_pct:.1f}%",
                                 delta="Alta" if variation_pct > 100 else "Moderada",
                                 delta_color="inverse" if variation_pct > 100 else "normal")
                    
                    # Alertas de volatilidade
                    if volatility_pct > 30 or variation_pct > 150:
                        st.error(f"""
                        ⚠️ **ATENÇÃO: Ação Muito Volátil**
                        
                        Esta ação apresenta alta volatilidade ({volatility_pct:.1f}%), o que pode resultar em predições menos precisas.
                        
                        **Espere métricas como:**
                        - R² negativo ou muito baixo
                        - MAPE > 25%
                        - MAE alto relativo ao preço
                        
                        **💡 Sugestões de ações mais estáveis:**
                        - PETR4.SA (Petrobras)
                        - VALE3.SA (Vale)
                        - ITUB4.SA (Itaú)
                        - BBDC4.SA (Bradesco)
                        """)
                        
                        if not st.checkbox("⚠️ Continuar mesmo assim", key=f"continue_{symbol}"):
                            st.stop()
                    
                    elif volatility_pct > 20:
                        st.warning(f"""
                        ⚡ **Volatilidade Moderada-Alta** ({volatility_pct:.1f}%)
                        
                        Predições podem ter precisão reduzida. Considere aumentar épocas ou testar com ações mais estáveis.
                        """)
                    
                    else:
                        st.success(f"""
                        ✅ **Volatilidade Adequada** ({volatility_pct:.1f}%)
                        
                        Esta ação tem características favoráveis para LSTM.
                        """)
                
                except Exception as e:
                    st.warning(f"⚠️ Não foi possível validar volatilidade: {str(e)}\nContinuando com treinamento...")
            
            with st.spinner(f"Treinando modelo para {symbol}... Isso pode levar alguns minutos."):
                # Barra de progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Coletando dados...")
                progress_bar.progress(25)
                
                # Treinar
                result = train_single_stock(symbol, epochs, batch_size, sequence_length)
                
                progress_bar.progress(100)
                status_text.text("✅ Concluído!")
                
                if result and result.get('success'):
                    st.success(f"✅ Modelo treinado com sucesso para {symbol}!")
                    
                    # Exibir métricas detalhadas
                    st.subheader("📊 Métricas do Modelo")
                    
                    metrics = result.get('metrics', {})
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("MAE", f"R$ {metrics.get('MAE', 0):.2f}", help="Erro Médio Absoluto")
                    with col2:
                        st.metric("RMSE", f"R$ {metrics.get('RMSE', 0):.2f}", help="Raiz do Erro Quadrático Médio")
                    with col3:
                        st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%", help="Erro Percentual Médio")
                    with col4:
                        st.metric("R²", f"{metrics.get('R2', 0):.4f}", help="Coeficiente de Determinação")
                    with col5:
                        st.metric("MSE", f"{metrics.get('MSE', 0):.2f}", help="Erro Quadrático Médio")
                    
                    # Informações adicionais
                    st.info(f"""
                    **Detalhes do Treinamento:**
                    - Amostras de treino: {result.get('train_samples', 'N/A')}
                    - Amostras de teste: {result.get('test_samples', 'N/A')}
                    - Modelo salvo em: `{result.get('model_path', 'N/A')}`
                    
                    💡 Métricas em escala real (R$). Compare com o console para validar.
                    """)
                    
                    st.balloons()
                    
                    # NÃO fazer rerun - deixar o usuário ver os resultados
                    st.info("✨ Treinamento concluído! Você pode treinar outra ação ou gerar predições na aba 'Predições'.")
                else:
                    error_msg = result.get('error', 'Erro desconhecido') if result else 'Erro na comunicação com a API'
                    st.error(f"❌ Erro no treinamento: {error_msg}")
    
    with tab2:
        st.subheader("Treinar Múltiplos Modelos")
        
        # Selecionar ações
        all_symbols = [s['symbol'] for s in stocks]
        
        selected_symbols = st.multiselect(
            "Selecione as ações para treinar",
            options=all_symbols,
            default=[s['symbol'] for s in untrained[:3]]  # Primeiras 3 não treinadas
        )
        
        # Opções
        col1, col2, col3 = st.columns(3)
        
        with col1:
            parallel = st.checkbox("Treinamento Paralelo", value=False, help="Treina múltiplas ações simultaneamente (mais rápido)")
        with col2:
            epochs = st.number_input("Épocas", min_value=10, max_value=500, value=150, step=10, key="batch_epochs",
                                   help="Mais épocas com Early Stopping")
        with col3:
            batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=16, step=8, key="batch_size",
                                       help="Menor para datasets pequenos")
        
        if not selected_symbols:
            st.info("Selecione pelo menos uma ação")
        else:
            st.info(f"📊 {len(selected_symbols)} ações selecionadas")
            
            if st.button(f"🚀 Treinar {len(selected_symbols)} Modelos", type="primary", use_container_width=True):
                with st.spinner(f"Treinando {len(selected_symbols)} modelos... Isso pode levar vários minutos."):
                    # Container para status
                    status_container = st.empty()
                    
                    # Treinar
                    results = train_multiple_stocks(selected_symbols, parallel, epochs, batch_size)
                    
                    if results:
                        summary = results.get('summary', {})
                        st.success(f"✅ Treinamento concluído!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", summary.get('total', 0))
                        with col2:
                            st.metric("Sucesso", summary.get('successful', 0))
                        with col3:
                            st.metric("Falhas", summary.get('failed', 0))
                        
                        # Detalhes
                        with st.expander("Ver Detalhes"):
                            for symbol, result in results.get('results', {}).items():
                                if result and result.get('success'):
                                    st.success(f"✅ {symbol}: Treinado com sucesso")
                                else:
                                    error_msg = result.get('error', 'Erro desconhecido') if result else 'Sem resposta'
                                    st.error(f"❌ {symbol}: {error_msg}")
                        
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Erro ao treinar modelos. Verifique os logs da API.")


def page_predictions():
    """Página de predições."""
    st.markdown('<h1 class="main-header">🔮 Predições</h1>', unsafe_allow_html=True)
    
    # Botão para forçar atualização
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Atualizar", help="Forçar atualização dos dados"):
            st.cache_data.clear()
            st.rerun()
    
    # Verificar quais ações têm modelos treinados
    stocks_data = list_stocks()
    
    if not stocks_data or not stocks_data.get('stocks'):
        st.warning("📝 Nenhuma ação na carteira. Adicione ações primeiro.")
        return
    
    stocks = stocks_data['stocks']
    
    # Filtrar ações treinadas (verificar status ou training_status)
    trained_stocks = [
        s for s in stocks 
        if s.get('status') == 'trained' or s.get('training_status') == 'completed'
    ]
    
    # Mostrar status
    st.info(f"📊 Ações na carteira: {len(stocks)} | Modelos treinados: {len(trained_stocks)}")
    
    if not trained_stocks:
        st.warning("⚠️ Nenhum modelo treinado ainda. Vá para 'Treinar Modelos' para treinar.")
        
        # Mostrar quais ações precisam treinar
        st.subheader("Ações Pendentes de Treinamento")
        for stock in stocks:
            status = stock.get('training_status', 'pending')
            symbol = stock['symbol']
            name = stock['name']
            
            if status == 'failed':
                st.error(f"❌ {symbol} - {name}: Falha no treinamento")
            else:
                st.warning(f"⏳ {symbol} - {name}: Pendente")
        return
    
    # Mostrar ações treinadas
    st.subheader("Ações com Modelos Treinados")
    for stock in trained_stocks:
        st.success(f"✅ {stock['symbol']} - {stock['name']} (Treinado em {stock.get('last_trained', 'N/A')})")
    
    st.divider()
    
    # Obter outlook
    days_ahead = st.slider("Dias para prever", min_value=1, max_value=30, value=5)
    
    if st.button("🔮 Gerar Predições", type="primary", use_container_width=True):
        with st.spinner("Gerando predições..."):
            outlook = get_portfolio_outlook(days_ahead)
            
            if not outlook:
                st.error("❌ Erro ao obter predições da API. Verifique se a API está rodando.")
                return
            
            if outlook.get('total_stocks', 0) == 0:
                st.warning("⚠️ Nenhuma ação com modelo treinado encontrada.")
                st.info(f"Debug: Resposta da API: {outlook}")
                return
            
            # Resumo geral
            st.subheader("Visão Geral da Carteira")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Ações Analisadas", outlook.get('total_stocks', 0))
            with col2:
                current_value = outlook.get('current_portfolio_value', 0)
                st.metric("Valor Atual", f"R$ {current_value:,.2f}")
            with col3:
                predicted_value = outlook.get('predicted_portfolio_value', 0)
                st.metric("Valor Previsto", f"R$ {predicted_value:,.2f}")
            with col4:
                change_pct = outlook.get('expected_change_pct', 0)
                st.metric("Variação Esperada", f"{change_pct:+.2f}%", delta=f"{change_pct:+.2f}%")
            
            # Outlook
            outlook_emoji = "📈" if outlook.get('outlook') == 'bullish' else "📉"
            outlook_text = "Alta" if outlook.get('outlook') == 'bullish' else "Baixa"
            outlook_color = "green" if outlook.get('outlook') == 'bullish' else "red"
            
            st.markdown(f"### {outlook_emoji} Tendência: <span style='color:{outlook_color};font-weight:bold'>{outlook_text}</span>", unsafe_allow_html=True)
            
            st.divider()
            
            # Predições individuais
            st.subheader("Predições por Ação")
            
            predictions = outlook.get('predictions', {})
            
            for symbol, pred_data in predictions.items():
                with st.expander(f"📊 {symbol} - {pred_data['current_price']:.2f}"):
                    # Métricas
                    col1, col2, col3 = st.columns(3)
                    
                    # Calcular preço previsto médio dos próximos dias
                    avg_predicted_price = sum(p['predicted_price'] for p in pred_data['predictions']) / len(pred_data['predictions'])
                    last_pred = pred_data['predictions'][-1]  # D+5
                    
                    with col1:
                        st.metric("Preço Atual", f"R$ {pred_data['current_price']:.2f}")
                    with col2:
                        st.metric(
                            "Preço Previsto (D+5)", 
                            f"R$ {last_pred['predicted_price']:.2f}",
                            help=f"Previsão para {last_pred['date']} | Média próximos 5 dias: R$ {avg_predicted_price:.2f}"
                        )
                    with col3:
                        st.metric("Variação", f"{last_pred['change_pct']:+.2f}%", delta=f"R$ {last_pred['change']:+.2f}")
                    
                    # Gráfico
                    historical = pred_data.get('historical', {})
                    hist_dates = historical.get('dates', [])
                    hist_prices = historical.get('prices', [])
                    
                    pred_dates = [p['date'] for p in pred_data['predictions']]
                    pred_prices = [p['predicted_price'] for p in pred_data['predictions']]
                    
                    # Debug: verificar dados
                    st.caption(f"🔍 Debug: {len(pred_data['predictions'])} predições recebidas | {len(hist_dates)} dias de histórico")
                    
                    fig = go.Figure()
                    
                    # Histórico
                    fig.add_trace(go.Scatter(
                        x=hist_dates,
                        y=hist_prices,
                        mode='lines',
                        name='Histórico',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>%{x}</b><br>Preço: R$ %{y:.2f}<extra></extra>'
                    ))
                    
                    # Linha de conexão (último histórico → primeira predição)
                    if hist_dates and pred_dates:
                        fig.add_trace(go.Scatter(
                            x=[hist_dates[-1], pred_dates[0]],
                            y=[hist_prices[-1], pred_prices[0]],
                            mode='lines',
                            name='Transição',
                            line=dict(color='gray', width=1, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Predições
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=pred_prices,
                        mode='lines+markers',
                        name='Predição',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8),
                        hovertemplate='<b>%{x}</b><br>Predição: R$ %{y:.2f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"Predição para {symbol}",
                        xaxis_title="Data",
                        yaxis_title="Preço (R$)",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Métricas do modelo
                    test_metrics = pred_data.get('test_metrics', {})
                    if test_metrics:
                        st.caption("**Métricas do Modelo:**")
                        mcol1, mcol2, mcol3 = st.columns(3)
                        with mcol1:
                            st.caption(f"MAE: {test_metrics.get('MAE', 0):.4f}")
                        with mcol2:
                            st.caption(f"RMSE: {test_metrics.get('RMSE', 0):.4f}")
                        with mcol3:
                            st.caption(f"MAPE: {test_metrics.get('MAPE', 0):.2f}%")


def page_analytics():
    """Página de análises avançadas."""
    st.markdown('<h1 class="main-header">📈 Análises Avançadas</h1>', unsafe_allow_html=True)
    
    # Obter dados
    outlook = get_portfolio_outlook(days_ahead=5)
    
    if not outlook or outlook.get('total_stocks', 0) == 0:
        st.warning("Gere predições primeiro na página 'Predições'")
        return
    
    predictions = outlook.get('predictions', {})
    
    # Análise de correlação
    st.subheader("📊 Análise Comparativa")
    
    # Criar DataFrame para comparação
    comparison_data = []
    for symbol, pred_data in predictions.items():
        last_pred = pred_data['predictions'][-1]
        comparison_data.append({
            'Símbolo': symbol,
            'Preço Atual': pred_data['current_price'],
            'Preço Previsto': last_pred['predicted_price'],
            'Variação (%)': last_pred['change_pct']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Gráfico de barras
    fig = px.bar(
        df_comparison,
        x='Símbolo',
        y='Variação (%)',
        title='Variação Esperada por Ação',
        color='Variação (%)',
        color_continuous_scale=['red', 'yellow', 'green'],
        text='Variação (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de comparação
    st.subheader("📋 Comparação Detalhada")
    st.dataframe(df_comparison.style.format({
        'Preço Atual': 'R$ {:.2f}',
        'Preço Previsto': 'R$ {:.2f}',
        'Variação (%)': '{:+.2f}%'
    }), use_container_width=True)
    
    # Top performers
    st.subheader("🏆 Melhores e Piores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Maiores Altas Esperadas**")
        top_gainers = df_comparison.nlargest(3, 'Variação (%)')
        for _, row in top_gainers.iterrows():
            st.success(f"**{row['Símbolo']}**: {row['Variação (%)']:+.2f}%")
    
    with col2:
        st.markdown("**📉 Maiores Quedas Esperadas**")
        top_losers = df_comparison.nsmallest(3, 'Variação (%)')
        for _, row in top_losers.iterrows():
            st.error(f"**{row['Símbolo']}**: {row['Variação (%)']:+.2f}%")


# ==================== NAVEGAÇÃO ====================

def main():
    """Função principal do app."""
    
    # Sidebar
    st.sidebar.title("📊 Portfolio Manager")
    st.sidebar.markdown("---")
    
    # Navegação
    page = st.sidebar.radio(
        "Navegação",
        options=[
            "📊 Visão Geral",
            "🔧 Gerenciar Ações",
            "🎯 Treinar Modelos",
            "🔮 Predições",
            "📈 Análises"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Informações")
    st.sidebar.info(
        "Sistema de gestão de carteira com "
        "modelos LSTM para predição de preços de ações."
    )
    
    # Renderizar página selecionada
    if page == "📊 Visão Geral":
        page_portfolio_overview()
    elif page == "🔧 Gerenciar Ações":
        page_manage_stocks()
    elif page == "🎯 Treinar Modelos":
        page_train_models()
    elif page == "🔮 Predições":
        page_predictions()
    elif page == "📈 Análises":
        page_analytics()


if __name__ == "__main__":
    main()
