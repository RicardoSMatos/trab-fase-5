"""
Dashboard Web Interativo para o Modelo de Previsão de Ações.

Este dashboard permite:
- Fazer predições interativas
- Visualizar dados históricos
- Ver métricas do modelo
- Explorar feature importance
- Comparar diferentes períodos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configuração da página
st.set_page_config(
    page_title="LSTM Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS customizados
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
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# URL da API
API_URL = "http://localhost:8000"


@st.cache_data
def load_data():
    """Carrega dados históricos."""
    try:
        data_path = Path(__file__).parent / "data" / "processed" / "stock_data_processed.csv"
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None


@st.cache_data
def load_feature_data():
    """Carrega dados com features."""
    try:
        data_path = Path(__file__).parent / "data" / "processed" / "stock_data_features.csv"
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        return data
    except Exception as e:
        return None


def check_api_health():
    """Verifica se a API está rodando."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def make_prediction(days: int):
    """Faz predição via API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"days": days},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erro na API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erro ao conectar com a API: {e}")
        return None


def plot_historical_data(data, period='1Y'):
    """Plota dados históricos com Plotly."""
    # Filtrar período
    if period == '1M':
        data_filtered = data.last('1M')
    elif period == '3M':
        data_filtered = data.last('3M')
    elif period == '6M':
        data_filtered = data.last('6M')
    elif period == '1Y':
        data_filtered = data.last('1Y')
    elif period == '2Y':
        data_filtered = data.last('2Y')
    else:  # ALL
        data_filtered = data
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar candlestick
    if all(col in data_filtered.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=data_filtered.index,
            open=data_filtered['Open'],
            high=data_filtered['High'],
            low=data_filtered['Low'],
            close=data_filtered['Close'],
            name='PETR4'
        ))
    else:
        # Se não tiver OHLC, plotar apenas Close
        fig.add_trace(go.Scatter(
            x=data_filtered.index,
            y=data_filtered['Close'],
            mode='lines',
            name='Close',
            line=dict(color='#1f77b4', width=2)
        ))
    
    # Layout
    fig.update_layout(
        title=f'Histórico de Preços - PETR4.SA ({period})',
        yaxis_title='Preço (R$)',
        xaxis_title='Data',
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_prediction(historical_data, prediction_data):
    """Plota predições junto com histórico."""
    # Últimos 60 dias de histórico
    recent_data = historical_data.last('60D')
    
    fig = go.Figure()
    
    # Dados históricos
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['Close'],
        mode='lines',
        name='Histórico',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Último ponto histórico
    last_date = recent_data.index[-1]
    last_price = recent_data['Close'].iloc[-1]
    
    # Predições
    pred_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=len(prediction_data['predictions']),
        freq='D'
    )
    
    # Linha conectando histórico às predições
    fig.add_trace(go.Scatter(
        x=[last_date, pred_dates[0]],
        y=[last_price, prediction_data['predictions'][0]],
        mode='lines',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False
    ))
    
    # Predições
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=prediction_data['predictions'],
        mode='lines+markers',
        name='Predições',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    
    # Adicionar anotações
    for i, (date, pred) in enumerate(zip(pred_dates, prediction_data['predictions'])):
        fig.add_annotation(
            x=date,
            y=pred,
            text=f'R$ {pred:.2f}',
            showarrow=False,
            yshift=15,
            font=dict(size=10, color='#ff7f0e')
        )
    
    fig.update_layout(
        title='Predições vs Histórico',
        yaxis_title='Preço (R$)',
        xaxis_title='Data',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_volume(data, period='1Y'):
    """Plota volume de negociação."""
    # Filtrar período
    if period == '1M':
        data_filtered = data.last('1M')
    elif period == '3M':
        data_filtered = data.last('3M')
    elif period == '6M':
        data_filtered = data.last('6M')
    elif period == '1Y':
        data_filtered = data.last('1Y')
    elif period == '2Y':
        data_filtered = data.last('2Y')
    else:
        data_filtered = data
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data_filtered.index,
        y=data_filtered['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Volume de Negociação',
        yaxis_title='Volume',
        xaxis_title='Data',
        height=300,
        template='plotly_white'
    )
    
    return fig


def plot_features(data):
    """Plota features técnicas."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Médias Móveis', 'RSI', 'MACD'),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Médias móveis
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    if 'MA_7' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_7'], name='MA 7',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'MA_21' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_21'], name='MA 21',
                      line=dict(color='green', width=1)),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        # Linhas de referência
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Sobrecompra", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                     annotation_text="Sobrevenda", row=2, col=1)
    
    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD',
                      line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        height=900,
        template='plotly_white',
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Data", row=3, col=1)
    fig.update_yaxes(title_text="Preço (R$)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


def main():
    """Função principal do dashboard."""
    
    # Header
    st.markdown('<p class="main-header">📈 LSTM Stock Predictor</p>', 
                unsafe_allow_html=True)
    st.markdown("### Dashboard Interativo de Previsão de Ações - Petrobras (PETR4.SA)")
    
    # Verificar API
    api_status = check_api_health()
    
    if api_status:
        st.success("✅ API conectada e funcionando")
    else:
        st.warning("⚠️ API não está respondendo. Inicie com: `uvicorn src.api.main:app --reload`")
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    
    # Carregar dados
    data = load_data()
    feature_data = load_feature_data()
    
    if data is None:
        st.error("Não foi possível carregar os dados. Verifique se os arquivos existem.")
        return
    
    # Informações sobre os dados
    st.sidebar.subheader("📊 Informações dos Dados")
    st.sidebar.metric("Total de Registros", len(data))
    st.sidebar.metric("Período", 
                     f"{data.index.min().strftime('%d/%m/%Y')} - {data.index.max().strftime('%d/%m/%Y')}")
    st.sidebar.metric("Preço Atual", f"R$ {data['Close'].iloc[-1]:.2f}")
    
    # Navegação
    page = st.sidebar.selectbox(
        "Navegação",
        ["🏠 Visão Geral", "🔮 Fazer Predição", "📊 Análise Técnica", "📈 Backtesting", "🎯 Feature Importance"]
    )
    
    # Páginas
    if page == "🏠 Visão Geral":
        show_overview(data)
    
    elif page == "🔮 Fazer Predição":
        show_prediction_page(data, api_status)
    
    elif page == "📊 Análise Técnica":
        show_technical_analysis(data, feature_data)
    
    elif page == "📈 Backtesting":
        show_backtesting_page()
    
    elif page == "🎯 Feature Importance":
        show_feature_importance_page()


def show_overview(data):
    """Página de visão geral."""
    st.header("🏠 Visão Geral do Mercado")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            "Preço Atual",
            f"R$ {current_price:.2f}",
            f"{price_change_pct:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Máxima (52 sem)",
            f"R$ {data['Close'].last('52W').max():.2f}"
        )
    
    with col3:
        st.metric(
            "Mínima (52 sem)",
            f"R$ {data['Close'].last('52W').min():.2f}"
        )
    
    with col4:
        avg_volume = data['Volume'].last('30D').mean()
        st.metric(
            "Volume Médio (30d)",
            f"{avg_volume/1e6:.1f}M"
        )
    
    # Seletor de período
    st.subheader("Gráfico de Preços")
    period = st.selectbox(
        "Selecione o período:",
        ['1M', '3M', '6M', '1Y', '2Y', 'ALL'],
        index=3
    )
    
    # Gráfico principal
    fig = plot_historical_data(data, period)
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume
    st.subheader("Volume de Negociação")
    fig_volume = plot_volume(data, period)
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Estatísticas
    st.subheader("📊 Estatísticas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Últimos 30 dias:**")
        stats_30d = data['Close'].last('30D').describe()
        st.dataframe(stats_30d, use_container_width=True)
    
    with col2:
        st.write("**Últimos 12 meses:**")
        stats_1y = data['Close'].last('1Y').describe()
        st.dataframe(stats_1y, use_container_width=True)


def show_prediction_page(data, api_status):
    """Página de predições."""
    st.header("🔮 Fazer Predição")
    
    if not api_status:
        st.error("A API não está rodando. Inicie-a primeiro!")
        st.code("uvicorn src.api.main:app --reload")
        return
    
    # Controles
    col1, col2 = st.columns([2, 1])
    
    with col1:
        days = st.slider(
            "Número de dias para prever:",
            min_value=1,
            max_value=30,
            value=5,
            help="Selecione quantos dias à frente você quer prever"
        )
    
    with col2:
        st.write("")
        st.write("")
        predict_button = st.button("🔮 Fazer Predição", type="primary")
    
    if predict_button:
        with st.spinner('Fazendo predição...'):
            result = make_prediction(days)
        
        if result:
            st.success("✅ Predição concluída!")
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Último Preço Real",
                    f"R$ {result['last_price']:.2f}"
                )
            
            with col2:
                first_pred = result['predictions'][0]
                change = first_pred - result['last_price']
                change_pct = (change / result['last_price']) * 100
                st.metric(
                    "Predição (D+1)",
                    f"R$ {first_pred:.2f}",
                    f"{change_pct:+.2f}%"
                )
            
            with col3:
                last_pred = result['predictions'][-1]
                total_change = last_pred - result['last_price']
                total_change_pct = (total_change / result['last_price']) * 100
                st.metric(
                    f"Predição (D+{days})",
                    f"R$ {last_pred:.2f}",
                    f"{total_change_pct:+.2f}%"
                )
            
            with col4:
                st.metric(
                    "Confiança",
                    result['confidence'].upper()
                )
            
            # Gráfico
            st.subheader("📈 Visualização das Predições")
            fig = plot_prediction(data, result)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de predições
            st.subheader("📋 Detalhes das Predições")
            pred_df = pd.DataFrame({
                'Data': result['dates'],
                'Preço Previsto (R$)': [f"{p:.2f}" for p in result['predictions']],
                'Variação (%)': [
                    f"{((p - result['last_price']) / result['last_price'] * 100):+.2f}%"
                    for p in result['predictions']
                ]
            })
            st.dataframe(pred_df, use_container_width=True)


def show_technical_analysis(data, feature_data):
    """Página de análise técnica."""
    st.header("📊 Análise Técnica")
    
    if feature_data is None:
        st.warning("Dados de features não disponíveis. Execute o preprocessamento primeiro.")
        return
    
    # Seletor de período
    period = st.selectbox(
        "Selecione o período:",
        ['1M', '3M', '6M', '1Y'],
        index=2
    )
    
    # Filtrar dados
    if period == '1M':
        data_filtered = feature_data.last('1M')
    elif period == '3M':
        data_filtered = feature_data.last('3M')
    elif period == '6M':
        data_filtered = feature_data.last('6M')
    else:
        data_filtered = feature_data.last('1Y')
    
    # Gráficos de features
    fig = plot_features(data_filtered)
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise atual
    st.subheader("📍 Situação Atual")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'RSI' in feature_data.columns:
            current_rsi = feature_data['RSI'].iloc[-1]
            st.metric("RSI Atual", f"{current_rsi:.2f}")
            
            if current_rsi > 70:
                st.warning("⚠️ Sobrecomprado")
            elif current_rsi < 30:
                st.info("ℹ️ Sobrevendido")
            else:
                st.success("✅ Neutro")
    
    with col2:
        if 'MACD' in feature_data.columns:
            current_macd = feature_data['MACD'].iloc[-1]
            st.metric("MACD Atual", f"{current_macd:.4f}")
            
            if current_macd > 0:
                st.success("📈 Tendência de Alta")
            else:
                st.error("📉 Tendência de Baixa")
    
    with col3:
        if 'Volatility' in feature_data.columns:
            current_vol = feature_data['Volatility'].iloc[-1]
            st.metric("Volatilidade", f"{current_vol:.4f}")


def show_backtesting_page():
    """Página de backtesting."""
    st.header("📈 Backtesting")
    
    st.info("Execute o script de backtesting para gerar resultados:")
    st.code("python src/models/backtesting.py")
    
    # Tentar carregar resultados
    results_path = Path("logs/backtest_results.json")
    
    if results_path.exists():
        import json
        
        with open(results_path) as f:
            results = json.load(f)
        
        st.success("✅ Resultados de backtesting carregados!")
        
        # Sumário
        st.subheader("📊 Resumo")
        summary = results['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Iterações",
                summary['total_iterations']
            )
        
        with col2:
            st.metric(
                "MAE Médio",
                f"R$ {summary['metrics']['mae']['mean']:.2f}"
            )
        
        with col3:
            st.metric(
                "MAPE Médio",
                f"{summary['metrics']['mape']['mean']:.2f}%"
            )
        
        with col4:
            st.metric(
                "Acurácia Direcional",
                f"{summary['metrics']['directional_accuracy']['mean']:.1f}%"
            )
        
        # Visualização
        viz_path = Path("logs/backtest_visualization.png")
        if viz_path.exists():
            st.subheader("📊 Visualizações")
            st.image(str(viz_path))
        
        # Tabela de resultados
        st.subheader("📋 Resultados Detalhados")
        df = pd.DataFrame(results['detailed_results'])
        st.dataframe(df, use_container_width=True)
    
    else:
        st.warning("Nenhum resultado de backtesting encontrado ainda.")


def show_feature_importance_page():
    """Página de feature importance."""
    st.header("🎯 Feature Importance")
    
    st.info("Execute o script de explainability para gerar resultados:")
    st.code("python src/models/explainability.py")
    
    # Tentar carregar resultados
    results_path = Path("logs/feature_importance.json")
    
    if results_path.exists():
        import json
        
        with open(results_path) as f:
            results = json.load(f)
        
        st.success("✅ Análise de feature importance carregada!")
        
        # Top features
        st.subheader("🏆 Features Mais Importantes")
        
        col1, col2, col3 = st.columns(3)
        
        top_features = results['top_5_features']
        
        for i, feature in enumerate(top_features[:3]):
            with [col1, col2, col3][i]:
                st.metric(f"#{i+1}", feature)
        
        # Gráfico
        viz_path = Path("logs/feature_importance.png")
        if viz_path.exists():
            st.subheader("📊 Importância das Features")
            st.image(str(viz_path))
        
        # Summary plot
        summary_path = Path("logs/shap_summary.png")
        if summary_path.exists():
            st.subheader("📊 SHAP Summary Plot")
            st.image(str(summary_path))
        
        # Tabela
        st.subheader("📋 Detalhes")
        df = pd.DataFrame(results['feature_importance'])
        st.dataframe(df, use_container_width=True)
        
        # Insights
        st.subheader("💡 Insights")
        st.write(f"**Feature mais importante:** {results['summary']['most_important']}")
        st.write(f"**Top 3 features representam:** {results['summary']['top_3_importance']:.1f}% da importância total")
    
    else:
        st.warning("Nenhum resultado de feature importance encontrado ainda.")


# Rodapé
def show_footer():
    """Mostra rodapé."""
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>📊 LSTM Stock Predictor Dashboard</p>
            <p>Desenvolvido como parte do projeto de Machine Learning Engineering - FIAP</p>
            <p>Dados: Petrobras (PETR4.SA) via yfinance</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    show_footer()
