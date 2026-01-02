# Dockerfile para LSTM Stock Predictor
# Multi-stage build com targets separados para API e Streamlit

# ============================================
# Stage 1: Base builder
# ============================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de requisitos
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --user --quiet -r requirements.txt

# ============================================
# Stage 2: API FastAPI
# ============================================
FROM python:3.12-slim AS api

WORKDIR /app

# Instalar curl para healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiar dependências instaladas
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copiar código-fonte
COPY src/ ./src/
COPY models/ ./models/

# Criar diretórios necessários
RUN mkdir -p logs data/raw data/processed

# Expor porta da API
EXPOSE 8000

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src:/app
ENV YF_FORCE_URLLIB=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para iniciar API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 3: Dashboard Streamlit
# ============================================
FROM python:3.12-slim AS streamlit

WORKDIR /app

# Instalar curl para healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiar dependências instaladas
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copiar código-fonte
COPY app_portfolio.py .
COPY src/ ./src/

# Criar diretórios necessários
RUN mkdir -p data models logs .streamlit

# Criar config padrão do Streamlit se não existir
RUN echo '[server]\nheadless = true\nport = 8501\nenableCORS = false\n[browser]\ngatherUsageStats = false' > .streamlit/config.toml

# Expor porta do Streamlit
EXPOSE 8501

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando para iniciar Streamlit
CMD ["python", "-m", "streamlit", "run", "app_portfolio.py", "--server.port=8501", "--server.address=0.0.0.0"]

