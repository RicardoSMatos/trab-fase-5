# 🚀 Guia Rápido de Execução

## Opção 1: Docker (Recomendado) ⚡

```bash
# 1. Build e iniciar todos os serviços
docker-compose up --build

# Ou em modo detached (background)
docker-compose up -d --build

# 2. Acessar:
# - API: http://localhost:8000
# - Documentação: http://localhost:8000/docs
# - Dashboard: http://localhost:8501
# - Portfolio Manager: http://localhost:8502
```

## Opção 2: Local (Python) 🐍

```bash
# 1. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Iniciar API
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 4. Em outro terminal, iniciar dashboards
streamlit run app_streamlit.py --server.port 8501
streamlit run app_portfolio.py --server.port 8502
```

## 🧪 Testar API

```bash
# Health check
curl http://localhost:8000/health

# Fazer predição
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA", "days": 5}'

# Ou usar o script de teste
python test_api.py
```

## 🛑 Parar e Limpar

```bash
# Parar containers
docker-compose down

# Parar e remover volumes
docker-compose down -v

# Remover tudo (incluindo imagens)
docker-compose down -v --rmi all
```

## 📊 Estrutura de Arquivos Importante

```
.
├── src/                    # Código fonte
│   ├── api/               # FastAPI
│   ├── data/              # Coleta e processamento
│   ├── models/            # Modelo LSTM
│   ├── monitoring/        # Logs e métricas
│   └── portfolio/         # Sistema multi-ações
├── models/                # Modelos treinados (.keras)
├── data/                  # Dados históricos
├── docker-compose.yml     # Orquestração Docker
├── Dockerfile            # Imagem Docker
├── requirements.txt      # Dependências Python
└── README.md            # Documentação completa
```

## ⚠️ Troubleshooting

**Porta já em uso:**
```bash
# Mudar porta no docker-compose.yml ou:
docker-compose -f docker-compose.yml up --force-recreate
```

**Modelo não encontrado:**
```bash
# Verificar se os modelos estão em models/
ls -la models/
```

**API não responde:**
```bash
# Ver logs
docker-compose logs api
```

## 📖 Documentação Completa

Veja [README.md](README.md) para documentação detalhada do projeto.
