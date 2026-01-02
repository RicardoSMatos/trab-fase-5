# Sistema de Previsão de Ações com LSTM e MLOps
**Machine Learning Engineering - Fase 5**  
**Pós-Graduação em Machine Learning Engineering**

---

## 🌐 Acesso à Aplicação

**A aplicação está disponível em: https://fiap-fase5.rmnegocios.com**

---

## 📋 Resumo do Projeto

Este projeto implementa um sistema completo de previsão de preços de ações da bolsa de valores brasileira (B3) utilizando redes neurais LSTM (Long Short-Term Memory), seguindo as melhores práticas de MLOps para deploy, monitoramento e manutenção de modelos em produção.

### 🎯 Objetivo

Desenvolver um modelo preditivo capaz de prever o preço de fechamento de ações listadas na B3, com deploy completo utilizando estratégias de MLOps, incluindo API REST, containerização, monitoramento e documentação.

### 🏢 Empresa Selecionada

**Petrobras (PETR4.SA)** - Maior empresa de energia do Brasil, com alta liquidez e volume de negociações, ideal para análise de séries temporais financeiras.

Além disso, o sistema suporta previsões para múltiplas ações simultaneamente através do módulo de gerenciamento de portfólio.

---

## 🚀 Tecnologias Utilizadas

### Machine Learning
- **TensorFlow/Keras**: Framework para construção e treinamento do modelo LSTM
- **Scikit-learn**: Pré-processamento e normalização dos dados
- **yFinance**: Coleta de dados históricos da bolsa de valores

### MLOps & Deploy
- **FastAPI**: API REST para servir o modelo em produção
- **Docker**: Containerização da aplicação
- **Docker Compose**: Orquestração de múltiplos serviços
- **Nginx**: Reverse proxy para produção
- **Uvicorn**: Servidor ASGI de alta performance

### Monitoramento & Observabilidade
- **Sistema de Logging**: Rastreamento de predições e métricas
- **Métricas Personalizadas**: Monitoramento de performance do modelo
- **Streamlit Dashboard**: Visualização interativa de resultados

### Qualidade de Código
- **Type Hints**: Tipagem estática para maior confiabilidade
- **Pydantic**: Validação de dados de entrada/saída
- **Estrutura Modular**: Separação clara de responsabilidades

---

## 🧠 Por Que LSTM?

A escolha do **LSTM (Long Short-Term Memory)** foi fundamentada nas características específicas de séries temporais financeiras:

1. **Memória de Longo Prazo**: Captura dependências temporais complexas em dados financeiros
2. **Tratamento de Não-Linearidade**: Modela padrões não-lineares presentes em séries temporais
3. **Resistência ao Vanishing Gradient**: Arquitetura projetada para aprender relações de longo prazo
4. **Performance Comprovada**: Amplamente utilizado em previsão de preços de ações

**Alternativas Consideradas:**
- ARIMA: Limitado para relações não-lineares
- Prophet: Melhor para dados com sazonalidade forte
- Transformers: Exigem maior volume de dados

---

## 📊 Arquitetura do Modelo

### Rede Neural LSTM

```
Camada de Entrada → LSTM (50 unidades) → Dropout (0.2) → 
LSTM (50 unidades) → Dropout (0.2) → 
Densa (25 unidades) → Densa (1 unidade - saída)
```

### Hiperparâmetros Principais
- **Janela Temporal**: 60 dias históricos para previsão
- **Épocas**: 50 com early stopping
- **Batch Size**: 32
- **Otimizador**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Dropout**: 0.2 para evitar overfitting

### Features Utilizadas
- Preço de Fechamento (Close)
- Médias Móveis (7, 21, 50 dias)
- Volatilidade
- Retorno Diário
- Volume Normalizado

---

## 📈 Resultados e Métricas

### Performance do Modelo
- **RMSE (Root Mean Squared Error)**: ~2.5% do valor médio
- **MAE (Mean Absolute Error)**: ~1.8% do valor médio
- **R² Score**: > 0.90 em dados de teste

### Capacidades do Sistema
✅ Previsão de preço de fechamento com alta acurácia  
✅ Suporte a múltiplas ações simultaneamente  
✅ API REST com validação automática de entrada  
✅ Monitoramento de performance em tempo real  
✅ Sistema de logging completo  
✅ Deploy containerizado pronto para produção  

---

## 🛠️ Como Executar o Projeto

### Pré-requisitos
- Docker e Docker Compose instalados
- Ou Python 3.9+ com pip

### Opção 1: Execução com Docker (Recomendado)

```bash
# 1. Build e iniciar os containers
docker-compose up --build

# A API estará disponível em:
# - API: http://localhost:8000
# - Documentação: http://localhost:8000/docs
# - Dashboard: http://localhost:8501
```

### Opção 2: Execução Local

```bash
# 1. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Iniciar a API
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 4. Em outro terminal, iniciar o dashboard (opcional)
streamlit run app_streamlit.py
```

---

## 🔌 Uso da API

### Fazer uma Previsão

**Endpoint:** `POST /predict`

```bash
# Exemplo com curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "PETR4.SA",
    "days": 5
  }'
```

**Resposta:**
```json
{
  "ticker": "PETR4.SA",
  "predictions": [42.35, 42.78, 43.12, 43.05, 43.45],
  "dates": ["2026-01-03", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"],
  "current_price": 42.10,
  "model_version": "1.0.0",
  "timestamp": "2026-01-02T10:30:00"
}
```

### Gerenciamento de Portfólio

**Endpoint:** `POST /portfolio/predict`

```bash
curl -X POST "http://localhost:8000/portfolio/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["PETR4.SA", "VALE3.SA", "BBAS3.SA"],
    "days": 3
  }'
```

### Verificar Status

```bash
curl http://localhost:8000/health
```

### Documentação Interativa

Acesse `http://localhost:8000/docs` para a documentação completa da API com interface Swagger.

---

## 📁 Estrutura do Projeto

```
.
├── src/
│   ├── api/              # API FastAPI
│   │   ├── main.py       # Endpoints e configuração
│   │   └── schemas.py    # Modelos Pydantic
│   ├── data/             # Coleta e pré-processamento
│   │   ├── collect_data.py
│   │   └── preprocessing.py
│   ├── models/           # Modelo LSTM
│   │   ├── lstm_model.py
│   │   ├── backtesting.py
│   │   └── explainability.py
│   ├── monitoring/       # Logging e métricas
│   │   ├── metrics.py
│   │   └── logging_config.py
│   └── portfolio/        # Sistema multi-ações
│       ├── portfolio_manager.py
│       └── multi_stock_predictor.py
├── models/               # Modelos treinados (.keras)
├── data/                 # Dados históricos e processados
├── logs/                 # Logs de execução e métricas
├── docker-compose.yml    # Orquestração de containers
├── Dockerfile           # Imagem Docker da aplicação
├── requirements.txt     # Dependências Python
└── README.md           # Este arquivo
```

---

## 🔍 Estratégia de MLOps Implementada

### 1. **Versionamento**
- ✅ Código versionado no Git
- ✅ Metadados do modelo salvos (arquitetura, hiperparâmetros)
- ✅ Registro de versões de treinamento

### 2. **Serialização do Modelo**
- ✅ Modelo salvo em formato `.keras` (nativo do TensorFlow)
- ✅ Metadados em JSON para rastreabilidade
- ✅ Checkpoint do melhor modelo durante treinamento

### 3. **Containerização**
- ✅ Dockerfile multi-stage para otimização
- ✅ Docker Compose para orquestração
- ✅ Ambiente isolado e reproduzível
- ✅ Variáveis de ambiente para configuração

### 4. **API REST**
- ✅ FastAPI com validação automática (Pydantic)
- ✅ Documentação automática (Swagger/OpenAPI)
- ✅ Endpoints para previsão e health check
- ✅ Tratamento de erros robusto

### 5. **Monitoramento**
- ✅ Logging estruturado de todas as predições
- ✅ Métricas de performance registradas
- ✅ Rastreamento de tempo de inferência
- ✅ Dashboard para visualização de resultados

### 6. **CI/CD Ready**
- ✅ Estrutura pronta para integração contínua
- ✅ Testes automatizados
- ✅ Build automatizado via Docker
- ✅ Configuração para deploy em cloud

### 7. **Documentação**
- ✅ README completo
- ✅ Docstrings em todas as funções
- ✅ Documentação da API (Swagger)
- ✅ Comentários explicativos no código

---

## 📊 Monitoramento em Produção

O sistema implementa monitoramento completo através de:

1. **Logging de Predições**: Cada predição é registrada com timestamp, entrada e saída
2. **Métricas de Performance**: RMSE, MAE e tempo de inferência
3. **Health Checks**: Endpoint para verificação de disponibilidade
4. **Dashboard Streamlit**: Visualização interativa de predições e análises

### Acessar Logs

```bash
# Logs da API
docker-compose logs api

# Métricas salvas
cat logs/metrics.json
```

---

## 🎓 Aprendizados e Desafios

### Principais Aprendizados
1. **Séries Temporais Financeiras**: Importância da normalização e feature engineering
2. **LSTM**: Ajuste fino de hiperparâmetros para convergência
3. **MLOps**: Integração de modelo com infraestrutura de produção
4. **API Design**: Criação de interfaces robustas e documentadas
5. **Docker**: Containerização e orquestração de serviços

### Desafios Superados
- Tratamento de dados faltantes em séries temporais
- Prevenção de overfitting com dropout e early stopping
- Balanceamento entre complexidade do modelo e tempo de inferência
- Implementação de sistema multi-ações escalável

---

## 🔮 Melhorias Futuras

- [ ] Implementar retreinamento automático (auto-retraining)
- [ ] Adicionar mais features técnicas (RSI, MACD, Bandas de Bollinger)
- [ ] Integrar com banco de dados para histórico de predições
- [ ] Implementar A/B testing de modelos
- [ ] Deploy em plataforma cloud (AWS/GCP/Azure)
- [ ] Adicionar autenticação JWT na API
- [ ] Implementar modelo ensemble combinando LSTM com outros algoritmos

---

## 📚 Referências

- **TensorFlow Documentation**: https://www.tensorflow.org/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Time Series Forecasting with LSTM**: Papers e tutoriais sobre aplicação em finanças
- **MLOps Principles**: Best practices para deploy de modelos de ML
- **yFinance Library**: Documentação para coleta de dados financeiros

---

## 👨‍🎓 Autor

**Estudante de Pós-Graduação em Machine Learning Engineering**  
FIAP - Fase 5  
Projeto de Machine Learning Engineering  
Janeiro de 2026

---

## 📝 Requisitos Atendidos

✅ Escolha de empresa listada na B3 (PETR4.SA)  
✅ Coleta de dados históricos via yFinance  
✅ Algoritmo de ML para séries temporais (LSTM)  
✅ Avaliação com métricas relevantes (RMSE, MAE, R²)  
✅ Serialização do modelo (.keras)  
✅ Ambiente virtualizado (Docker + requirements.txt)  
✅ API REST para predições (FastAPI)  
✅ Monitoramento em produção (Logging + Métricas)  
✅ Documentação completa do projeto  
✅ Repositório GitHub organizado  

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do programa de pós-graduação em Machine Learning Engineering.

---

**🚀 Projeto pronto para produção e avaliação!**
