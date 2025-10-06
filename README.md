# Tech Challenge - Fase 03
## Modelo de Machine Learning para Predição de Valor de Jogadores FIFA 26

### 📋 Sobre o Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge da Fase 03** da Pós-Tech em Machine Learning e implementa um sistema completo de predição do valor de mercado de jogadores do FIFA 26. O sistema é composto por uma API REST desenvolvida em FastAPI que serve um modelo XGBoost treinado para fazer predições em tempo real, além de um dashboard interativo para visualização dos resultados.

### 🎯 Contexto e Objetivos

O **Tech Challenge** é um projeto integrador que engloba os conhecimentos obtidos em todas as disciplinas da fase, representando 90% da nota final. Este projeto específico atende aos seguintes requisitos:

#### 📌 Requisitos Atendidos
- ✅ **Construção de API**: API REST que coleta/processa dados e utiliza modelo de ML
- ✅ **Modelo de Machine Learning**: XGBoost para predição de valores de jogadores
- ✅ **Base de Dados**: [Dataset FIFA 26](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data) com informações detalhadas de jogadores
- ✅ **Código no GitHub**: Repositório completo com documentação técnica
- ✅ **Aplicação Produtiva**: Dashboard interativo que consome o modelo
- ✅ **Documentação Completa**: Storytelling técnico de todo o pipeline

#### 🎮 Problema Resolvido
O projeto resolve o problema de **avaliação automatizada de jogadores de futebol** utilizando características como idade, overall, potencial, salário, clube e nacionalidade para predizer o valor de mercado em euros. Isso é útil para:

- 📊 **Clubes**: Avaliação de transferências e negociações
- 🎯 **Scouts**: Identificação de talentos sub/supervalorizados  
- 📈 **Analistas**: Estudos de mercado e tendências
- 🎮 **Gamers**: Otimização de estratégias no FIFA

### 🔬 Metodologia Técnica

O projeto segue uma metodologia robusta de ciência de dados:

1. **📥 Coleta de Dados**: Dataset FIFA 26 com milhares de jogadores
2. **🔍 Análise Exploratória**: Compreensão das distribuições e correlações
3. **🛠️ Preprocessing**: Pipeline automatizado de limpeza e transformação
4. **🤖 Modelagem**: XGBoost com validação cruzada e otimização
5. **📊 Validação**: Métricas MAE, RMSE e R² para avaliação
6. **🚀 Deploy**: API FastAPI + Dashboard Streamlit em produção
7. **📋 Monitoramento**: Histórico de predições e análises comparativas

### 🏗️ Arquitetura do Sistema

```
├── app/                    # API FastAPI
│   └── main.py            # Aplicação principal
├── ml/                    # Pipeline de Machine Learning
│   ├── train.py          # Script de treinamento
│   └── models/           # Modelos salvos
│       └── fc26_value_model.joblib
├── dashboard/             # Dashboard Streamlit
│   ├── app.py            # Interface web interativa
│   └── requirements.txt  # Dependências do dashboard
├── data/                  # Dados de treinamento
│   └── fc26_players.csv  # Dataset de jogadores FIFA 26
├── docs/                  # Documentação e exemplos
│   └── http/
│       └── predict.http  # Exemplos de requisições
├── requirements.txt       # Dependências principais do projeto
├── Dockerfile            # Container da aplicação
└── docker-compose.yaml   # Orquestração
```

### 🚀 Como Executar

#### Pré-requisitos
- Python 3.10+
- Docker (opcional)

#### Instalação Local

1. **Clone o repositório**
```bash
git clone <url-do-repositorio>
cd postechml-tech-challenge-fase-03
```

2. **Crie um ambiente virtual**
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# ou
.venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Treine o modelo** (se necessário)
```bash
cd ml
python train.py
```

5. **Execute a API**
```bash
cd app
python main.py
```

A API estará disponível em `http://127.0.0.1:8000`

### 🎨 Dashboard Interativo

O projeto inclui um **dashboard web interativo** desenvolvido em Streamlit que consome a API de predição, atendendo ao requisito de "alimentar uma aplicação simples ou dashboard".

#### Funcionalidades do Dashboard

🔧 **Interface Intuitiva**
- Sidebar com controles para entrada de dados
- Sliders para idade, overall e potencial
- Seletores para posição e nacionalidade
- Input numérico para salário

📊 **Visualizações Avançadas**
- **Métricas Principais**: Valor predito, overall e salário
- **Gráfico Radar**: Perfil completo do jogador
- **Análise Comparativa**: Como o overall afeta o valor
- **Histórico de Predições**: Registro das consultas realizadas

🎯 **Recursos Especiais**
- Predição em tempo real via API
- Comparação automática de diferentes cenários
- Interface responsiva e moderna
- Tratamento de erros de conexão

#### Como Usar o Dashboard

1. **Inicie a API** (terminal 1):
```bash
cd app
python main.py
```

2. **Execute o dashboard** (terminal 2):
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

3. **Acesse**: `http://localhost:8501`

4. **Configure** os parâmetros do jogador na sidebar

5. **Clique** em "🔮 Predizer Valor" para ver os resultados

#### Tecnologias do Dashboard
- **Streamlit**: Framework principal para interface web
- **Plotly**: Gráficos interativos (radar e linha)
- **Requests**: Comunicação com a API FastAPI
- **Pandas**: Manipulação de dados do histórico

### 🤖 Modelo de Machine Learning

#### Características do Modelo
- **Algoritmo**: XGBoost Regressor
- **Tipo**: Regressão para predição de valores contínuos
- **Validação**: Cross-validation com 5 folds
- **Métricas**: MAE, RMSE, R²

#### Features Utilizadas

**Numéricas:**
- `age`: Idade do jogador
- `overall`: Rating geral do jogador
- `potential`: Potencial máximo do jogador
- `wage_eur`: Salário em euros

**Categóricas:**
- `club_name`: Nome do clube
- `club_position`: Posição no clube
- `nationality_name`: Nacionalidade

#### Pipeline de Preprocessing
1. **Imputação**: Mediana para numéricas, "unknown" para categóricas
2. **Normalização**: StandardScaler para features numéricas
3. **Encoding**: OneHotEncoder para variáveis categóricas
4. **Tratamento de valores ausentes**: SimpleImputer

### 📡 API Endpoints

#### POST `/predict`
Realiza predição do valor de mercado de um jogador.

**Request Body:**
```json
{
  "age": 21,
  "overall": 70,
  "potential": 85,
  "wage_eur": 50000,
  "club_name": "FC Example",
  "club_position": "ST",
  "nationality_name": "Brazil"
}
```

**Response:**
```json
{
  "predicted_value_eur": 15750000.0
}
```

### 📊 Exemplo de Uso

Você pode testar a API usando o arquivo de exemplo em `docs/http/predict.http` ou fazendo uma requisição direta:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 21,
       "overall": 70,
       "potential": 85,
       "wage_eur": 50000,
       "club_name": "FC Example",
       "club_position": "ST",
       "nationality_name": "Brazil"
     }'
```

### 🔍 Interpretabilidade do Modelo

O projeto utiliza **SHAP (SHapley Additive exPlanations)** para fornecer explicações sobre as predições do modelo, permitindo entender quais features mais influenciam no valor predito de um jogador.

### 📈 Métricas de Performance

O modelo é avaliado usando:
- **MAE (Mean Absolute Error)**: Erro médio absoluto
- **RMSE (Root Mean Square Error)**: Raiz do erro quadrático médio
- **R² (Coefficient of Determination)**: Coeficiente de determinação

### 🛠️ Tecnologias Utilizadas

- **Python 3.10+**: Linguagem principal
- **FastAPI**: Framework web para API REST
- **XGBoost**: Algoritmo de machine learning
- **Scikit-learn**: Pipeline de preprocessing
- **Pandas**: Manipulação de dados (compatível com Python 3.13)
- **SHAP**: Interpretabilidade do modelo
- **Joblib**: Serialização do modelo
- **Streamlit**: Dashboard interativo
- **Plotly**: Visualizações interativas
- **Docker**: Containerização
- **Uvicorn**: Servidor ASGI

### 📝 Dataset

O projeto utiliza o dataset `fc26_players.csv` contendo informações de jogadores do FIFA 26, incluindo:
- **Dados demográficos**: Idade, nacionalidade
- **Estatísticas do jogo**: Overall, potential
- **Informações contratuais**: Salário, clube, posição
- **Valor de mercado**: Target para predição

### 🔄 Pipeline de Desenvolvimento

1. **Coleta de Dados**: Dataset FIFA 26 com características dos jogadores
2. **Análise Exploratória**: Compreensão das distribuições e correlações
3. **Preprocessing**: Limpeza, imputação e transformação de dados
4. **Treinamento**: XGBoost com validação cruzada 5-fold
5. **Validação**: Avaliação com métricas MAE, RMSE e R²
6. **Deploy**: API FastAPI servindo o modelo treinado
7. **Interface**: Dashboard Streamlit para interação visual
8. **Monitoramento**: Histórico de predições e análises comparativas

### 📋 Valor Entregue

Este projeto demonstra competências essenciais em Machine Learning:

#### 🎯 **Competências Técnicas**
- Desenvolvimento de pipeline ML end-to-end
- Criação de APIs REST para modelos de ML
- Construção de interfaces web interativas
- Validação rigorosa de modelos preditivos
- Interpretabilidade e explicabilidade de IA

#### 📊 **Competências de Negócio**
- Resolução de problema real do mercado esportivo
- Análise de dados para tomada de decisão
- Comunicação técnica através de dashboards
- Documentação profissional de projetos

### 🚀 Próximos Passos

- [ ] Implementação de monitoramento contínuo do modelo
- [ ] Dashboard avançado com mais análises estatísticas
- [ ] Retreinamento automático com novos dados
- [ ] Testes unitários e de integração
- [ ] Pipeline CI/CD para deploy automatizado
- [ ] Coleta de dados em tempo real via APIs externas
- [ ] Modelo ensemble com múltiplos algoritmos

### 👥 Contribuições

Este projeto foi desenvolvido como parte do **Tech Challenge da Pós-Tech em Machine Learning**, demonstrando a aplicação prática dos conceitos aprendidos durante o curso em um caso de uso real e relevante.

### 📄 Licença

Este projeto é desenvolvido para fins educacionais como parte do programa de Pós-Graduação em Machine Learning.

---

## 📋 Resumo Executivo

O **FIFA 26 Player Value Predictor** é uma solução completa de Machine Learning que resolve o problema de avaliação automatizada de jogadores de futebol. Utilizando um modelo XGBoost treinado com dados do FIFA 26, o sistema oferece predições precisas através de uma API REST e interface web intuitiva.

**Diferenciais do Projeto:**
- ✅ **Pipeline ML Completo**: Da coleta à produção
- ✅ **Interface Moderna**: Dashboard interativo e responsivo  
- ✅ **Arquitetura Escalável**: API REST + Frontend separados
- ✅ **Qualidade Técnica**: Validação rigorosa e documentação completa
- ✅ **Aplicabilidade Real**: Solução para mercado esportivo

Este projeto exemplifica como técnicas de Machine Learning podem ser aplicadas para resolver problemas reais, criando valor tanto técnico quanto de negócio através de uma solução robusta e bem documentada.
