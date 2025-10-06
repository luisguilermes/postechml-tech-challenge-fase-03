import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Configuração da página
st.set_page_config(
    page_title="FIFA 26 - Preditor de Valor de Jogadores",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL da API
API_URL = "http://127.0.0.1:8000/predict"

# Título principal
st.title("⚽ FIFA 26 - Dashboard de Predição de Valor de Jogadores")
st.markdown("---")

# Sidebar para entrada de dados
st.sidebar.header("🔧 Configuração do Jogador")

# Inputs do usuário
age = st.sidebar.slider("Idade", min_value=16, max_value=45, value=25, step=1)
overall = st.sidebar.slider("Overall", min_value=40, max_value=99, value=75, step=1)
potential = st.sidebar.slider("Potencial", min_value=40, max_value=99, value=80, step=1)
wage_eur = st.sidebar.number_input("Salário (EUR)", min_value=0, max_value=1000000, value=50000, step=5000)

# Inputs categóricos
club_position = st.sidebar.selectbox("Posição", [
    "ST", "CF", "LW", "RW", "CAM", "CM", "CDM",
    "LM", "RM", "LB", "RB", "CB", "GK"
], index=0)
nationality_name = st.sidebar.selectbox("Nacionalidade", [
    "Brazil", "Argentina", "Spain", "France", "Germany", "England",
    "Portugal", "Italy", "Netherlands", "Belgium", "Croatia", "Poland"
], index=0)

# Botão de predição
predict_button = st.sidebar.button("🔮 Predizer Valor", type="primary")

# Layout principal com colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Predição de Valor")

    if predict_button:
        # Dados para a API
        player_data = {
            "age": age,
            "overall": overall,
            "potential": potential,
            "wage_eur": wage_eur,
            # "club_name": club_name,
            "club_position": club_position,
            "nationality_name": nationality_name
        }

        try:
            # Fazer requisição para a API
            response = requests.post(API_URL, json=player_data)

            if response.status_code == 200:
                result = response.json()
                predicted_value = result["predicted_value_eur"]

                # Exibir resultado principal
                st.success(f"✅ Predição realizada com sucesso!")

                # Informações do jogador predito
                st.markdown(f"""
                ### 🏃‍♂️ Jogador Analisado
                **Posição:** {club_position}  
                **Nacionalidade:** {nationality_name}
                """)

                # Métricas principais
                col_metric1, col_metric2, col_metric3 = st.columns(3)

                with col_metric1:
                    st.metric(
                        label="💰 Valor Predito",
                        value=f"€{predicted_value:,.0f}",
                        delta=None
                    )

                with col_metric2:
                    st.metric(
                        label="🎯 Overall",
                        value=f"{overall}",
                        delta=f"{potential - overall} potencial"
                    )

                with col_metric3:
                    st.metric(
                        label="💼 Salário",
                        value=f"€{wage_eur:,.0f}",
                        delta=None
                    )

                # Gráfico de valor vs características
                fig_radar = go.Figure()

                categories = ['Overall', 'Potencial', 'Idade (norm)', 'Salário (norm)']
                values = [
                    overall,
                    potential,
                    (45 - age) * 2.2,  # Normalizar idade (inverso)
                    min(wage_eur / 10000, 99)  # Normalizar salário
                ]

                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Características do Jogador',
                    line_color='rgb(0, 123, 255)'
                ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="🎯 Perfil do Jogador",
                    title_x=0.5
                )

                st.plotly_chart(fig_radar, use_container_width=True)

                # Análise comparativa
                st.subheader("📈 Análise Comparativa")

                # Simulação de valores para diferentes overall
                overall_range = list(range(60, 95, 5))
                predicted_values = []

                for ovr in overall_range:
                    temp_data = player_data.copy()
                    temp_data["overall"] = ovr
                    temp_response = requests.post(API_URL, json=temp_data)
                    if temp_response.status_code == 200:
                        predicted_values.append(temp_response.json()["predicted_value_eur"])
                    else:
                        predicted_values.append(0)

                # Gráfico de linha
                fig_line = px.line(
                    x=overall_range,
                    y=predicted_values,
                    title="💹 Valor vs Overall (mantendo outras características)",
                    labels={"x": "Overall", "y": "Valor Predito (EUR)"}
                )

                fig_line.add_vline(x=overall, line_dash="dash", line_color="red",
                                  annotation_text="Jogador Atual")

                st.plotly_chart(fig_line, use_container_width=True)

            else:
                st.error(f"❌ Erro na API: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Erro de conexão com a API. Verifique se o servidor está rodando em http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"❌ Erro inesperado: {str(e)}")

with col2:
    st.header("ℹ️ Informações do Jogador")

    # Card com informações do jogador
    st.markdown(f"""
    <div style="
        background-color: #000;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
    ">
        <p><strong>Posição:</strong> {club_position}</p>
        <p><strong>Nacionalidade:</strong> {nationality_name}</p>
        <p><strong>Idade:</strong> {age} anos</p>
        <p><strong>Overall:</strong> {overall}/99</p>
        <p><strong>Potencial:</strong> {potential}/99</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Sobre o Modelo")
    st.info("""
    🤖 **Algoritmo:** XGBoost Regressor
    
    📊 **Features:** Age, Overall, Potential, Wage, Position, Nationality
    
    🎯 **Tipo:** Regressão para valores contínuos
    
    ✅ **Validação:** Cross-validation 5-folds
    """)

    st.markdown("### 🎮 Como Usar")
    st.markdown("""
    1. Ajuste os parâmetros na barra lateral
    2. Clique em "Predizer Valor"
    3. Analise os resultados e gráficos
    4. Compare diferentes cenários
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏆 Tech Challenge - Fase 03 | Pós-Tech Machine Learning</p>
    <p>Desenvolvido com Streamlit + FastAPI + XGBoost</p>
</div>
""", unsafe_allow_html=True)

# Histórico de predições (opcional)
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if predict_button and 'result' in locals():
    # Adicionar ao histórico
    st.session_state.predictions_history.append({
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'overall': overall,
        'predicted_value': predicted_value
    })

    # Mostrar histórico se houver dados
    if len(st.session_state.predictions_history) > 0:
        st.header("📚 Histórico de Predições")
        df_history = pd.DataFrame(st.session_state.predictions_history)
        st.dataframe(df_history, use_container_width=True)
