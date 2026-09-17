# -*- coding: utf-8 -*-
"""
Despliegue: Predicción de ataque al corazón
--------------------------------------------
Basado en el modelo KNN entregado por la cátedra, con un módulo adicional
de EXPLICABILIDAD y SIMULACIÓN construido a partir de la propia memoria
del modelo (KNN guarda internamente todos los pacientes de entrenamiento),
sin necesidad de un dataset externo.
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# 1. CARGA DEL MODELO
# =========================================================
FILENAME = "modelo-class.pkl"


@st.cache_resource
def cargar_modelo():
    with open(FILENAME, "rb") as f:
        modelo, labelencoder, variables, min_max_scaler = pickle.load(f)
    return modelo, labelencoder, list(variables), min_max_scaler


modelo, labelencoder, variables, min_max_scaler = cargar_modelo()


def col(substr):
    """Busca de forma robusta el nombre real de una columna dummy,
    sin depender de caracteres raros (comillas, mayúsculas, etc.)."""
    for v in variables:
        if substr in v:
            return v
    raise ValueError(f"No se encontró ninguna columna que contenga: {substr}")


COL_NEVER = col("never smoked")
COL_UNKNOWN = col("Unknown")
COL_SMOKES = col("smokes")
COL_HYP = col("hypertension")
COL_HD = col("heart_disease")
COL_MARRIED = col("ever_married")

# Tasa de riesgo de toda la población que "recuerda" el modelo (KNN memoriza sus datos)
TASA_POBLACIONAL = float(modelo._y.mean())
TOTAL_PACIENTES = modelo._fit_X.shape[0]

# =========================================================
# 2. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# =========================================================
st.set_page_config(
    page_title="Predicción de ataque al corazón",
    page_icon="🫀",
    layout="wide",
)

st.markdown("""
<style>
body { font-family: Arial, sans-serif; }

.titulo {
    font-size: 42px; font-weight: bold; color: #B71C1C;
    text-align: center; margin-bottom: 5px;
}
.subtitulo {
    font-size: 18px; color: #555555; text-align: center; margin-bottom: 30px;
}
.card {
    background-color: #FFFFFF; padding: 25px; border-radius: 18px;
    border: 1px solid #E5E5E5; box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.seccion {
    color: #B71C1C; font-size: 25px; font-weight: bold; margin-bottom: 15px;
}
.resultado {
    background-color: #FFF3F3; border-radius: 18px; padding: 30px;
    text-align: center; border: 2px solid #EF9A9A;
}
.resultado-titulo { color: #B71C1C; font-size: 24px; font-weight: bold; }
.gemelo {
    background-color: #F5F5FF; border-radius: 18px; padding: 20px;
    border: 2px solid #C5CAE9;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="titulo">🫀 Predicción de ataque al corazón</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitulo">Sistema de predicción basado en Machine Learning '
    '+ panel de explicabilidad por vecinos más cercanos</div>',
    unsafe_allow_html=True,
)

# =========================================================
# 3. FUNCIONES DE APOYO (preprocesamiento + explicabilidad)
# =========================================================
def preparar_datos(age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status):
    """Convierte las respuestas del formulario (en español) directamente
    en el vector one-hot + normalizado que espera el modelo."""
    fila = pd.DataFrame([{v: 0 for v in variables}])
    fila["age"] = age
    fila["avg_glucose_level"] = avg_glucose_level

    if hypertension == "Sí":
        fila[COL_HYP] = 1
    if heart_disease == "Sí":
        fila[COL_HD] = 1
    if ever_married == "Sí":
        fila[COL_MARRIED] = 1

    if smoking_status == "Nunca ha fumado":
        fila[COL_NEVER] = 1
    elif smoking_status == "Fuma actualmente":
        fila[COL_SMOKES] = 1
    elif smoking_status == "Desconocido":
        fila[COL_UNKNOWN] = 1
    # "Fumó anteriormente" queda como categoría base (todo en cero)

    fila = fila[variables]
    fila[["age", "avg_glucose_level"]] = min_max_scaler.transform(fila[["age", "avg_glucose_level"]])
    return fila


def score_riesgo_por_vecinos(fila_preparada, k=15):
    """Score de riesgo (0-100%) basado en la proporción de los k pacientes
    más parecidos en la memoria del KNN que sí desarrollaron la condición."""
    k = min(k, TOTAL_PACIENTES)
    distancias, indices = modelo.kneighbors(fila_preparada.values, n_neighbors=k)
    y_vecinos = modelo._y[indices[0]]
    return float(y_vecinos.mean()) * 100, distancias[0], indices[0]


def gemelo_digital(fila_preparada):
    """El paciente real más parecido (k=1) — el mismo que usa el modelo para decidir."""
    dist, idx = modelo.kneighbors(fila_preparada.values, n_neighbors=1)
    vecino = modelo._fit_X[idx[0]]
    clase = modelo._y[idx[0]][0]
    edad_glucosa = min_max_scaler.inverse_transform(vecino[:, :2])[0]
    return {
        "distancia": dist[0][0],
        "clase": labelencoder.inverse_transform([clase])[0],
        "edad": edad_glucosa[0],
        "glucosa": edad_glucosa[1],
        "hipertension": "Sí" if vecino[0, variables.index(COL_HYP)] == 1 else "No",
        "enf_cardiaca": "Sí" if vecino[0, variables.index(COL_HD)] == 1 else "No",
        "casado": "Sí" if vecino[0, variables.index(COL_MARRIED)] == 1 else "No",
    }


def curva_sensibilidad(base_kwargs, variable, rango, k=15):
    """Recalcula el score de riesgo variando UNA sola variable, dejando
    las demás fijas en los valores actuales del formulario."""
    scores = []
    for valor in rango:
        kwargs = dict(base_kwargs)
        kwargs[variable] = valor
        fila = preparar_datos(**kwargs)
        score, _, _ = score_riesgo_por_vecinos(fila, k=k)
        scores.append(score)
    return scores


def gauge_riesgo(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%"},
        title={"text": "Score de riesgo (según pacientes similares)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#B71C1C"},
            "steps": [
                {"range": [0, 33], "color": "#E8F5E9"},
                {"range": [33, 66], "color": "#FFF3E0"},
                {"range": [66, 100], "color": "#FFEBEE"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# =========================================================
# 4. NAVEGACIÓN POR PESTAÑAS
# =========================================================
tab_pred, tab_explica, tab_simula = st.tabs(
    ["🩺 Predicción", "🔎 Explicabilidad", "🎚️ Simulador de riesgo"]
)

# ---------------------------------------------------------
# TAB 1 — PREDICCIÓN
# ---------------------------------------------------------
with tab_pred:
    st.markdown("""
    <div class="card">
    <div class="seccion">¿Cómo funciona?</div>
    <p>El sistema recibe información del paciente, prepara los datos y los procesa
    mediante un modelo de clasificación KNN (K-Vecinos más Cercanos).</p>
    <p>Además del resultado, la pestaña <b>Explicabilidad</b> te muestra
    <i>por qué</i> el modelo decidió eso, comparando al paciente con casos reales
    de su memoria, y <b>Simulador</b> te deja mover variables para ver cómo cambia
    el riesgo.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="seccion">👤 Información del paciente</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad", min_value=1, max_value=120, value=40, step=1)
    with col2:
        avg_glucose_level = st.number_input(
            "Nivel promedio de glucosa", min_value=0.0, max_value=300.0, value=100.0, step=0.1
        )

    col1, col2 = st.columns(2)
    with col1:
        hypertension = st.selectbox("¿Tiene hipertensión?", ["No", "Sí"])
        heart_disease = st.selectbox("¿Tiene enfermedad cardíaca?", ["No", "Sí"])
    with col2:
        ever_married = st.selectbox("¿Se ha casado alguna vez?", ["No", "Sí"])
        smoking_status = st.selectbox(
            "Estado de tabaquismo",
            ["Nunca ha fumado", "Fumó anteriormente", "Fuma actualmente", "Desconocido"],
        )

    datos = pd.DataFrame([[age, hypertension, heart_disease, ever_married, avg_glucose_level, smoking_status]],
                          columns=["age", "hypertension", "heart_disease", "ever_married",
                                   "avg_glucose_level", "smoking_status"])
    st.dataframe(datos, use_container_width=True)

    # --- preprocesamiento + predicción (se reutiliza en las otras pestañas) ---
    fila_preparada = preparar_datos(age, hypertension, heart_disease, ever_married,
                                     avg_glucose_level, smoking_status)
    Y_pred = modelo.predict(fila_preparada.values)
    resultado = labelencoder.inverse_transform(Y_pred)[0]
    score, distancias_k, indices_k = score_riesgo_por_vecinos(fila_preparada)

    color_resultado = "#B71C1C" if resultado == "Yes" else "#2E7D32"
    texto_resultado = "⚠️ Riesgo alto" if resultado == "Yes" else "✅ Riesgo bajo"

    st.markdown(f"""
    <div class="resultado">
    <div class="resultado-titulo">🫀 Resultado</div><br>
    <div style="font-size: 30px; font-weight: bold; color:{color_resultado};">{texto_resultado}</div>
    <p style="color:#777; margin-top:10px;">Predicción directa del modelo: <b>{resultado}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <div class="seccion">🤖 Información del modelo</div>
    <p><b>Modelo:</b> K-Nearest Neighbors (KNN)</p>
    <p><b>Tipo:</b> Clasificación</p>
    <p><b>Preprocesamiento:</b> Variables categóricas (one-hot) + normalización Min-Max</p>
    <p><b>Pacientes en la memoria del modelo:</b> {total}</p>
    <p><b>Tasa de riesgo observada en esos pacientes:</b> {tasa:.1f}%</p>
    </div>
    """.format(total=TOTAL_PACIENTES, tasa=TASA_POBLACIONAL * 100), unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2 — EXPLICABILIDAD (lo distintivo #1)
# ---------------------------------------------------------
with tab_explica:
    st.markdown('<div class="seccion">🔎 ¿Por qué el modelo predijo esto?</div>', unsafe_allow_html=True)
    st.write(
        "Un KNN no tiene 'coeficientes' como una regresión: decide mirando a sus "
        "pacientes más parecidos. Aquí exponemos exactamente esos vecinos, "
        "para que la predicción deje de ser una caja negra."
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.plotly_chart(gauge_riesgo(score), use_container_width=True)
        st.caption(
            f"De los 15 pacientes más parecidos a este, "
            f"{round(score/100*15)} desarrollaron la condición."
        )

    with c2:
        gemelo = gemelo_digital(fila_preparada)
        clase_txt = "⚠️ Sí desarrolló la condición" if gemelo["clase"] == "Yes" else "✅ No la desarrolló"
        st.markdown(f"""
        <div class="gemelo">
        <b>🧑‍⚕️ Gemelo digital (paciente real más parecido)</b><br><br>
        Edad: <b>{gemelo['edad']:.0f} años</b><br>
        Glucosa promedio: <b>{gemelo['glucosa']:.1f}</b><br>
        Hipertensión: <b>{gemelo['hipertension']}</b> &nbsp;|&nbsp;
        Enf. cardíaca: <b>{gemelo['enf_cardiaca']}</b> &nbsp;|&nbsp;
        Casado/a: <b>{gemelo['casado']}</b><br><br>
        Resultado real de ese paciente: <b>{clase_txt}</b><br>
        Distancia (similitud) al paciente ingresado: <b>{gemelo['distancia']:.3f}</b>
        <br><span style="color:#888; font-size:13px;">(0 = idéntico)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="seccion">📊 Paciente vs. promedio poblacional</div>', unsafe_allow_html=True)

    prom_glucosa = float(pd.DataFrame(
        min_max_scaler.inverse_transform(modelo._fit_X[:, :2]), columns=["age", "avg_glucose_level"]
    )["avg_glucose_level"].mean())
    prom_edad = float(pd.DataFrame(
        min_max_scaler.inverse_transform(modelo._fit_X[:, :2]), columns=["age", "avg_glucose_level"]
    )["age"].mean())

    fig_comp = go.Figure(data=[
        go.Bar(name="Paciente", x=["Edad", "Glucosa"], y=[age, avg_glucose_level], marker_color="#B71C1C"),
        go.Bar(name="Promedio poblacional", x=["Edad", "Glucosa"], y=[prom_edad, prom_glucosa], marker_color="#90A4AE"),
    ])
    fig_comp.update_layout(barmode="group", height=350, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------
# TAB 3 — SIMULADOR DE SENSIBILIDAD (lo distintivo #2)
# ---------------------------------------------------------
with tab_simula:
    st.markdown('<div class="seccion">🎚️ ¿Cómo cambia el riesgo si...?</div>', unsafe_allow_html=True)
    st.write(
        "Deja fijas las demás respuestas del formulario y mueve una sola variable "
        "para ver, en vivo, cómo se mueve el score de riesgo del modelo."
    )

    variable_simular = st.radio(
        "Variable a explorar", ["Edad", "Nivel de glucosa"], horizontal=True
    )

    base_kwargs = dict(
        age=age, hypertension=hypertension, heart_disease=heart_disease,
        ever_married=ever_married, avg_glucose_level=avg_glucose_level,
        smoking_status=smoking_status,
    )

    if variable_simular == "Edad":
        rango = np.arange(1, 121, 2)
        scores = curva_sensibilidad(base_kwargs, "age", rango)
        x_actual = age
        titulo_x = "Edad"
    else:
        rango = np.arange(50, 301, 5)
        scores = curva_sensibilidad(base_kwargs, "avg_glucose_level", rango)
        x_actual = avg_glucose_level
        titulo_x = "Nivel de glucosa"

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=rango, y=scores, mode="lines", line=dict(color="#B71C1C", width=3),
                                  name="Score de riesgo"))
    fig_sim.add_vline(x=x_actual, line_dash="dash", line_color="#555",
                       annotation_text="Valor actual", annotation_position="top")
    fig_sim.update_layout(
        xaxis_title=titulo_x, yaxis_title="Score de riesgo (%)",
        height=400, margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.caption(
        "El score se calcula mirando, para cada valor simulado, qué proporción de los "
        "15 pacientes más parecidos de la memoria del modelo desarrollaron la condición."
    )
