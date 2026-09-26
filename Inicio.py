import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

st.set_page_config(
    page_title="Analizador TF-IDF",
    page_icon="✦",
    layout="wide"
)

# ---------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Fondo general */
    .stApp {
        background-color: #F6F3EE;
    }

    /* Contenedor principal */
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Encabezado */
    .header {
        background-color: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-radius: 22px;
        padding: 30px 35px;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(40, 35, 30, 0.05);
    }

    .tag {
        display: inline-block;
        background-color: #EEE9DF;
        color: #625B50;
        padding: 6px 13px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    .header h1 {
        color: #292621;
        font-size: 36px;
        margin: 0;
        font-weight: 700;
    }

    .header p {
        color: #756F66;
        font-size: 15px;
        margin-top: 10px;
        line-height: 1.6;
        max-width: 780px;
    }

    /* Tarjetas */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 5px 18px rgba(40, 35, 30, 0.04);
    }

    .card-title {
        color: #302C27;
        font-size: 19px;
        font-weight: 650;
        margin-bottom: 4px;
    }

    .card-description {
        color: #827B72;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* Labels */
    label {
        color: #403B35 !important;
        font-weight: 600 !important;
    }

    /* Text area */
    textarea {
        background-color: #FBFAF8 !important;
        color: #302C27 !important;
        border: 1px solid #DCD6CC !important;
        border-radius: 13px !important;
    }

    textarea:focus {
        border: 1px solid #9A8F7E !important;
        box-shadow: 0 0 0 1px #9A8F7E !important;
    }

    /* Campo de pregunta */
    input {
        background-color: #FBFAF8 !important;
        color: #302C27 !important;
        border: 1px solid #DCD6CC !important;
        border-radius: 13px !important;
    }

    input:focus {
        border: 1px solid #9A8F7E !important;
        box-shadow: 0 0 0 1px #9A8F7E !important;
    }

    /* Botones */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid #DDD6CB;
        background-color: #FAF8F4;
        color: #403A33;
        font-weight: 550;
        transition: 0.2s;
    }

    .stButton > button:hover {
        border-color: #9B907F;
        color: #292621;
        background-color: #F0ECE5;
    }

    /* Botón principal */
    .stButton > button[kind="primary"] {
        background-color: #3B3732;
        color: white;
        border: none;
        border-radius: 13px;
        padding: 10px 25px;
        font-weight: 650;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #292621;
        color: white;
    }

    /* Preguntas sugeridas */
    .suggestion-title {
        color: #302C27;
        font-size: 18px;
        font-weight: 650;
        margin-bottom: 5px;
    }

    .suggestion-description {
        color: #827B72;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    /* Resultado */
    .answer-box {
        background-color: #F3EFE7;
        border-left: 4px solid #8B806F;
        border-radius: 12px;
        padding: 18px 20px;
        color: #35312C;
        margin-top: 10px;
    }

    .answer-label {
        color: #756B5D;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 5px;
    }

    .answer-text {
        color: #302C27;
        font-size: 16px;
        line-height: 1.5;
    }

    /* Métrica */
    .metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E0D8;
        border-radius: 15px;
        padding: 15px 18px;
        text-align: center;
        margin-top: 15px;
    }

    .metric-label {
        color: #827B72;
        font-size: 12px;
    }

    .metric-value {
        color: #302C27;
        font-size: 25px;
        font-weight: 700;
        margin-top: 3px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Separador */
    hr {
        border: none;
        border-top: 1px solid #E4DED4;
        margin: 28px 0;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------

st.markdown("""
<div class="header">

    <div class="tag">PROCESAMIENTO DE LENGUAJE NATURAL</div>

    <h1>Analizador de similitud textual</h1>

    <p>
        Explora cómo TF-IDF identifica relaciones entre una pregunta
        y diferentes documentos en español. La herramienta transforma
        los textos en vectores y calcula cuál contiene la información
        más relacionada con tu consulta.
    </p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DOCUMENTOS DE EJEMPLO
# ---------------------------------------------------------

default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""


# ---------------------------------------------------------
# STEMMER EN ESPAÑOL
# ---------------------------------------------------------

stemmer = SnowballStemmer("spanish")


def tokenize_and_stem(text):

    # Minúsculas
    text = text.lower()

    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)

    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]

    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]

    return stems


# ---------------------------------------------------------
# LAYOUT PRINCIPAL
# ---------------------------------------------------------

col1, col2 = st.columns([2.2, 1], gap="large")


# ---------------------------------------------------------
# COLUMNA IZQUIERDA
# ---------------------------------------------------------

with col1:

    st.markdown("""
    <div class="card">

        <div class="card-title">
            Documentos de análisis
        </div>

        <div class="card-description">
            Escribe un documento por línea. Puedes modificar los textos
            para experimentar con diferentes resultados.
        </div>

    </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "Documentos",
        default_docs,
        height=180,
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="height: 8px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">

        <div class="card-title">
            Consulta
        </div>

        <div class="card-description">
            Escribe una pregunta relacionada con alguno de los documentos.
        </div>

    </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "Pregunta",
        "¿Dónde juegan el perro y el gato?",
        label_visibility="collapsed"
    )


# ---------------------------------------------------------
# COLUMNA DERECHA
# ---------------------------------------------------------

with col2:

    st.markdown("""
    <div class="card">

        <div class="suggestion-title">
            Explora una consulta
        </div>

        <div class="suggestion-description">
            Selecciona una de estas preguntas para probar
            rápidamente el funcionamiento del modelo.
        </div>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "¿Dónde juegan el perro y el gato?",
        use_container_width=True
    ):
        st.session_state.question = "¿Dónde juegan el perro y el gato?"
        st.rerun()

    if st.button(
        "¿Qué hacen los niños en el parque?",
        use_container_width=True
    ):
        st.session_state.question = "¿Qué hacen los niños en el parque?"
        st.rerun()

    if st.button(
        "¿Cuándo cantan los pájaros?",
        use_container_width=True
    ):
        st.session_state.question = "¿Cuándo cantan los pájaros?"
        st.rerun()

    if st.button(
        "¿Dónde suena la música alta?",
        use_container_width=True
    ):
        st.session_state.question = "¿Dónde suena la música alta?"
        st.rerun()

    if st.button(
        "¿Qué animal maúlla durante la noche?",
        use_container_width=True
    ):
        st.session_state.question = "¿Qué animal maúlla durante la noche?"
        st.rerun()


# ---------------------------------------------------------
# ACTUALIZAR PREGUNTA
# ---------------------------------------------------------

if "question" in st.session_state:
    question = st.session_state.question


# ---------------------------------------------------------
# BOTÓN ANALIZAR
# ---------------------------------------------------------

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

_, button_col, _ = st.columns([1, 1.2, 1])

with button_col:

    analizar = st.button(
        "Analizar texto",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------
# PROCESAMIENTO
# ---------------------------------------------------------

if analizar:

    documents = [
        d.strip()
        for d in text_input.split("\n")
        if d.strip()
    ]

    if len(documents) < 1:

        st.error("Ingresa al menos un documento.")

    elif not question.strip():

        st.error("Escribe una pregunta.")

    else:

        # -------------------------------------------------
        # CREAR VECTORIZADOR TF-IDF
        # -------------------------------------------------

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )

        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)


        # -------------------------------------------------
        # MATRIZ TF-IDF
        # -------------------------------------------------

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">

            <div class="card-title">
                Matriz TF-IDF
            </div>

            <div class="card-description">
                Representación numérica de las palabras presentes
                en cada documento.
            </div>

        </div>
        """, unsafe_allow_html=True)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[
                f"Doc {i+1}"
                for i in range(len(documents))
            ]
        )

        st.dataframe(
            df_tfidf.round(3),
            use_container_width=True
        )


        # -------------------------------------------------
        # SIMILITUD CON LA PREGUNTA
        # -------------------------------------------------

        question_vec = vectorizer.transform([question])

        similarities = cosine_similarity(
            question_vec,
            X
        ).flatten()


        # -------------------------------------------------
        # MEJOR RESPUESTA
        # -------------------------------------------------

        best_idx = similarities.argmax()

        best_doc = documents[best_idx]

        best_score = similarities[best_idx]


        # -------------------------------------------------
        # RESULTADO
        # -------------------------------------------------

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">

            <div class="card-title">
                Resultado del análisis
            </div>

            <div class="card-description">
                El sistema compara tu pregunta con todos los documentos
                y selecciona el que presenta mayor similitud.
            </div>

        </div>
        """, unsafe_allow_html=True)


        # Pregunta
        st.markdown(
            f"""
            <div class="answer-box">

                <div class="answer-label">
                    Consulta realizada
                </div>

                <div class="answer-text">
                    {question}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # Respuesta
        if best_score > 0.01:

            st.markdown(
                f"""
                <div class="answer-box">

                    <div class="answer-label">
                        Documento más relevante
                    </div>

                    <div class="answer-text">
                        {best_doc}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="answer-box">

                    <div class="answer-label">
                        Documento encontrado con baja similitud
                    </div>

                    <div class="answer-text">
                        {best_doc}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # SIMILITUD
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="metric-box">

                <div class="metric-label">
                    Nivel de similitud
                </div>

                <div class="metric-value">
                    {best_score:.3f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )
