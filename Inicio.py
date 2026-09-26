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
    page_title="TF-IDF en Español",
    page_icon="🔎",
    layout="wide"
)

# ---------------------------------------------------------
# DISEÑO VISUAL
# ---------------------------------------------------------

st.markdown("""
<style>

    .stApp {
        background-color: #F7F4EF;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .titulo {
        background-color: #FFFFFF;
        padding: 25px 30px;
        border-radius: 18px;
        border: 1px solid #E5E0D8;
        margin-bottom: 25px;
    }

    .titulo h1 {
        color: #2F2B27;
        font-size: 34px;
        margin-bottom: 5px;
    }

    .titulo p {
        color: #777067;
        font-size: 15px;
        margin: 0;
    }

    .seccion {
        background-color: #FFFFFF;
        padding: 20px 24px;
        border-radius: 16px;
        border: 1px solid #E5E0D8;
        margin-bottom: 15px;
    }

    .seccion h3 {
        color: #36312C;
        margin-bottom: 5px;
    }

    .seccion p {
        color: #817970;
        font-size: 14px;
    }

    .resultado {
        background-color: #EEE9E1;
        padding: 18px 22px;
        border-radius: 14px;
        border-left: 4px solid #8C8172;
        margin-top: 10px;
    }

    .resultado-titulo {
        color: #70675D;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .resultado-texto {
        color: #302C27;
        font-size: 16px;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #D8D1C7;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------

st.markdown("""
<div class="titulo">

    <h1>🔎 Analizador TF-IDF en Español</h1>

    <p>
        Explora cómo un modelo de procesamiento de lenguaje puede
        encontrar el documento más relacionado con una pregunta.
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

    text = text.lower()

    text = re.sub(
        r'[^a-záéíóúüñ\s]',
        ' ',
        text
    )

    tokens = [
        t for t in text.split()
        if len(t) > 1
    ]

    stems = [
        stemmer.stem(t)
        for t in tokens
    ]

    return stems


# ---------------------------------------------------------
# COLUMNAS
# ---------------------------------------------------------

col1, col2 = st.columns([2, 1])


# ---------------------------------------------------------
# COLUMNA PRINCIPAL
# ---------------------------------------------------------

with col1:

    st.markdown("""
    <div class="seccion">

        <h3>📝 Documentos</h3>

        <p>
            Escribe un documento por línea. Puedes modificar
            estos ejemplos para experimentar con TF-IDF.
        </p>

    </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "Documentos (uno por línea):",
        default_docs,
        height=150
    )

    st.markdown("""
    <div class="seccion">

        <h3>❓ Pregunta</h3>

        <p>
            Escribe una pregunta relacionada con la información
            de los documentos.
        </p>

    </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "Escribe tu pregunta:",
        "¿Dónde juegan el perro y el gato?"
    )


# ---------------------------------------------------------
# PREGUNTAS SUGERIDAS
# ---------------------------------------------------------

with col2:

    st.markdown("""
    <div class="seccion">

        <h3>💡 Preguntas sugeridas</h3>

        <p>
            Selecciona una pregunta para probar rápidamente
            el análisis.
        </p>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "¿Dónde juegan el perro y el gato?",
        use_container_width=True
    ):
        st.session_state.question = \
            "¿Dónde juegan el perro y el gato?"
        st.rerun()

    if st.button(
        "¿Qué hacen los niños en el parque?",
        use_container_width=True
    ):
        st.session_state.question = \
            "¿Qué hacen los niños en el parque?"
        st.rerun()

    if st.button(
        "¿Cuándo cantan los pájaros?",
        use_container_width=True
    ):
        st.session_state.question = \
            "¿Cuándo cantan los pájaros?"
        st.rerun()

    if st.button(
        "¿Dónde suena la música alta?",
        use_container_width=True
    ):
        st.session_state.question = \
            "¿Dónde suena la música alta?"
        st.rerun()

    if st.button(
        "¿Qué animal maúlla durante la noche?",
        use_container_width=True
    ):
        st.session_state.question = \
            "¿Qué animal maúlla durante la noche?"
        st.rerun()


# ---------------------------------------------------------
# ACTUALIZAR PREGUNTA
# ---------------------------------------------------------

if 'question' in st.session_state:
    question = st.session_state.question


# ---------------------------------------------------------
# BOTÓN ANALIZAR
# ---------------------------------------------------------

st.markdown("")

if st.button(
    "🔎 Analizar",
    type="primary",
    use_container_width=True
):

    documents = [
        d.strip()
        for d in text_input.split("\n")
        if d.strip()
    ]

    if len(documents) < 1:

        st.error(
            "⚠️ Ingresa al menos un documento."
        )

    elif not question.strip():

        st.error(
            "⚠️ Escribe una pregunta."
        )

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

        st.markdown("""
        <div class="seccion">

            <h3>📊 Matriz TF-IDF</h3>

            <p>
                Esta tabla muestra el peso que tiene cada palabra
                dentro de los documentos analizados.
            </p>

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

        question_vec = vectorizer.transform(
            [question]
        )

        similarities = cosine_similarity(
            question_vec,
            X
        ).flatten()


        # -------------------------------------------------
        # ENCONTRAR MEJOR RESPUESTA
        # -------------------------------------------------

        best_idx = similarities.argmax()

        best_doc = documents[best_idx]

        best_score = similarities[best_idx]


        # -------------------------------------------------
        # RESPUESTA
        # -------------------------------------------------

        st.markdown("""
        <div class="seccion">

            <h3>🎯 Resultado</h3>

            <p>
                El sistema encontró el documento con mayor
                similitud respecto a tu pregunta.
            </p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="resultado">

                <div class="resultado-titulo">
                    Tu pregunta
                </div>

                <div class="resultado-texto">
                    {question}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        if best_score > 0.01:

            st.success(
                f"**Respuesta:** {best_doc}"
            )

            st.info(
                f"📈 Similitud: {best_score:.3f}"
            )

        else:

            st.warning(
                f"**Respuesta (baja confianza):** {best_doc}"
            )

            st.info(
                f"📉 Similitud: {best_score:.3f}"
            )
