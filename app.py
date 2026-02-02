import json
import joblib
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud

DATA_PATH = "data/mental_heath_feature_engineered.csv"
BEST_META_PATH = "models/best_model_meta.json"
FALLBACK_MODEL_PATH = "models/clf_lgb_bge.txt"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"
FALLBACK_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["status"] = df["status"].astype(str)
    return df


@st.cache_resource
def load_artifacts():
    model_type = "lightgbm"
    model_path = FALLBACK_MODEL_PATH
    embedding_model_name = FALLBACK_EMBEDDING_MODEL_NAME

    if os.path.exists(BEST_META_PATH):
        with open(BEST_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model_type = meta.get("model_type", model_type)
        embedding_model_name = meta.get("embedding_model_name", embedding_model_name)
        model_filename = meta.get("model_filename")
        if model_filename:
            model_path = os.path.join("models", model_filename)

    embedder = SentenceTransformer(embedding_model_name)
    if model_type == "lightgbm":
        classifier = lgb.Booster(model_file=model_path)
    else:
        classifier = joblib.load(model_path)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return embedder, classifier, label_encoder, model_type


def render_accessible_header():
    st.set_page_config(
        page_title="Mental Health Dashboard",
        page_icon="🧠",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .app-title { font-size: 2.2rem; font-weight: 700; }
        .caption { font-size: 0.95rem; color: #333333; }
        .block-container { padding-top: 1.5rem; }
        .stTextInput, .stTextArea { border: 1px solid #4f4f4f; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="app-title">Mental Health Text Classifier</div>', unsafe_allow_html=True)
    st.write(
        "Exploration des donnees, visualisations interactives et moteur de prediction sur des textes."
    )


def render_eda(df):
    st.subheader("Analyse exploratoire")
    st.caption(
        "Statistiques descriptives et graphiques interactifs. Palette couleur compatible daltonisme."
    )

    with st.expander("Statistiques descriptives"):
        st.dataframe(df.describe(include="all"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig_status = px.bar(
            status_counts,
            x="status",
            y="count",
            color="status",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Distribution des statuts",
        )
        fig_status.update_layout(
            xaxis_title="Statut",
            yaxis_title="Nombre d'exemples",
            legend_title_text="Statut",
        )
        st.plotly_chart(fig_status, use_container_width=True)
        st.caption("Graphique a barres interactif. Lecture clavier et tooltips inclus.")

    with col2:
        fig_scatter = px.scatter(
            df,
            x="word_count",
            y="text_length",
            color="status",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Longueur du texte vs nombre de mots",
            hover_data=["text"],
        )
        fig_scatter.update_layout(
            xaxis_title="Nombre de mots",
            yaxis_title="Longueur normalisee du texte",
            legend_title_text="Statut",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Nuage de points interactif. Les tooltips affichent le texte.")

    st.subheader("Nuage de mots")
    combined_text = " ".join(df["text"].astype(str).tolist())
    wc = WordCloud(width=900, height=450, background_color="white").generate(combined_text)
    st.image(wc.to_array(), caption="WordCloud des textes", use_column_width=True)


def render_prediction(df):
    st.subheader("Moteur de prediction")
    st.caption(
        "Entrez un texte ou choisissez un exemple. Le modele renvoie une prediction de statut."
    )

    example_ids = df["Unique_ID"].astype(str).tolist()
    selected_id = st.selectbox("Choisir un exemple (optionnel)", [""] + example_ids)
    default_text = ""
    if selected_id:
        default_text = df.loc[df["Unique_ID"].astype(str) == selected_id, "text"].values[0]

    user_text = st.text_area(
        "Texte a analyser",
        value=default_text,
        height=140,
        help="Saisissez un texte libre. Laissez vide pour tester les exemples.",
    )

    if st.button("Predire le statut", type="primary"):
        if not user_text.strip():
            st.warning("Veuillez saisir un texte avant de lancer la prediction.")
            return
        try:
            embedder, classifier, label_encoder, model_type = load_artifacts()
        except Exception as exc:
            st.error(f"Impossible de charger les artefacts: {exc}")
            return

        emb = embedder.encode([user_text], convert_to_numpy=True)
        if model_type == "lightgbm":
            probs = classifier.predict(emb)
            pred_idx = int(np.argmax(probs, axis=1)[0])
        else:
            pred_idx = int(classifier.predict(emb)[0])
        prediction = label_encoder.inverse_transform([pred_idx])[0]
        st.success(f"Prediction: {prediction}")


def main():
    render_accessible_header()
    df = load_data()
    render_eda(df)
    render_prediction(df)


if __name__ == "__main__":
    main()
