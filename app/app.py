import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mental_heath_feature_engineered.csv"
BEST_META_PATH = BASE_DIR / "models" / "best_model_meta.json"
FALLBACK_MODEL_PATH = BASE_DIR / "models" / "clf_lgb_bge.txt"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.joblib"
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

    if BEST_META_PATH.exists():
        with open(BEST_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model_type = meta.get("model_type", model_type)
        embedding_model_name = meta.get("embedding_model_name", embedding_model_name)
        model_filename = meta.get("model_filename")
        if model_filename:
            model_path = BASE_DIR / "models" / model_filename

    embedder = SentenceTransformer(embedding_model_name)
    if model_type == "lightgbm":
        classifier = lgb.Booster(model_file=str(model_path))
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
        .app-title {
            font-size: 2.0rem;
            font-weight: 700;
            line-height: 1.2;
            white-space: normal;
            word-break: break-word;
        }
        .caption { font-size: 0.95rem; color: #333333; }
        .block-container { padding-top: 2.5rem; }
        .app-title { margin-top: 0.25rem; }
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
    st.image(wc.to_array(), caption="WordCloud des textes", width="stretch")


def render_prediction(df):
    st.subheader("Moteur de prediction")
    st.caption(
        "Entrez un texte ou choisissez un exemple. Le modele renvoie une prediction de statut."
    )

    preview_len = 80
    sample_size = 40
    statuses = df["status"].dropna().unique().tolist()
    per_class = max(1, sample_size // max(len(statuses), 1))
    remainder = sample_size % max(len(statuses), 1)
    samples = []
    for i, status in enumerate(statuses):
        group = df[df["status"] == status]
        n = min(len(group), per_class + (1 if i < remainder else 0))
        if n > 0:
            samples.append(group.sample(n=n, random_state=42))
    if samples:
        sampled = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        sampled = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    example_texts = [""] + sampled["text"].astype(str).tolist()
    selected_text = st.selectbox(
        "Choisir un exemple (optionnel)",
        example_texts,
        format_func=lambda t: "" if t == "" else t[:preview_len].replace("\n", " "),
    )
    default_text = ""
    if selected_text:
        default_text = selected_text

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
