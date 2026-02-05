# Imports standard et tiers
import json
import os
from pathlib import Path

# Imports ML et visualisation
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud

# Chemins du projet et artefacts
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mental_heath_feature_engineered.csv"
BEST_META_PATH = BASE_DIR / "artefacts" / "best_model_meta.json"
FALLBACK_MODEL_PATH = BASE_DIR / "artefacts" / "clf_lgb_bge.txt"
LABEL_ENCODER_PATH = BASE_DIR / "artefacts" / "label_encoder.joblib"
FALLBACK_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


@st.cache_data
def load_data():
    # Chargement des donnees et normalisation du label
    df = pd.read_csv(DATA_PATH)
    df["status"] = df["status"].astype(str)
    return df


@st.cache_resource
def load_artifacts():
    # Valeurs par defaut si le best model n est pas present
    model_type = "lightgbm"
    model_path = FALLBACK_MODEL_PATH
    embedding_model_name = FALLBACK_EMBEDDING_MODEL_NAME

    # Lecture du best model si disponible
    if BEST_META_PATH.exists():
        with open(BEST_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model_type = meta.get("model_type", model_type)
        embedding_model_name = meta.get("embedding_model_name", embedding_model_name)
        model_filename = meta.get("model_filename")
        if model_filename:
            model_path = BASE_DIR / "artefacts" / model_filename

    # Chargement de l embedder et du classifieur
    embedder = SentenceTransformer(embedding_model_name)
    if model_type == "lightgbm":
        classifier = lgb.Booster(model_file=str(model_path))
    else:
        classifier = joblib.load(model_path)
    # Chargement du label encoder
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return embedder, classifier, label_encoder, model_type


def render_accessible_header():
    # Configuration globale de la page
    st.set_page_config(
        page_title="Mental Health Dashboard",
        page_icon="🧠",
        layout="wide",
    )
    # Styles de base pour lisibilite
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
    # Titre et introduction
    st.markdown('<div class="app-title">Mental Health Text Classifier</div>', unsafe_allow_html=True)
    st.write(
        "Exploration des donnees, visualisations interactives et moteur de prediction sur des textes."
    )




def render_eda(df, accessibility_mode):
    # En tete de l analyse exploratoire
    st.subheader("Analyse exploratoire")
    if accessibility_mode:
        st.caption(
            "Mode accessibilite actif: contraste renforce, symboles et tableau de valeurs sous les graphiques."
        )
    else:
        st.caption(
            "Statistiques descriptives et graphiques interactifs. Palette couleur compatible daltonisme."
        )

    # Statistiques descriptives
    with st.expander("Statistiques descriptives"):
        st.dataframe(df.describe(include="all"), use_container_width=True)

    # Distribution des statuts
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig_status = px.bar(
        status_counts,
        x="status",
        y="count",
        color="status",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Distribution des statuts",
        pattern_shape="status" if accessibility_mode else None,
    )
    fig_status.update_layout(
        xaxis_title="Statut",
        yaxis_title="Nombre d'exemples",
        legend_title_text="Statut",
        height=520 if accessibility_mode else 440,
    )
    if accessibility_mode:
        fig_status.update_traces(marker_line_width=1.2)
    st.plotly_chart(fig_status, use_container_width=True)
    st.caption("Graphique a barres interactif. Lecture clavier et tooltips inclus.")
    if accessibility_mode:
        st.dataframe(status_counts, use_container_width=True)

    # Nombre de stresswords par statut
    stress_flag = (
        df["has_stress_keyword"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    stress_stats = (
        df.assign(has_stress_keyword=stress_flag)
        .groupby("status")["has_stress_keyword"]
        .agg(rate="mean", total="sum", count="count")
        .reset_index()
    )
    stress_stats["rate_pct"] = (stress_stats["rate"] * 100).round(1)
    fig_stress = px.bar(
        stress_stats,
        x="status",
        y="total",
        color="status",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Nombre de textes avec stresswords par statut",
        pattern_shape="status" if accessibility_mode else None,
        text="total",
    )
    fig_stress.update_layout(
        xaxis_title="Statut",
        yaxis_title="Nombre de textes avec stresswords",
        legend_title_text="Statut",
        height=520 if accessibility_mode else 440,
    )
    if accessibility_mode:
        fig_stress.update_traces(marker_line_width=1.2)
    st.plotly_chart(fig_stress, use_container_width=True)
    st.caption("Nombre de textes contenant un mot-cle de stress (feature engineering).")
    if accessibility_mode:
        st.dataframe(stress_stats, use_container_width=True)

    # Distribution du nombre de mots par statut
    fig_box = px.box(
        df,
        x="status",
        y="word_count",
        color="status",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Distribution du nombre de mots par statut",
        points="outliers",
    )
    fig_box.update_layout(
        xaxis_title="Statut",
        yaxis_title="Nombre de mots",
        showlegend=False,
        height=520 if accessibility_mode else 440,
    )
    if accessibility_mode:
        fig_box.update_traces(marker_line_width=1.0)
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Boite a moustaches: variabilite et outliers par statut.")
    if accessibility_mode:
        summary = df.groupby("status")[["word_count"]].describe().round(3)
        summary = summary.reset_index()
        st.dataframe(summary, use_container_width=True)


    # WordCloud global
    st.subheader("Nuage de mots")
    combined_text = " ".join(df["text"].astype(str).tolist())
    wc = WordCloud(width=900, height=450, background_color="white").generate(combined_text)
    st.image(wc.to_array(), caption="WordCloud des textes", width="stretch")
    if accessibility_mode:
        st.caption(
            "Alternative textuelle: le WordCloud est un complement visuel. "
            "Voir les statistiques ci-dessus pour une lecture exacte."
        )


def render_prediction(df):
    # Bloc prediction
    st.subheader("Moteur de prediction")
    st.caption(
        "Entrez un texte ou choisissez un exemple. Le modele renvoie une prediction de statut."
    )

    # Echantillonnage equilibre des exemples
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

    # Liste deroulante des exemples
    example_texts = [""] + sampled["text"].astype(str).tolist()
    selected_text = st.selectbox(
        "Choisir un exemple (optionnel)",
        example_texts,
        format_func=lambda t: "" if t == "" else t[:preview_len].replace("\n", " "),
    )
    default_text = ""
    if selected_text:
        default_text = selected_text

    # Zone de saisie du texte
    user_text = st.text_area(
        "Texte a analyser",
        value=default_text,
        height=140,
        help="Saisissez un texte libre. Laissez vide pour tester les exemples.",
    )

    if st.button("Predire le statut", type="primary"):
        # Validation de la saisie
        if not user_text.strip():
            st.warning("Veuillez saisir un texte avant de lancer la prediction.")
            return
        # Chargement des artefacts
        try:
            embedder, classifier, label_encoder, model_type = load_artifacts()
        except Exception as exc:
            st.error(f"Impossible de charger les artefacts: {exc}")
            return

        # Encodage du texte et prediction
        emb = embedder.encode([user_text], convert_to_numpy=True)
        if model_type == "lightgbm":
            probs = classifier.predict(emb)
            pred_idx = int(np.argmax(probs, axis=1)[0])
        else:
            pred_idx = int(classifier.predict(emb)[0])
        prediction = label_encoder.inverse_transform([pred_idx])[0]
        st.success(f"Prediction: {prediction}")


def main():
    # Enchainement des sections de la page
    render_accessible_header()
    df = load_data()
    accessibility_mode = st.toggle("Mode accessibilite (WCAG essentiels)", value=False)
    render_eda(df, accessibility_mode)
    render_prediction(df)


if __name__ == "__main__":
    main()
