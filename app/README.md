# Streamlit app

Ce dossier contient l application Streamlit.

## Demarrage

Depuis la racine du projet :

```bash
streamlit run app/app.py
```

## Artefacts requis

Le dashboard charge les fichiers suivants depuis `../artefacts` :

- `best_model_meta.json`
- `label_encoder.joblib`
- le fichier de modele best (ex: `clf_lgb_bge.txt`)

Si ces fichiers manquent, recupere les outputs Kaggle puis relance l app.

## Recuperation des artefacts Kaggle

Apres execution du notebook Kaggle et sauvegarde de la version avec outputs :

```bash
kaggle kernels output pierreschnoering/nlp-sant-mentale-baseline-sbert-vs-bge-m3 -p artefacts -o
```
