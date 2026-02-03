Streamlit app

Demarrage (depuis la racine) :
  streamlit run app/app.py

Artefacts requis dans ../models :
  - best_model_meta.json
  - label_encoder.joblib
  - fichier du modele best (ex: clf_lgb_bge.txt)

Recuperation des artefacts Kaggle :
  kaggle kernels output pierreschnoering/nlp-sant-mentale-baseline-sbert-vs-bge-m3 -p models -o
