# Artefacts

Ce dossier contient les artefacts nécessaires au dashboard.

## Recuperer les outputs Kaggle

Apres execution du notebook Kaggle et sauvegarde de la version avec outputs :

```bash
kaggle kernels output pierreschnoering/nlp-sant-mentale-baseline-sbert-vs-bge-m3 -p artefacts -o
```

## Fichiers attendus

- `best_model_meta.json`
- `label_encoder.joblib`
- fichier du modele best (ex: `clf_lgb_bge.txt`)
