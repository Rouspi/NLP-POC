# Notebooks

Ce dossier contient les notebooks du projet et le metadata Kaggle associe.

## Recuperer le notebook depuis Kaggle

Depuis la racine du projet :

```bash
kaggle kernels pull pierreschnoering/nlp-sant-mentale-baseline-sbert-vs-bge-m3 -p notebook --metadata
```

## Sorties Kaggle utiles

Les artefacts du meilleur modele sont generes dans `/kaggle/working` puis
telecharges localement avec :

```bash
kaggle kernels output pierreschnoering/nlp-sant-mentale-baseline-sbert-vs-bge-m3 -p artefacts -o
```

## Bonnes pratiques

- Toujours sauvegarder la version Kaggle avec les outputs.
- Conserver `kernel-metadata.json` a jour si le notebook change de nom.
- Lancer les notebooks depuis la racine pour garder des chemins stables.
