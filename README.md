# Sparkle_Movie

# 🎬 Sparkle Movie — Système de Recommandation (Spark + FastAPI)

## 📌 Description

Ce projet implémente un système de recommandation de films basé sur le dataset MovieLens (ml-latest-small).

Il repose sur :

- Collaborative Filtering via ALS (Spark MLlib)
- Filtrage basé sur le contenu via TF-IDF
- Pipeline paramétrable via config.yaml et variables d’environnement
- Logging complet des entraînements
- API REST avec FastAPI
- Route de healthcheck
- Gestion robuste des erreurs

Le projet est conçu pour être déployable (Docker / serveur dédié).

---

# 🏗 Structure du projet

```
sparkle-movie/
│
├── config.yaml
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── train.py
│   └── evaluate.py
│
├── api/
│   ├── __init__.py
│   └── app.py
│
├── data/
├── models/
│   ├── models_simple/
│   └── runs/
│
└── README.md
```

---

# ⚙️ Configuration du Pipeline

Le pipeline est paramétrable via :

## 1️⃣ config.yaml

Exemple :

```yaml
paths:
  project_root: "."
  data_root: "data"
  models_dir: "models"

dataset:
  movielens_url: "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
  seed: 42
  split:
    train_ratio: 0.8
    val_ratio: 0.1

spark:
  master: "local[*]"
  app_name_train: "sparkle-movie-train"
  app_name_eval: "sparkle-movie-evaluate"
  shuffle_partitions: 8

als:
  grid:
    rank: [10, 20, 50]
    regParam: [0.01, 0.05, 0.1]
    maxIter: [10, 15]
  final_refit_on_trainval: true
  nonnegative: true
  cold_start_strategy: "drop"

tfidf:
  num_features: 4096
  stopwords: "english"
```

---

## 2️⃣ Variables d’environnement (override possible)

Les paramètres peuvent être surchargés :

```
PROJECT_ROOT=/app
DATA_ROOT=/app/data
MODELS_DIR=/app/models
SEED=123
TRAIN_RATIO=0.85
VAL_RATIO=0.10
ALS_RANKS=10,30
ALS_REGPARAMS=0.05,0.1
ALS_MAXITERS=10
TFIDF_NUM_FEATURES=2048
```

---

# 🚀 Entraînement

## Lancer l'entraînement

```
python -m src.train
```

## Étapes réalisées

1. Téléchargement du dataset (si absent)
2. Nettoyage minimal
3. Split train / validation / test
4. Grid Search ALS
5. Sélection des meilleurs hyperparamètres
6. Refit sur train+validation
7. Évaluation sur test
8. Construction TF-IDF
9. Sauvegarde des artefacts
10. Logging complet du run

---

# 📊 Logs d'entraînement

À chaque exécution :

```
models/runs/YYYYMMDD_HHMMSS/
```

Contient :

- train_config.json
- summary.json
- grid_results.json

Informations enregistrées :

- Date et heure UTC
- Hyperparamètres testés
- Meilleurs hyperparamètres
- RMSE validation
- RMSE test
- MAE test
- Chemins des artefacts sauvegardés

---

# 💾 Artefacts déployables

Enregistrés dans :

```
models/models_simple/
```

Contenu :

- als_user_factors.pkl
- als_item_factors.pkl
- tfidf_features.pkl
- metrics.json

L’API n’utilise pas Spark au runtime.
Les facteurs ALS exportés sont suffisants.

---

# 🧪 Évaluation

```
python -m src.evaluate
```

Produit :

- models/models_simple/evaluate.json
- Log dans models/runs/

---

# 🌐 API REST (FastAPI)

L’API charge au démarrage :

- Les facteurs ALS
- Les métriques
- Le fichier movies.csv

## Démarrage local

```
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

# 🔎 Endpoints

## 1) Health Check

### GET /health

Vérifie :

- Modèle chargé
- Artefacts présents
- Chemins corrects
- Paramètres du modèle

### Réponse 200

```json
{
  "status": "ok",
  "models_dir": "/app/models",
  "data_root": "/app/data",
  "movielens_folder": "ml-latest-small",
  "best_params": {
    "rank": 20,
    "regParam": 0.1,
    "maxIter": 10
  }
}
```

### Réponse 500

```json
{
  "status": "ko",
  "error": "Model/data not loaded"
}
```

---

## 2) Recommandations

### GET /recommendations

### Paramètres :

- user_id (int, obligatoire)
- top_n (int, optionnel, 1–100, défaut = 10)

### Exemple :

```
/recommendations?user_id=1&top_n=5
```

### Réponse 200

```json
{
  "user_id": 1,
  "top_n": 5,
  "recommendations": [
    {
      "movieId": 3114,
      "title": "Toy Story 2 (1999)",
      "genres": "Adventure|Animation|Children|Comedy|Fantasy",
      "score": 5.12
    }
  ]
}
```

---

## Gestion des erreurs

### 404 – Utilisateur inconnu

```json
{
  "error": "Unknown user_id",
  "details": {
    "user_id": 999999
  }
}
```

### 500 – Modèle absent

```json
{
  "error": "Model/data not loaded"
}
```

---

# 🧠 Logique de recommandation

L’API :

1. Charge les facteurs ALS exportés
2. Calcule le produit scalaire :

```
score = user_vector · item_vector
```

3. Trie les scores
4. Retourne les Top-N films

---

# ✅ Conformité aux exigences

✔ Pipeline paramétrable via YAML et ENV  
✔ Grid Search manuel  
✔ Sauvegarde des hyperparamètres  
✔ Logging complet des runs  
✔ Sauvegarde des métriques (RMSE, MAE)  
✔ API REST fonctionnelle  
✔ Route /recommendations  
✔ Route /health  
✔ Gestion robuste des erreurs  
✔ Documentation complète des endpoints  

---

# 🚀 Prochaine étape

Dockerisation et déploiement sur Dokploy.