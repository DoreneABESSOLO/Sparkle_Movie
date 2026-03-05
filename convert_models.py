# convert_models.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models_simple")

ALS_USER_PKL = MODELS_DIR / "als_user_factors.pkl"
ALS_ITEM_PKL = MODELS_DIR / "als_item_factors.pkl"

OUT_USER_IDS = MODELS_DIR / "user_ids.npy"
OUT_USER_VEC = MODELS_DIR / "user_vectors.npy"
OUT_ITEM_IDS = MODELS_DIR / "movie_ids.npy"
OUT_ITEM_VEC = MODELS_DIR / "item_vectors.npy"


def _describe(obj, name: str):
    print(f"\n=== {name} ===")
    print("type:", type(obj))
    if isinstance(obj, pd.DataFrame):
        print("shape:", obj.shape)
        print("columns:", list(obj.columns))
        print(obj.head(2))
    elif isinstance(obj, dict):
        print("keys:", list(obj.keys()))
        for k, v in obj.items():
            print(f"  - {k}: {type(v)}")
    else:
        # fallback
        try:
            attrs = [a for a in dir(obj) if not a.startswith("_")]
            print("attrs:", attrs[:20], "...")
        except Exception:
            pass


def _to_2d_array(features):
    """
    Convertit une colonne/list de vecteurs en array 2D (n, rank).
    features peut être:
    - pandas Series contenant des listes / np.ndarray
    - list de listes / np.ndarray
    - np.ndarray 2D
    """
    if isinstance(features, np.ndarray) and features.ndim == 2:
        arr = features
    else:
        # Series/list/np.ndarray 1D d'objets => vstack
        if isinstance(features, np.ndarray) and features.ndim == 1:
            iterable = features.tolist()
        else:
            iterable = list(features)
        arr = np.vstack([np.asarray(v, dtype=np.float32) for v in iterable])

    if arr.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={arr.shape}")
    return arr.astype(np.float32)


def _extract_ids_and_vectors(obj, kind: str):
    """
    kind = "user" ou "item"
    Retourne: (ids_1d, vectors_2d)

    Gère plusieurs formats possibles:
    - DataFrame avec colonnes (userId/movieId) + features
    - DataFrame avec colonnes (id) + features
    - dict avec clés possibles: ids, id, userId, movieId, factors, features, vectors...
    """
    id_candidates = ["userId", "movieId", "id", "userid", "movieid", "uid", "mid"]
    vec_candidates = ["features", "factors", "vectors", "vector", "embedding", "embeddings"]

    # --- cas DataFrame ---
    if isinstance(obj, pd.DataFrame):
        cols = list(obj.columns)

        id_col = None
        for c in id_candidates:
            if c in cols:
                id_col = c
                break

        vec_col = None
        for c in vec_candidates:
            if c in cols:
                vec_col = c
                break

        if id_col is None:
            # si aucune colonne id connue: on tente la 1ère colonne "int-like"
            for c in cols:
                if pd.api.types.is_integer_dtype(obj[c]) or pd.api.types.is_numeric_dtype(obj[c]):
                    id_col = c
                    break

        if vec_col is None:
            # si aucune colonne features connue, on cherche une colonne "object" contenant des arrays/listes
            for c in cols:
                if obj[c].dtype == "object":
                    sample = obj[c].dropna().iloc[0] if obj[c].dropna().shape[0] else None
                    if isinstance(sample, (list, np.ndarray)):
                        vec_col = c
                        break

        if id_col is None or vec_col is None:
            raise KeyError(
                f"Impossible de détecter id_col/vec_col dans DataFrame ({kind}). "
                f"colonnes={cols}"
            )

        ids = obj[id_col].to_numpy()
        vecs = _to_2d_array(obj[vec_col])
        return ids, vecs

    # --- cas dict ---
    if isinstance(obj, dict):
        keys = list(obj.keys())

        id_key = None
        for k in id_candidates + ["ids", "index", "indices"]:
            if k in keys:
                id_key = k
                break

        vec_key = None
        for k in vec_candidates:
            if k in keys:
                vec_key = k
                break

        # si pas trouvé, on essaie heuristique:
        if id_key is None:
            for k in keys:
                v = obj[k]
                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    # ids = liste/array 1D d'entiers
                    if isinstance(v, np.ndarray) and v.ndim == 1:
                        id_key = k
                        break
                    if isinstance(v, list) and isinstance(v[0], (int, np.integer)):
                        id_key = k
                        break

        if vec_key is None:
            for k in keys:
                v = obj[k]
                # vecteurs = array 2D ou liste de vecteurs
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    vec_key = k
                    break
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, np.ndarray)):
                    vec_key = k
                    break

        if id_key is None or vec_key is None:
            raise KeyError(
                f"Impossible de détecter id_key/vec_key dans dict ({kind}). keys={keys}"
            )

        ids = np.array(obj[id_key])
        vecs = _to_2d_array(obj[vec_key])
        return ids, vecs

    raise TypeError(f"Format non supporté pour {kind}: {type(obj)}")


def main():
    if not ALS_USER_PKL.exists() or not ALS_ITEM_PKL.exists():
        raise FileNotFoundError("Pickles ALS introuvables dans models_simple/")

    with open(ALS_USER_PKL, "rb") as f:
        als_users = pickle.load(f)
    with open(ALS_ITEM_PKL, "rb") as f:
        als_items = pickle.load(f)

    # debug lisible
    _describe(als_users, "ALS USERS PKL")
    _describe(als_items, "ALS ITEMS PKL")

    user_ids, user_vecs = _extract_ids_and_vectors(als_users, kind="user")
    item_ids, item_vecs = _extract_ids_and_vectors(als_items, kind="item")

    # sanity checks
    if user_vecs.shape[0] != len(user_ids):
        raise ValueError("Mismatch users: ids != vectors rows")
    if item_vecs.shape[0] != len(item_ids):
        raise ValueError("Mismatch items: ids != vectors rows")

    np.save(OUT_USER_IDS, user_ids)
    np.save(OUT_USER_VEC, user_vecs)
    np.save(OUT_ITEM_IDS, item_ids)
    np.save(OUT_ITEM_VEC, item_vecs)

    print("\n✅ Conversion OK")
    print("Saved:", OUT_USER_IDS)
    print("Saved:", OUT_USER_VEC)
    print("Saved:", OUT_ITEM_IDS)
    print("Saved:", OUT_ITEM_VEC)
    print("Shapes:",
          "\n - user_ids:", user_ids.shape,
          "\n - user_vectors:", user_vecs.shape,
          "\n - movie_ids:", item_ids.shape,
          "\n - item_vectors:", item_vecs.shape)


if __name__ == "__main__":
    main()