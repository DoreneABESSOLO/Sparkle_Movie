import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# -----------------------------
# Config via variables d'env
# -----------------------------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app"))
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models_simple"))
MOVIELENS_FOLDER = os.getenv("MOVIELENS_FOLDER", "ml-latest-small")

MOVIES_CSV = DATA_ROOT / MOVIELENS_FOLDER / "movies.csv"

USER_IDS_NPY = MODELS_DIR / "user_ids.npy"
USER_VECS_NPY = MODELS_DIR / "user_vectors.npy"
MOVIE_IDS_NPY = MODELS_DIR / "movie_ids.npy"
ITEM_VECS_NPY = MODELS_DIR / "item_vectors.npy"

# -----------------------------
# Globals chargés au démarrage
# -----------------------------
MOVIES_DF: pd.DataFrame | None = None
MOVIE_ID_TO_TITLE: Dict[int, str] = {}
USER_IDS: np.ndarray | None = None
USER_VECS: np.ndarray | None = None
MOVIE_IDS: np.ndarray | None = None
ITEM_VECS: np.ndarray | None = None

app = FastAPI(title="Sparkle Movie API", version="1.0.0")


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def _load_artifacts() -> None:
    global MOVIES_DF, MOVIE_ID_TO_TITLE
    global USER_IDS, USER_VECS, MOVIE_IDS, ITEM_VECS

    # 1) CSV movies
    if not _exists(MOVIES_CSV):
        raise FileNotFoundError(f"Missing movies.csv: {MOVIES_CSV}")

    MOVIES_DF = pd.read_csv(MOVIES_CSV)
    # movies.csv = movieId,title,genres
    MOVIE_ID_TO_TITLE = dict(
        zip(MOVIES_DF["movieId"].astype(int).tolist(), MOVIES_DF["title"].astype(str).tolist())
    )

    # 2) NPY ALS clean
    for p in [USER_IDS_NPY, USER_VECS_NPY, MOVIE_IDS_NPY, ITEM_VECS_NPY]:
        if not _exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}")

    USER_IDS = np.load(USER_IDS_NPY)
    USER_VECS = np.load(USER_VECS_NPY)
    MOVIE_IDS = np.load(MOVIE_IDS_NPY)
    ITEM_VECS = np.load(ITEM_VECS_NPY)

    if USER_VECS.ndim != 2 or ITEM_VECS.ndim != 2:
        raise ValueError(f"Bad vectors shape: user={USER_VECS.shape}, item={ITEM_VECS.shape}")
    if USER_VECS.shape[1] != ITEM_VECS.shape[1]:
        raise ValueError(f"Rank mismatch: user={USER_VECS.shape}, item={ITEM_VECS.shape}")


def _loaded_ok() -> bool:
    return (
        MOVIES_DF is not None
        and USER_IDS is not None
        and USER_VECS is not None
        and MOVIE_IDS is not None
        and ITEM_VECS is not None
        and len(MOVIE_ID_TO_TITLE) > 0
    )


def _recommend_for_user(user_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
    # index utilisateur
    idx = np.where(USER_IDS == user_id)[0]
    if idx.size == 0:
        raise KeyError(f"Unknown user_id={user_id}")

    uvec = USER_VECS[idx[0]]  # (rank,)

    # scores = dot(u, item_vecs)
    scores = ITEM_VECS @ uvec  # (n_items,)

    # top_n indices
    top_idx = np.argpartition(scores, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    recs = []
    for i in top_idx:
        mid = int(MOVIE_IDS[i])
        recs.append(
            {
                "movieId": mid,
                "title": MOVIE_ID_TO_TITLE.get(mid, f"movieId={mid}"),
                "score": float(scores[i]),
            }
        )
    return recs


@app.on_event("startup")
def startup_event():
    try:
        _load_artifacts()
        print("[api] ✅ Artifacts loaded OK")
    except Exception as e:
        # On ne crash pas le service : /health te dira pourquoi
        print("[api] ❌ Failed to load artifacts:", repr(e))


@app.get("/health")
def health():
    # diagnostics utiles
    payload = {
        "status": "ok" if _loaded_ok() else "ko",
        "models_dir": str(MODELS_DIR),
        "data_root": str(DATA_ROOT),
        "movielens_folder": MOVIELENS_FOLDER,
        "movies_csv": str(MOVIES_CSV),
        "movies_csv_exists": _exists(MOVIES_CSV),
        "user_ids_exists": _exists(USER_IDS_NPY),
        "user_vectors_exists": _exists(USER_VECS_NPY),
        "movie_ids_exists": _exists(MOVIE_IDS_NPY),
        "item_vectors_exists": _exists(ITEM_VECS_NPY),
    }
    if not _loaded_ok():
        return JSONResponse(status_code=500, content=payload)
    return payload


@app.get("/recommendations")
def recommendations(
    user_id: int = Query(..., description="MovieLens userId"),
    top_n: int = Query(10, ge=1, le=50, description="Number of recommendations"),
):
    if not _loaded_ok():
        return JSONResponse(status_code=503, content={"error": "Model artifacts not loaded"})
    try:
        recs = _recommend_for_user(user_id=user_id, top_n=top_n)
        return {"user_id": user_id, "top_n": top_n, "recommendations": recs}
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": repr(e)})