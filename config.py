from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Charge config.yaml + override ENV.
    ENV supportées (exemples):
      CONFIG_PATH, PROJECT_ROOT, DATA_ROOT, MODELS_DIR,
      SPARK_SHUFFLE_PARTITIONS,
      TRAIN_RATIO, VAL_RATIO, SEED,
      ALS_RANKS="10,20,50", ALS_REGPARAMS="0.01,0.05,0.1", ALS_MAXITERS="10,15",
      TFIDF_NUM_FEATURES
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # --- ENV overrides simples ---
    project_root = os.environ.get("PROJECT_ROOT")
    data_root = os.environ.get("DATA_ROOT")
    models_dir = os.environ.get("MODELS_DIR")

    if project_root:
        cfg["paths"]["project_root"] = project_root
    if data_root:
        cfg["paths"]["data_root"] = data_root
    if models_dir:
        cfg["paths"]["models_dir"] = models_dir

    # dataset split/seed
    if os.environ.get("SEED"):
        cfg["dataset"]["seed"] = int(os.environ["SEED"])
    if os.environ.get("TRAIN_RATIO"):
        cfg["dataset"]["split"]["train_ratio"] = float(os.environ["TRAIN_RATIO"])
    if os.environ.get("VAL_RATIO"):
        cfg["dataset"]["split"]["val_ratio"] = float(os.environ["VAL_RATIO"])

    # spark
    if os.environ.get("SPARK_SHUFFLE_PARTITIONS"):
        cfg["spark"]["shuffle_partitions"] = int(os.environ["SPARK_SHUFFLE_PARTITIONS"])

    # ALS grid
    if os.environ.get("ALS_RANKS"):
        cfg["als"]["grid"]["rank"] = [int(x) for x in os.environ["ALS_RANKS"].split(",")]
    if os.environ.get("ALS_REGPARAMS"):
        cfg["als"]["grid"]["regParam"] = [float(x) for x in os.environ["ALS_REGPARAMS"].split(",")]
    if os.environ.get("ALS_MAXITERS"):
        cfg["als"]["grid"]["maxIter"] = [int(x) for x in os.environ["ALS_MAXITERS"].split(",")]

    # TFIDF
    if os.environ.get("TFIDF_NUM_FEATURES"):
        cfg["tfidf"]["num_features"] = int(os.environ["TFIDF_NUM_FEATURES"])

    return cfg


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convertit paths relatifs -> absolus à partir de project_root.
    """
    pr = Path(cfg["paths"]["project_root"]).resolve()
    cfg["paths"]["project_root"] = str(pr)
    cfg["paths"]["data_root"] = str((pr / cfg["paths"]["data_root"]).resolve())
    cfg["paths"]["models_dir"] = str((pr / cfg["paths"]["models_dir"]).resolve())
    return cfg