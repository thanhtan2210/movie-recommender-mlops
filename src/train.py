"""Train the model for one cutoff: PureSVD blended with recent popularity.

The parameters in configs/train.yaml are frozen: they were chosen on the
validation window of cutoff 2019-06-01 (see reports/2019-06-01/).

Run from the repo root:
  python -m src.train --cutoff 2019-06-01

Writes artifacts/<cutoff>/state.npz and model_meta.json (not committed).
"""
import argparse
import datetime
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Tuple

import pandas as pd
import yaml

from src.baseline import PopularityRecommender
from src.config import DataConfig, load_data_config
from src.data import data_config_of, ensure_processed, load_stats, read_parquet
from src.model import BlendRecommender
from src.prepare import git_state, iso, prepare
from src.recommender import Recommender

TRAIN_CONFIG_PATH = os.path.join("configs", "train.yaml")
ARTIFACT_DIR = "artifacts"
META_FILE = "model_meta.json"
BLEND, POPULARITY = BlendRecommender.model_type, PopularityRecommender.model_type


@dataclass(frozen=True)
class TrainConfig:
    k: int = 64
    train_window: str = "1y"
    blend_weight: float = 4.0
    random_state: int = 42


def load_train_config(path: str = TRAIN_CONFIG_PATH) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        return TrainConfig(**(yaml.safe_load(f) or {}))


def artifact_dir(cutoff: str) -> str:
    return os.path.join(ARTIFACT_DIR, cutoff)


def model_params(model_type: str, train_config: TrainConfig) -> Dict[str, Any]:
    """The parameters that define a model of this type (popularity has none)."""
    return asdict(train_config) if model_type == BLEND else {}


def fit_model(model_type: str, train: pd.DataFrame, config: DataConfig, train_config: TrainConfig) -> Recommender:
    """Fit a recommender of the given type on ratings before config.cutoff."""
    if model_type == BLEND:
        return BlendRecommender.fit(train, config.cutoff, config.cutoff_timestamp, config.positive_threshold,
                                    **asdict(train_config))
    if model_type == POPULARITY:
        return PopularityRecommender.fit(train, config.cutoff_timestamp, config.positive_threshold)
    raise ValueError(f"Unknown model type {model_type!r}")


def inner_split(train: pd.DataFrame, movies: pd.DataFrame, config: DataConfig,
                days: int) -> Tuple[pd.DataFrame, pd.DataFrame, DataConfig]:
    """Carve an evaluation window out of the end of train, with the filters of src.prepare.

    Returns (ratings before cutoff - days, liked ratings in [cutoff - days, cutoff)
    of eligible users, the config with the cutoff moved back).
    """
    day = datetime.date.fromisoformat(config.cutoff) - datetime.timedelta(days=days)
    inner_config = replace(config, cutoff=day.isoformat(), test_window_days=days)
    prepared = prepare(train, movies, inner_config)
    # Guard: the inner training data stops before the window, and the window stops before the cutoff.
    if len(prepared.train) and prepared.train["timestamp"].max() >= inner_config.cutoff_timestamp:
        raise AssertionError("the inner training set contains ratings from the evaluation window")
    if len(prepared.test) and prepared.test["timestamp"].max() >= config.cutoff_timestamp:
        raise AssertionError("the inner evaluation window reaches the cutoff")
    return prepared.train, prepared.test, inner_config


def main():
    parser = argparse.ArgumentParser(description="Train the blend model on all ratings before the cutoff.")
    parser.add_argument("--cutoff", help="YYYY-MM-DD; default: the cutoff in configs/data.yaml")
    parser.add_argument("--config", default=TRAIN_CONFIG_PATH)
    args = parser.parse_args()
    cutoff = args.cutoff or load_data_config().cutoff
    train_config = load_train_config(args.config)

    directory = ensure_processed(cutoff)
    stats = load_stats(directory)
    config = data_config_of(stats)
    started = time.time()
    model = fit_model(BLEND, read_parquet(directory, "train.parquet"), config, train_config)
    train_seconds = round(time.time() - started, 1)

    os.makedirs(artifact_dir(cutoff), exist_ok=True)
    path = model.save_state(artifact_dir(cutoff))
    meta = {
        "model_type": BLEND, **asdict(train_config), "cutoff": cutoff,
        "n_users": int(len(model.user_ids)), "n_items": int(len(model.item_ids)),
        "liked_interactions_in_window": int(model.liked.nnz),
        "item_factors_sha256": model.svd.factors_sha256(),
        "train_sha256": stats["sha256"]["train.parquet"],
        "train_seconds": train_seconds, "trained_at_utc": iso(time.time()), **git_state(),
    }
    with open(os.path.join(artifact_dir(cutoff), META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Trained {BLEND} (k = {train_config.k}, train_window = {train_config.train_window}, "
          f"blend_weight = {train_config.blend_weight}) on {len(model.user_ids):,} users x "
          f"{len(model.item_ids):,} movies in {train_seconds} s")
    print(f"Saved {path} ({os.path.getsize(path) / 1024 ** 2:.1f} MB)")


if __name__ == "__main__":
    main()
