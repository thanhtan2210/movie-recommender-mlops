"""Train PureSVD: choose k on a validation window inside train, then fit on all of train.

The test set is never read here. The validation split reuses the logic of
src.prepare with the cutoff moved back: train_val = ratings before the
validation cutoff, validation = liked ratings between it and the real cutoff.

Run from the repo root:
  python -m src.train --cutoff 2019-06-01

Writes reports/<cutoff>/validation.csv and artifacts/<cutoff>/model.npz + model_meta.json.
"""
import argparse
import datetime
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import yaml

from src import metrics
from src.config import DataConfig, load_data_config
from src.data import build_interactions, data_config_of, ensure_processed, load_stats, read_parquet
from src.evaluate import TOP_N, build_targets, head_mask, report_dir
from src.model import PureSVD, artifact_dir
from src.prepare import git_state, iso, prepare

TRAIN_CONFIG_PATH = os.path.join("configs", "train.yaml")
VALIDATION_FILE = "validation.csv"


@dataclass(frozen=True)
class TrainConfig:
    k_values: Tuple[int, ...] = (32, 64, 128, 256)
    validation_days: int = 31
    random_state: int = 42


def load_train_config(path: str = TRAIN_CONFIG_PATH) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    if "k_values" in values:
        values["k_values"] = tuple(values["k_values"])
    return TrainConfig(**values)


def validation_config(config: DataConfig, validation_days: int) -> DataConfig:
    """The data config with the cutoff moved back, so the 'test' window is the end of train."""
    day = datetime.date.fromisoformat(config.cutoff) - datetime.timedelta(days=validation_days)
    return replace(config, cutoff=day.isoformat(), test_window_days=validation_days)


def validation_split(train: pd.DataFrame, movies: pd.DataFrame, config: DataConfig, validation_days: int):
    """Split train into (train_val, validation) with the same filters as src.prepare."""
    val_config = validation_config(config, validation_days)
    prepared = prepare(train, movies, val_config)
    # Guard: nothing from the real cutoff onwards, and no overlap between the two parts.
    if len(prepared.train) and prepared.train["timestamp"].max() >= val_config.cutoff_timestamp:
        raise AssertionError("train_val contains ratings from the validation window")
    if len(prepared.test) and prepared.test["timestamp"].max() >= config.cutoff_timestamp:
        raise AssertionError("the validation window reaches the test cutoff")
    return prepared.train, prepared.test, val_config


def sweep(train_val: pd.DataFrame, validation: pd.DataFrame, threshold: float, k_values, random_state: int) -> List[Dict[str, Any]]:
    """Fit one model per k on train_val and score it on the validation users."""
    interactions = build_interactions(train_val, threshold)
    users, rows, targets = build_targets(interactions, validation)
    liked_rows, seen_rows = interactions.liked[rows], interactions.seen[rows]
    head = head_mask(interactions)
    indices = metrics.bootstrap_indices(len(users))

    rows_out = []
    for k in k_values:
        started = time.time()
        model = PureSVD.fit(interactions.liked, k, random_state)
        result = metrics.evaluate(model.recommend(liked_rows, seen_rows, TOP_N), targets, head, indices)
        rows_out.append({
            "k": k,
            "validation_users": int(len(users)),
            "hit_rate_at_10": result["hit_rate_at_10"]["value"],
            "recall_at_10": result["recall_at_10"]["value"],
            "ndcg_at_10": result["ndcg_at_10"]["value"],
            "ndcg_ci95_low": result["ndcg_at_10"]["ci95_low"],
            "ndcg_ci95_high": result["ndcg_at_10"]["ci95_high"],
            "catalog_coverage": result["catalog_coverage"]["value"],
            "long_tail_share": result["long_tail_share"]["value"],
        })
        print(f"  k = {k:>4}: NDCG@10 {rows_out[-1]['ndcg_at_10']:.4f} "
              f"[{rows_out[-1]['ndcg_ci95_low']:.4f}, {rows_out[-1]['ndcg_ci95_high']:.4f}] | "
              f"HitRate@10 {rows_out[-1]['hit_rate_at_10']:.4f} | coverage {rows_out[-1]['catalog_coverage']:.4f} "
              f"({time.time() - started:.0f} s)")
    return rows_out


def choose_k(rows: List[Dict[str, Any]]) -> int:
    """The k with the highest validation NDCG@10; the smaller k wins a tie."""
    return max(rows, key=lambda row: (row["ndcg_at_10"], -row["k"]))["k"]


def main():
    parser = argparse.ArgumentParser(description="Choose k on a validation window, then train PureSVD on all of train.")
    parser.add_argument("--cutoff", help="YYYY-MM-DD; default: the cutoff in configs/data.yaml")
    parser.add_argument("--config", default=TRAIN_CONFIG_PATH)
    args = parser.parse_args()
    cutoff = args.cutoff or load_data_config().cutoff
    train_config = load_train_config(args.config)

    directory = ensure_processed(cutoff)
    stats = load_stats(directory)
    config = data_config_of(stats)
    movies = read_parquet(directory, "movies.parquet")

    # 1. Choose k without touching the test set.
    train_val, validation, val_config = validation_split(
        read_parquet(directory, "train.parquet"), movies, config, train_config.validation_days)
    print(f"Validation: train_val before {val_config.cutoff} ({len(train_val):,} rows), "
          f"{len(validation):,} liked ratings of {validation['userId'].nunique():,} users "
          f"from {val_config.cutoff} up to {cutoff}")
    rows = sweep(train_val, validation, config.positive_threshold, train_config.k_values, train_config.random_state)
    best_k = choose_k(rows)
    del train_val, validation
    pa.default_memory_pool().release_unused()

    report = pd.DataFrame(rows)
    report["selected"] = report["k"] == best_k
    os.makedirs(report_dir(cutoff), exist_ok=True)
    report.to_csv(os.path.join(report_dir(cutoff), VALIDATION_FILE), index=False, encoding="utf-8")
    print(f"Selected k = {best_k}")

    # 2. Fit on all of train with the chosen k.
    interactions = build_interactions(read_parquet(directory, "train.parquet"), config.positive_threshold)
    started = time.time()
    model = PureSVD.fit(interactions.liked, best_k, train_config.random_state,
                        item_ids=interactions.item_ids, user_ids=interactions.user_ids)
    train_seconds = round(time.time() - started, 1)
    model.save(artifact_dir(cutoff), {
        "cutoff": cutoff,
        "random_state": train_config.random_state,
        "positive_threshold": config.positive_threshold,
        "liked_interactions": int(interactions.liked.nnz),
        "train_sha256": stats["sha256"]["train.parquet"],
        "validation": {"cutoff": val_config.cutoff, "k_values": list(train_config.k_values),
                       "selected_by": "ndcg_at_10"},
        "train_seconds": train_seconds,
        "trained_at_utc": iso(time.time()),
        **git_state(),
    })
    print(f"Trained k = {best_k} on {interactions.liked.shape[0]:,} users x {interactions.liked.shape[1]:,} movies "
          f"({interactions.liked.nnz:,} liked ratings) in {train_seconds} s")
    print(f"Saved {artifact_dir(cutoff)} and {os.path.join(report_dir(cutoff), VALIDATION_FILE)}")


if __name__ == "__main__":
    main()
