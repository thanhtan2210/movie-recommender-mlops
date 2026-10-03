"""Evaluate the trained PureSVD model against the popularity baseline on the test set.

Run from the repo root, after src.train:
  python -m src.evaluate --cutoff 2019-06-01

Writes reports/<cutoff>/test_metrics.json (metrics with bootstrap intervals,
paired differences) and reports/<cutoff>/examples.json (three users to read).
"""
import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src import baseline, metrics
from src.config import load_data_config
from src.data import Interactions, build_interactions, data_config_of, ensure_processed, load_stats, read_parquet
from src.model import NO_RECOMMENDATION, PureSVD, artifact_dir, load_meta
from src.prepare import git_state, iso

REPORT_DIR = "reports"
TEST_METRICS_FILE = "test_metrics.json"
EXAMPLES_FILE = "examples.json"
TOP_N = 10
HEAD_SHARE = 0.2  # the 20% most rated movies in train are the "head"; the rest is the long tail
N_EXAMPLES = 3
RECENT_LIKED = 5
# Above these, a result on MovieLens with a time split is more likely leakage than skill.
MAX_PLAUSIBLE_HIT_RATE = 0.6
MAX_PLAUSIBLE_NDCG = 0.4


def report_dir(cutoff: str) -> str:
    return os.path.join(REPORT_DIR, cutoff)


def build_targets(interactions: Interactions, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Users to evaluate, their rows in the training matrices, and their liked movies as a boolean matrix."""
    users = np.unique(frame["userId"].to_numpy())
    targets = np.zeros((len(users), len(interactions.item_ids)), dtype=bool)
    targets[np.searchsorted(users, frame["userId"].to_numpy()), interactions.cols_of(frame["movieId"].to_numpy())] = True
    return users, interactions.rows_of(users), targets


def head_mask(interactions: Interactions) -> np.ndarray:
    """True for the HEAD_SHARE of movies with the most ratings in train."""
    counts = np.bincount(interactions.seen.indices, minlength=len(interactions.item_ids))
    head = np.zeros(len(counts), dtype=bool)
    head[np.argsort(-counts, kind="stable")[:math.ceil(HEAD_SHARE * len(counts))]] = True
    return head


def titles(columns, interactions: Interactions, title_of: Dict[int, str]) -> List[str]:
    return [title_of.get(int(interactions.item_ids[c]), "?") for c in columns if c != NO_RECOMMENDATION]


def build_examples(train: pd.DataFrame, movies: pd.DataFrame, interactions: Interactions, users: np.ndarray,
                   targets: np.ndarray, recommendations: Dict[str, np.ndarray], threshold: float,
                   seed: int = metrics.SEED) -> List[Dict[str, Any]]:
    """A few randomly chosen test users: recent likes, what each recommender showed, what they went on to like."""
    title_of = dict(zip(movies["movieId"].tolist(), movies["title"].tolist()))
    chosen = np.sort(np.random.default_rng(seed).choice(len(users), size=min(N_EXAMPLES, len(users)), replace=False))
    examples = []
    for position in chosen:
        history = train[(train["userId"] == users[position]) & (train["rating"] >= threshold)]
        recent = history.sort_values(["timestamp", "movieId"]).tail(RECENT_LIKED)
        example = {
            "user_id": int(users[position]),
            "liked_movies_in_train": int(len(history)),
            "last_liked_in_train": [title_of.get(int(m), "?") for m in recent["movieId"]][::-1],
            "liked_in_test_window": titles(np.flatnonzero(targets[position]), interactions, title_of),
        }
        for name, recs in recommendations.items():
            example[f"{name}_top_10"] = titles(recs[position], interactions, title_of)
            example[f"{name}_hits"] = int(metrics.hits_matrix(recs[position:position + 1], targets[position:position + 1]).sum())
        examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate PureSVD and the popularity baseline on the test set.")
    parser.add_argument("--cutoff", help="YYYY-MM-DD; default: the cutoff in configs/data.yaml")
    args = parser.parse_args()
    cutoff = args.cutoff or load_data_config().cutoff

    directory = ensure_processed(cutoff)
    stats = load_stats(directory)
    config = data_config_of(stats)
    model, meta = PureSVD.load(artifact_dir(cutoff)), load_meta(artifact_dir(cutoff))
    if meta["train_sha256"] != stats["sha256"]["train.parquet"]:
        raise SystemExit("The model was trained on a different train.parquet. Run src.train again.")

    train = read_parquet(directory, "train.parquet")
    test = read_parquet(directory, "test.parquet")
    movies = read_parquet(directory, "movies.parquet")
    interactions = build_interactions(train, config.positive_threshold)
    if not np.array_equal(interactions.item_ids, model.item_ids):
        raise SystemExit("The model's movie index does not match the training set.")

    users, rows, targets = build_targets(interactions, test)
    liked_rows, seen_rows = interactions.liked[rows], interactions.seen[rows]
    recommendations = {
        "pure_svd": model.recommend(liked_rows, seen_rows, TOP_N),
        "popularity": baseline.recommend(
            baseline.popularity_ranking(train, interactions.item_ids, config.cutoff_timestamp,
                                        config.positive_threshold), seen_rows, TOP_N),
    }

    head = head_mask(interactions)
    indices = metrics.bootstrap_indices(len(users))
    results = {name: metrics.evaluate(recs, targets, head, indices) for name, recs in recommendations.items()}
    difference = metrics.paired_difference(recommendations["pure_svd"], recommendations["popularity"],
                                           targets, head, indices)
    difference["catalog_coverage"] = {
        "value": results["pure_svd"]["catalog_coverage"]["value"] - results["popularity"]["catalog_coverage"]["value"]}

    print(f"Test: {len(test):,} liked ratings of {len(users):,} users, {len(interactions.item_ids):,} movies, k = {model.k}")
    for name, result in results.items():
        print(f"  {name:<11} " + " | ".join(
            f"{label} {result[key]['value']:.4f} [{result[key]['ci95_low']:.4f}, {result[key]['ci95_high']:.4f}]"
            for label, key in [("HitRate@10", "hit_rate_at_10"), ("Recall@10", "recall_at_10"),
                               ("NDCG@10", "ndcg_at_10"), ("long tail", "long_tail_share")])
              + f" | coverage {result['catalog_coverage']['value']:.4f}")
    print("  SVD - popularity (paired): " + " | ".join(
        f"{key} {value['value']:+.4f} [{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"
        for key, value in difference.items() if "ci95_low" in value))

    for name, result in results.items():
        if result["hit_rate_at_10"]["value"] > MAX_PLAUSIBLE_HIT_RATE or result["ndcg_at_10"]["value"] > MAX_PLAUSIBLE_NDCG:
            raise SystemExit(f"{name}: the result is implausibly high for this task: suspect leakage. Nothing was written.")

    report = {
        "cutoff": cutoff,
        "k": model.k,
        "top_n": TOP_N,
        "evaluated_users": int(len(users)),
        "test_rows": int(len(test)),
        "catalog_movies": int(len(interactions.item_ids)),
        "head_movies": int(head.sum()),
        "targets_already_rated_in_train": int((targets & (seen_rows.toarray() > 0)).sum()),
        "bootstrap": {"samples": metrics.BOOTSTRAP_SAMPLES, "seed": metrics.SEED, "resampled_unit": "user"},
        "popularity_window_utc": {"from": iso(baseline.window_start(config.cutoff_timestamp)),
                                  "to_exclusive": iso(config.cutoff_timestamp)},
        "models": results,
        "difference_pure_svd_minus_popularity": difference,
        "train_seconds": meta["train_seconds"],
        "train_sha256": meta["train_sha256"],
        "model_item_factors_sha256": meta["item_factors_sha256"],
        "model_git_commit": meta.get("git_commit"),
        **git_state(),
    }
    os.makedirs(report_dir(cutoff), exist_ok=True)
    with open(os.path.join(report_dir(cutoff), TEST_METRICS_FILE), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    examples = build_examples(train, movies, interactions, users, targets, recommendations, config.positive_threshold)
    with open(os.path.join(report_dir(cutoff), EXAMPLES_FILE), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved {os.path.join(report_dir(cutoff), TEST_METRICS_FILE)} and {EXAMPLES_FILE}")


if __name__ == "__main__":
    main()
