"""Score recommenders on an evaluation window: the liked ratings of a set of users.

Used by the pipeline for the gate and production windows. As a command it
scores the model saved by src.train against popularity on the test window:

  python -m src.evaluate --cutoff 2019-06-01

and writes artifacts/<cutoff>/evaluation.json (not committed).
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from src import metrics
from src.baseline import PopularityRecommender
from src.config import load_data_config
from src.data import data_config_of, ensure_processed, load_stats, read_parquet
from src.model import BlendRecommender
from src.recommender import STATE_FILE, Recommender

TOP_N = 10
HEAD_SHARE = 0.2  # the 20% most rated movies in train are the "head"; the rest is the long tail
EVALUATION_FILE = "evaluation.json"
# Above these, a result on MovieLens with a time split is more likely leakage than skill.
MAX_PLAUSIBLE_HIT_RATE = 0.6
MAX_PLAUSIBLE_NDCG = 0.4


def head_mask(seen) -> np.ndarray:
    """True for the HEAD_SHARE of movie columns with the most ratings in train."""
    counts = np.bincount(seen.indices, minlength=seen.shape[1])
    head = np.zeros(len(counts), dtype=bool)
    head[np.argsort(-counts, kind="stable")[:math.ceil(HEAD_SHARE * len(counts))]] = True
    return head


@dataclass
class Evaluation:
    users: np.ndarray  # evaluated user ids
    targets: np.ndarray  # (n_users, n_items) bool: liked in the window
    head: np.ndarray
    indices: np.ndarray  # bootstrap resamples shared by all recommenders (paired)
    recommendations: Dict[str, np.ndarray]  # name -> (n_users, TOP_N) movie columns
    results: Dict[str, Dict[str, Any]]  # name -> metrics with intervals

    def difference(self, a: str, b: str) -> Dict[str, Any]:
        """Metrics of `a` minus metrics of `b`, bootstrapped on the same resampled users."""
        difference = metrics.paired_difference(self.recommendations[a], self.recommendations[b], self.targets,
                                               self.head, self.indices)
        difference["catalog_coverage"] = {
            "value": self.results[a]["catalog_coverage"]["value"] - self.results[b]["catalog_coverage"]["value"]}
        return difference


def evaluate_window(recommenders: Dict[str, Recommender], eval_frame: pd.DataFrame) -> Evaluation:
    """Score every recommender on the same users and targets.

    All recommenders must have been fitted on the same training data, so that
    they share the movie index and the definition of the head.
    """
    reference = next(iter(recommenders.values()))
    for recommender in recommenders.values():
        if not np.array_equal(recommender.item_ids, reference.item_ids):
            raise ValueError("The recommenders were not fitted on the same training data.")

    users = np.unique(eval_frame["userId"].to_numpy())
    columns = np.searchsorted(reference.item_ids, eval_frame["movieId"].to_numpy())
    if not np.array_equal(reference.item_ids[np.minimum(columns, len(reference.item_ids) - 1)],
                          eval_frame["movieId"].to_numpy()):
        raise KeyError("The evaluation window contains movies outside the training catalogue.")
    targets = np.zeros((len(users), len(reference.item_ids)), dtype=bool)
    targets[np.searchsorted(users, eval_frame["userId"].to_numpy()), columns] = True

    head = head_mask(reference.seen)
    indices = metrics.bootstrap_indices(len(users))
    recommendations = {name: recommender.recommend_columns(users, TOP_N) for name, recommender in recommenders.items()}
    results = {name: metrics.evaluate(recs, targets, head, indices) for name, recs in recommendations.items()}
    return Evaluation(users=users, targets=targets, head=head, indices=indices,
                      recommendations=recommendations, results=results)


def check_plausible(results: Dict[str, Dict[str, Any]]) -> None:
    for name, result in results.items():
        if result["hit_rate_at_10"]["value"] > MAX_PLAUSIBLE_HIT_RATE or result["ndcg_at_10"]["value"] > MAX_PLAUSIBLE_NDCG:
            raise SystemExit(f"{name}: the result is implausibly high for this task: suspect leakage. Nothing was written.")


def format_result(name: str, result: Dict[str, Any]) -> str:
    return f"  {name:<12} " + " | ".join(
        f"{label} {result[key]['value']:.4f} [{result[key]['ci95_low']:.4f}, {result[key]['ci95_high']:.4f}]"
        for label, key in [("HitRate@10", "hit_rate_at_10"), ("Recall@10", "recall_at_10"),
                           ("NDCG@10", "ndcg_at_10"), ("long tail", "long_tail_share")]
    ) + f" | coverage {result['catalog_coverage']['value']:.4f}"


def format_difference(label: str, difference: Dict[str, Any]) -> str:
    return f"  {label} (paired): " + " | ".join(
        f"{key} {value['value']:+.4f} [{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"
        for key, value in difference.items() if "ci95_low" in value)


def main():
    from src.train import artifact_dir  # imported here: src.train imports this module

    parser = argparse.ArgumentParser(description="Score the trained model and popularity on the test window.")
    parser.add_argument("--cutoff", help="YYYY-MM-DD; default: the cutoff in configs/data.yaml")
    args = parser.parse_args()
    cutoff = args.cutoff or load_data_config().cutoff

    directory = ensure_processed(cutoff)
    config = data_config_of(load_stats(directory))
    model = BlendRecommender.load_state(os.path.join(artifact_dir(cutoff), STATE_FILE))
    train = read_parquet(directory, "train.parquet")
    test = read_parquet(directory, "test.parquet")
    popularity = PopularityRecommender.fit(train, config.cutoff_timestamp, config.positive_threshold)

    evaluation = evaluate_window({"blend": model, "popularity": popularity}, test)
    difference = evaluation.difference("blend", "popularity")
    print(f"Test window of {cutoff}: {len(test):,} liked ratings of {len(evaluation.users):,} users")
    for name, result in evaluation.results.items():
        print(format_result(name, result))
    print(format_difference("blend - popularity", difference))
    check_plausible(evaluation.results)

    path = os.path.join(artifact_dir(cutoff), EVALUATION_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cutoff": cutoff, "evaluated_users": int(len(evaluation.users)), "models": evaluation.results,
                   "difference_blend_minus_popularity": difference}, f, indent=2)
        f.write("\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
