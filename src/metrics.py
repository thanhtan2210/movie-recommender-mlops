"""Top-n ranking metrics with bootstrap confidence intervals over users.

`recommended` is an (n_users, n) array of movie columns, best first, with
NO_RECOMMENDATION for empty slots. `targets` is an (n_users, n_items)
boolean array: the movies each user liked in the evaluation window.
"""
from typing import Any, Dict

import numpy as np

from src.recommender import NO_RECOMMENDATION

BOOTSTRAP_SAMPLES = 1000
SEED = 42


def hits_matrix(recommended: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """hits[u, r] is True when the movie at rank r is one of user u's targets."""
    valid = recommended != NO_RECOMMENDATION
    hits = np.take_along_axis(targets, np.where(valid, recommended, 0), axis=1)
    return hits & valid


def per_user_metrics(recommended: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
    n = recommended.shape[1]
    hits = hits_matrix(recommended, targets)
    n_hits = hits.sum(axis=1)
    n_targets = targets.sum(axis=1)
    best_possible = np.minimum(n_targets, n)

    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.concatenate([[0.0], np.cumsum(discounts)])[best_possible]

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(best_possible > 0, n_hits / best_possible, 0.0)
        ndcg = np.where(ideal > 0, dcg / ideal, 0.0)
    return {"hit_rate": (n_hits > 0).astype(float), "recall": recall, "ndcg": ndcg}


def bootstrap_indices(n_users: int, samples: int = BOOTSTRAP_SAMPLES, seed: int = SEED) -> np.ndarray:
    """Resampled user positions, shape (samples, n_users). Reuse the same array to pair two models."""
    return np.random.default_rng(seed).integers(0, n_users, size=(samples, n_users))


def mean_with_ci(values: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    return {"value": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def ratio_with_ci(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    draws = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"value": float(numerator.sum() / denominator.sum()), "ci95_low": float(low), "ci95_high": float(high)}


def tail_counts(recommended: np.ndarray, head: np.ndarray):
    """Per user: recommendations outside the head, and recommendations made."""
    valid = recommended != NO_RECOMMENDATION
    in_head = head[np.where(valid, recommended, 0)]
    return (valid & ~in_head).sum(axis=1).astype(float), valid.sum(axis=1).astype(float)


def evaluate(recommended: np.ndarray, targets: np.ndarray, head: np.ndarray, indices: np.ndarray) -> Dict[str, Any]:
    """All metrics of one recommender. `head` marks the most rated movies (boolean per column)."""
    per_user = per_user_metrics(recommended, targets)
    tail, made = tail_counts(recommended, head)
    distinct = np.unique(recommended[recommended != NO_RECOMMENDATION])
    return {
        "hit_rate_at_10": mean_with_ci(per_user["hit_rate"], indices),
        "recall_at_10": mean_with_ci(per_user["recall"], indices),
        "ndcg_at_10": mean_with_ci(per_user["ndcg"], indices),
        "long_tail_share": ratio_with_ci(tail, made, indices),
        # A distinct count shrinks under resampling, so a bootstrap interval would
        # sit below the estimate; coverage is reported as a point estimate.
        "catalog_coverage": {"value": float(len(distinct) / targets.shape[1]), "distinct_movies": int(len(distinct))},
    }


def paired_difference(recommended_a: np.ndarray, recommended_b: np.ndarray, targets: np.ndarray,
                      head: np.ndarray, indices: np.ndarray) -> Dict[str, Any]:
    """Metrics of A minus metrics of B, bootstrapped on the same resampled users."""
    a, b = per_user_metrics(recommended_a, targets), per_user_metrics(recommended_b, targets)
    difference = {f"{name}_at_10": mean_with_ci(a[name] - b[name], indices) for name in ("hit_rate", "recall", "ndcg")}

    (tail_a, made_a), (tail_b, made_b) = tail_counts(recommended_a, head), tail_counts(recommended_b, head)
    draws = (tail_a[indices].sum(axis=1) / made_a[indices].sum(axis=1)
             - tail_b[indices].sum(axis=1) / made_b[indices].sum(axis=1))
    low, high = np.percentile(draws, [2.5, 97.5])
    difference["long_tail_share"] = {
        "value": float(tail_a.sum() / made_a.sum() - tail_b.sum() / made_b.sum()),
        "ci95_low": float(low), "ci95_high": float(high),
    }
    return difference
