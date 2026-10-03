import math

import numpy as np
import pytest

from src import metrics
from src.model import NO_RECOMMENDATION

N_ITEMS = 10


def targets_of(*liked):
    matrix = np.zeros((len(liked), N_ITEMS), dtype=bool)
    for user, movies in enumerate(liked):
        matrix[user, list(movies)] = True
    return matrix


# Three users, top-3 lists, worked out by hand:
#   A: target movie 2 found at rank 2 of 3; the other target (9) is missed.
#   B: nothing found.
#   C: its only target found at rank 1; the third slot is empty.
RECOMMENDED = np.array([[5, 2, 7], [1, 4, 6], [3, 0, NO_RECOMMENDATION]])
TARGETS = targets_of({2, 9}, {8}, {3})


def test_per_user_metrics_on_a_hand_computed_example():
    result = metrics.per_user_metrics(RECOMMENDED, TARGETS)

    assert result["hit_rate"].tolist() == [1.0, 0.0, 1.0]
    # Recall divides by min(n, number of targets): A has 2 targets, 1 found.
    assert result["recall"].tolist() == [0.5, 0.0, 1.0]
    dcg_a = 1 / math.log2(3)                 # one hit at rank 2
    ideal_a = 1 + 1 / math.log2(3)           # two targets at ranks 1 and 2
    assert result["ndcg"] == pytest.approx([dcg_a / ideal_a, 0.0, 1.0])


def test_recall_denominator_is_capped_at_n():
    recommended = np.array([[0, 1, 2]])
    targets = targets_of({0, 1, 2, 3, 4, 5})  # six targets, only three slots

    result = metrics.per_user_metrics(recommended, targets)

    assert result["recall"].tolist() == [1.0]
    assert result["ndcg"] == pytest.approx([1.0])


def test_empty_slots_never_count_as_hits():
    recommended = np.array([[NO_RECOMMENDATION, NO_RECOMMENDATION, NO_RECOMMENDATION]])

    assert metrics.hits_matrix(recommended, targets_of({0})).sum() == 0


def test_evaluate_coverage_and_long_tail():
    head = np.zeros(N_ITEMS, dtype=bool)
    head[[0, 1]] = True
    indices = metrics.bootstrap_indices(3)

    result = metrics.evaluate(RECOMMENDED, TARGETS, head, indices)

    assert result["hit_rate_at_10"]["value"] == pytest.approx(2 / 3)
    # 8 distinct movies recommended out of 10.
    assert result["catalog_coverage"] == {"value": 0.8, "distinct_movies": 8}
    # 8 recommendations made, 2 of them (movies 1 and 0) in the head.
    assert result["long_tail_share"]["value"] == pytest.approx(6 / 8)
    for name in ("hit_rate_at_10", "recall_at_10", "ndcg_at_10", "long_tail_share"):
        assert result[name]["ci95_low"] <= result[name]["value"] <= result[name]["ci95_high"]


def test_bootstrap_is_reproducible():
    values = np.array([1.0, 0.0, 0.0, 1.0, 0.0] * 40)

    first = metrics.mean_with_ci(values, metrics.bootstrap_indices(len(values)))
    second = metrics.mean_with_ci(values, metrics.bootstrap_indices(len(values)))

    assert first == second
    assert first["ci95_low"] < first["value"] < first["ci95_high"]
    assert metrics.bootstrap_indices(200).shape == (metrics.BOOTSTRAP_SAMPLES, 200)


def test_paired_difference_uses_the_same_users_for_both_models():
    rng = np.random.default_rng(1)
    n_users = 300
    targets = np.zeros((n_users, N_ITEMS), dtype=bool)
    targets[np.arange(n_users), rng.integers(0, N_ITEMS, n_users)] = True
    model_a = np.argsort(rng.random((n_users, N_ITEMS)), axis=1)[:, :3]
    head = np.zeros(N_ITEMS, dtype=bool)
    indices = metrics.bootstrap_indices(n_users)

    same = metrics.paired_difference(model_a, model_a, targets, head, indices)
    assert all(value == {"value": 0.0, "ci95_low": 0.0, "ci95_high": 0.0} for value in same.values())

    # B is A plus one extra hit for 10 users: every resample shows B >= A, so the
    # paired interval excludes zero although the two unpaired intervals overlap.
    model_b = model_a.copy()
    misses = np.flatnonzero(metrics.per_user_metrics(model_a, targets)["hit_rate"] == 0)[:10]
    model_b[misses, 0] = np.argmax(targets[misses], axis=1)
    difference = metrics.paired_difference(model_b, model_a, targets, head, indices)["hit_rate_at_10"]

    assert difference["value"] == pytest.approx(10 / n_users)
    assert difference["ci95_low"] > 0
    a = metrics.evaluate(model_a, targets, head, indices)["hit_rate_at_10"]
    b = metrics.evaluate(model_b, targets, head, indices)["hit_rate_at_10"]
    assert a["ci95_high"] > b["ci95_low"]
