"""Tests of the offline evaluation protocol on synthetic ratings and the fixture database."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_offline as ev
from tests.conftest import MOVIES, write_synthetic_ratings

CATALOG_IDS = np.array(sorted(m[0] for m in MOVIES))


def ratings_frame(rows):
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    return frame.astype(ev.RATING_DTYPES)


@pytest.fixture
def ratings_csv(tmp_path):
    rows = [
        # user 1: three liked catalogue movies
        (1, 1, 5.0, 100), (1, 2, 4.0, 200), (1, 6, 4.5, 300),
        # user 2: one liked catalogue movie, one disliked, one liked movie outside the catalogue
        (2, 1, 4.0, 100), (2, 2, 2.0, 200), (2, 999999, 5.0, 300),
        # user 3: two liked
        (3, 32, 4.0, 100), (3, 50, 5.0, 200),
    ]
    path = tmp_path / "ratings.csv"
    ratings_frame(rows).to_csv(path, index=False)
    return str(path)


def test_count_liked_ignores_low_ratings_and_movies_outside_the_catalog(ratings_csv, monkeypatch):
    monkeypatch.setattr(ev, "CHUNK_ROWS", 3)  # force several chunks
    liked = ev.count_liked_per_user(ratings_csv, CATALOG_IDS)

    assert liked[1] == 3
    assert liked[2] == 1
    assert liked[3] == 2


def test_sample_users_is_deterministic_and_respects_the_minimum(monkeypatch):
    monkeypatch.setattr(ev, "MIN_LIKED", 2)
    liked = np.array([0, 5, 1, 2, 9, 2, 0, 3])

    assert list(ev.sample_users(liked, n_users=100)) == [1, 3, 4, 5, 7]
    first = ev.sample_users(liked, n_users=3)
    assert len(first) == 3
    assert list(first) == list(ev.sample_users(liked, n_users=3))
    assert set(first) <= {1, 3, 4, 5, 7}


def test_load_user_ratings_keeps_only_sampled_users_and_catalog_movies(ratings_csv, monkeypatch):
    monkeypatch.setattr(ev, "CHUNK_ROWS", 3)
    loaded = ev.load_user_ratings(ratings_csv, np.array([2, 3]), CATALOG_IDS)

    assert set(loaded["userId"]) == {2, 3}
    assert 999999 not in set(loaded["movieId"])
    assert len(loaded) == 4


def test_build_cases_hides_the_last_liked_movie():
    ratings = ratings_frame([
        (7, 1, 5.0, 100),     # liked seed
        (7, 2, 2.0, 150),     # disliked, but already seen -> excluded
        (7, 6, 4.0, 200),     # liked seed
        (7, 32, 4.5, 300),    # last liked -> target
        (7, 50, 3.0, 300),    # rated at the target's timestamp -> excluded
        (7, 296, 1.0, 400),   # rated after the target -> still a candidate
    ])

    (case,) = ev.build_cases(ratings)

    assert case.user_id == 7
    assert case.target == 32
    assert case.seeds == {1: 5.0, 6: 4.0}
    assert sorted(case.exclude) == [1, 2, 6, 50]
    # No leakage: the target is neither a seed nor excluded.
    assert case.target not in case.seeds
    assert case.target not in case.exclude
    assert 296 not in case.exclude


def test_build_cases_breaks_timestamp_ties_deterministically():
    ratings = ratings_frame([(8, 50, 5.0, 100), (8, 6, 5.0, 100), (8, 1, 4.0, 50)])

    (case,) = ev.build_cases(ratings)

    assert case.target == 50  # same timestamp: the larger movieId is last
    assert case.seeds == {1: 4.0, 6: 5.0}


def test_build_cases_skips_users_without_a_seed():
    assert ev.build_cases(ratings_frame([(9, 1, 5.0, 100), (9, 2, 1.0, 200)])) == []


def test_ndcg_with_a_single_relevant_item():
    assert ev.ndcg_at_k([5, 6, 7], 5) == 1.0
    assert ev.ndcg_at_k([5, 6, 7], 7) == pytest.approx(1 / math.log2(4))
    assert ev.ndcg_at_k([5, 6, 7], 8) == 0.0


def test_recommend_popular_skips_seen_movies():
    assert ev.recommend_popular([10, 20, 30, 40], exclude=[20], top_k=2) == [10, 30]


def test_reranker_weights_sum_to_one_and_match_the_default():
    for pop_weight in ev.POP_WEIGHTS:
        assert sum(ev.reranker_weights(pop_weight).values()) == pytest.approx(1.0)
    assert ev.reranker_weights(ev.DEFAULT_POP_WEIGHT) == {"sim_weight": 0.6, "pop_weight": 0.3, "qual_weight": 0.1}


def test_bootstrap_intervals_bracket_the_estimate_and_are_reproducible():
    values = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 25)
    indices = ev.bootstrap_indices(len(values))

    result = ev.mean_with_ci(values, indices)
    assert result["ci95_low"] <= result["value"] <= result["ci95_high"]
    assert result == ev.mean_with_ci(values, ev.bootstrap_indices(len(values)))

    ratio = ev.ratio_with_ci(values, np.ones_like(values), indices)
    assert ratio["value"] == pytest.approx(values.mean())


def test_evaluate_end_to_end_on_the_fixture_database(engine):
    cases = [
        ev.Case(user_id=1, target=79132, seeds={2571: 5.0, 541: 4.0}, exclude=[2571, 541, 2]),
        ev.Case(user_id=2, target=318, seeds={296: 5.0, 50: 4.5}, exclude=[296, 50]),
        ev.Case(user_id=3, target=100001, seeds={1: 4.0}, exclude=[1]),
    ]

    report = ev.evaluate(engine, cases, n_candidates=6)

    assert set(report["results"]) == {"popularity", "content_default", "content_as_served"}
    assert [row["pop_weight"] for row in report["tradeoff"]] == ev.POP_WEIGHTS
    for result in report["results"].values():
        assert 0.0 <= result["hit_rate_at_10"]["value"] <= 1.0
        assert 0.0 <= result["long_tail_share"]["value"] <= 1.0
        assert 0.0 < result["catalog_coverage"]["value"] <= 1.0
    # The default row of the sweep is the default recommender.
    default_row = next(row for row in report["tradeoff"] if row["pop_weight"] == ev.DEFAULT_POP_WEIGHT)
    assert default_row["hit_rate_at_10"] == report["results"]["content_default"]["hit_rate_at_10"]["value"]
    # The catalogue has 12 movies, so some users cannot get 6 unseen candidates... but 9 remain here.
    assert report["users_with_fewer_candidates_than_requested"] == 0
    # 20% of 12 movies, rounded up.
    assert report["head_movies"] == 3


def test_evaluate_never_recommends_excluded_movies(engine, monkeypatch):
    seen = {}
    original = ev.recommend_content

    def spy(pool, pop_weight, top_k=ev.TOP_K):
        recs = original(pool, pop_weight, top_k)
        seen.setdefault("recs", []).append(recs)
        return recs

    monkeypatch.setattr(ev, "recommend_content", spy)
    exclude = [2571, 541, 2, 1, 6]
    ev.evaluate(engine, [ev.Case(1, 79132, {2571: 5.0, 541: 4.0}, exclude)], n_candidates=5)

    assert seen["recs"]
    for recs in seen["recs"]:
        assert not set(exclude) & set(recs)


def test_popularity_only_weight_ranks_the_pool_by_rating_count(engine):
    pool = engine.retrieve_candidates(engine.get_user_vector({1: 5.0}), 11, exclude_ids=[1])

    ranked = ev.recommend_content(pool, pop_weight=0.9, top_k=3)  # similarity weight 0

    by_count = sorted(pool, key=lambda m: (-(0.9 * m["rating_count"] / max(p["rating_count"] for p in pool)
                                             + 0.1 * m["avg_rating"] / 5.0)))
    assert ranked == [m["movie_id"] for m in by_count[:3]]


@pytest.fixture
def small_run(lancedb_dir, tmp_path, monkeypatch):
    """Run main() on the fixture database with a handful of synthetic users."""
    ratings_path = tmp_path / "ratings.csv"
    write_synthetic_ratings(ratings_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ev, "MIN_LIKED", 3)
    monkeypatch.setattr(ev, "N_USERS", 4)
    monkeypatch.setattr(ev, "N_CANDIDATES", 5)
    monkeypatch.setattr(ev, "BOOTSTRAP_SAMPLES", 50)
    monkeypatch.setattr(sys, "argv", ["evaluate_offline", "--db", lancedb_dir, "--ratings", str(ratings_path)])
    return tmp_path


def test_main_writes_the_reports(small_run, monkeypatch):
    monkeypatch.setattr(ev, "SUSPICIOUS_HIT_RATE", 1.1)  # 12 movies: a high hit rate is expected here

    ev.main()

    with open(small_run / ev.EVAL_PATH, encoding="utf-8") as f:
        report = json.load(f)
    assert report["data"]["evaluated_users"] == 4
    assert report["data"]["eligible_users"] == 6
    assert report["protocol"]["seed"] == ev.SEED
    tradeoff = pd.read_csv(small_run / ev.TRADEOFF_PATH)
    assert list(tradeoff["pop_weight"]) == ev.POP_WEIGHTS

    first = (small_run / ev.EVAL_PATH).read_text(encoding="utf-8")
    ev.main()
    assert (small_run / ev.EVAL_PATH).read_text(encoding="utf-8") == first


def test_main_refuses_to_write_an_implausible_result(small_run, monkeypatch):
    monkeypatch.setattr(ev, "SUSPICIOUS_HIT_RATE", -1.0)

    with pytest.raises(SystemExit, match="suspect leakage"):
        ev.main()

    assert not os.path.exists(small_run / ev.EVAL_PATH)


def test_main_needs_a_database_or_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["evaluate_offline", "--db", "missing_db"])
    monkeypatch.setattr(ev, "load_dotenv", lambda: None)

    with pytest.raises(SystemExit, match="nothing to evaluate"):
        ev.main()
