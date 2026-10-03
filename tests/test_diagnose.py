import json
import os

import numpy as np
import pandas as pd
import pytest

from src import diagnose
from src.model import NO_RECOMMENDATION
from tests.test_train_evaluate import run_evaluate, run_train, workspace  # noqa: F401  (fixture)

MOVIES = pd.DataFrame({
    "movieId": [10, 20, 30, 40],
    "title": ["Old One (1994)", "Blade Runner 2049 (2017)", "New One (2018)", "No Year Given"],
})
ITEM_IDS = np.array([10, 20, 30, 40])


def test_release_year_comes_from_the_parentheses_in_the_title():
    years = diagnose.release_years(MOVIES, ITEM_IDS)

    assert years[:3].tolist() == [1994.0, 2017.0, 2018.0]   # "2049" is not in parentheses
    assert np.isnan(years[3])
    # Columns follow item_ids, not the order of the movies table.
    assert diagnose.release_years(MOVIES.iloc[::-1], ITEM_IDS)[:3].tolist() == [1994.0, 2017.0, 2018.0]


def test_year_profile_counts_repeated_movies_and_skips_missing_years():
    years = diagnose.release_years(MOVIES, ITEM_IDS)

    profile = diagnose.year_profile(np.array([0, 2, 2, 3]), years)   # 1994, 2018, 2018, no year

    assert profile == {"movies": 4, "without_year": 1,
                       "share_released_from_2018": pytest.approx(2 / 3), "median_year": 2018.0}


def test_metrics_on_subset_only_counts_the_given_targets():
    recommended = np.array([[0, 1], [2, 3], [1, 0]])
    new_targets = np.array([[False, False, True, False],     # user 0: new target (col 2) not recommended
                            [False, False, True, False],     # user 1: new target recommended
                            [False, False, False, False]])   # user 2: has no new target -> not evaluated

    result = diagnose.metrics_on_subset(recommended, new_targets)

    assert result["users"] == 2 and result["targets"] == 2
    assert result["hit_rate_at_10"]["value"] == 0.5
    assert result["recall_at_10"]["value"] == 0.5
    assert diagnose.metrics_on_subset(recommended, np.zeros_like(new_targets)) == {"users": 0, "targets": 0}


def test_long_tail_share_can_leave_out_new_movies():
    recommended = np.array([[0, 1, 2], [2, 3, NO_RECOMMENDATION]])
    head = np.array([True, False, False, False])
    not_new = np.array([True, True, False, True])              # column 2 is a new movie

    everything = diagnose.long_tail_share(recommended, head, np.ones(4, dtype=bool))
    without_new = diagnose.long_tail_share(recommended, head, not_new)

    assert everything == {"recommendations": 5, "long_tail_share": pytest.approx(4 / 5)}
    assert without_new == {"recommendations": 3, "long_tail_share": pytest.approx(2 / 3)}


def test_diagnose_end_to_end(workspace, monkeypatch):  # noqa: F811
    run_train(monkeypatch)
    run_evaluate(monkeypatch)
    first_report = open(os.path.join("reports", "2019-06-01", "test_metrics.json"), encoding="utf-8").read()
    monkeypatch.setattr("sys.argv", ["diagnose", "--cutoff", "2019-06-01"])

    diagnose.main()

    with open(os.path.join("reports", "2019-06-01", "diagnostics.json"), encoding="utf-8") as f:
        report = json.load(f)
    assert set(report["release_year"]) == {"train_liked_ratings_all_time", "train_liked_ratings_last_90_days",
                                           "test_targets", "pure_svd_top_10", "popularity_top_10"}
    # Half of the synthetic movies are labelled 2018.
    assert report["catalog"]["new_movies"] == 20
    for label in ("new", "old"):
        for name in ("pure_svd", "popularity"):
            assert report["by_target_age"][label][name]["users"] > 0
    assert set(report["long_tail_share"]["popularity"]) == {
        "all_recommendations", "excluding_movies_from_2018", "only_movies_from_2018"}
    # Diagnostics measure; they do not touch the test report.
    assert open(os.path.join("reports", "2019-06-01", "test_metrics.json"), encoding="utf-8").read() == first_report
