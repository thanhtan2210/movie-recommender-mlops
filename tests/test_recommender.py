"""The two recommenders: same interface, cold-start fallback, state on disk."""
import numpy as np
import pytest

from src.baseline import PopularityRecommender
from src.data import build_interactions, train_window_start
from src.model import BlendRecommender
from src.recommender import NO_RECOMMENDATION
from tests.conftest import CUTOFF, DAY, ratings_frame

RECENT = CUTOFF - 10 * DAY
OLD = CUTOFF - 500 * DAY

# Users 1-3 like movies 10 and 20, users 4-6 like movies 30 and 40; movie 50 is
# the most liked recently. User 7 liked movie 10 long ago and rated 20 badly.
ROWS = [
    *[(u, m, 5.0, RECENT) for u in (1, 2, 3) for m in (10, 20)],
    *[(u, m, 5.0, RECENT) for u in (4, 5, 6) for m in (30, 40)],
    *[(u, 50, 4.0, RECENT) for u in (1, 2, 3, 4, 5)],
    (7, 10, 5.0, OLD), (7, 20, 1.0, RECENT),
]


@pytest.fixture
def train():
    return ratings_frame(rows=ROWS)


@pytest.fixture
def blend(train):
    return BlendRecommender.fit(train, "2019-06-01", CUTOFF, 4.0, k=2, train_window="1y", blend_weight=0.5)


@pytest.fixture
def popularity(train):
    return PopularityRecommender.fit(train, CUTOFF, 4.0)


def test_train_window_start():
    assert train_window_start("2019-06-01", "all") is None
    assert train_window_start("2019-06-01", "1y") == 1527811200    # 2018-06-01T00:00:00Z
    assert train_window_start("2019-06-01", "3y") == 1464739200    # 2016-06-01T00:00:00Z
    assert train_window_start("2020-02-29", "1y") == 1551312000    # 2019-02-28T00:00:00Z
    with pytest.raises(ValueError):
        train_window_start("2019-06-01", "6m")


def test_liked_window_keeps_the_index_and_the_seen_movies(train):
    everything = build_interactions(train, 4.0)
    recent = build_interactions(train, 4.0, liked_since=train_window_start("2019-06-01", "1y"))

    user_7 = int(np.searchsorted(recent.user_ids, 7))
    assert recent.user_ids.tolist() == everything.user_ids.tolist()
    assert recent.item_ids.tolist() == everything.item_ids.tolist() == [10, 20, 30, 40, 50]
    assert everything.liked[user_7].toarray().tolist() == [[1.0, 0.0, 0.0, 0.0, 0.0]]
    assert recent.liked[user_7].toarray().tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0]]   # the old like is dropped
    assert recent.seen[user_7].toarray().tolist() == [[1, 1, 0, 0, 0]]              # but it is still "seen"


def test_popularity_recommends_the_most_liked_unrated_movies(popularity):
    # Liked in the last 90 days: movie 50 five times, movies 10-40 three times each.
    assert popularity.item_ids[popularity.popularity_ranking].tolist() == [50, 10, 20, 30, 40]

    recs = popularity.recommend([1, 6], n=2)

    assert recs.tolist() == [[30, 40], [50, 10]]     # user 1 rated 10, 20, 50; user 6 rated 30, 40


def test_blend_never_recommends_a_rated_movie(blend, train):
    for user in (1, 4, 7):
        rated = set(train.loc[train["userId"] == user, "movieId"])
        recs = [m for m in blend.recommend([user], n=5)[0].tolist() if m != NO_RECOMMENDATION]
        assert recs and not rated & set(recs)


def test_blend_follows_the_users_taste(blend):
    # User 6 liked 30 and 40 like users 4 and 5, who also liked 50.
    assert blend.recommend([6], n=1).tolist() == [[50]]


def test_both_recommenders_fall_back_to_popularity_for_unknown_users(blend, popularity):
    expected = [50, 10, 20]

    assert blend.recommend([999], n=3).tolist() == [expected]
    assert popularity.recommend([999], n=3).tolist() == [expected]
    # Known and unknown users in one call keep their order.
    mixed = blend.recommend([999, 6, 0], n=1)
    assert mixed.tolist() == [[50], [50], [50]]


def test_a_user_with_no_liked_movie_in_the_window_gets_popularity_minus_rated(blend):
    # User 7 only has an old like: the SVD scores are constant, so popularity decides.
    assert blend.recommend([7], n=3).tolist() == [[50, 30, 40]]


def test_personalises_only_with_liked_ratings_in_the_window(blend, popularity):
    assert blend.personalises(6) is True
    assert blend.personalises(7) is False        # known, but the only like is older than the window
    assert blend.personalises(999) is False      # unknown
    assert popularity.personalises(6) is False   # the baseline never personalises
    assert blend.row_of(999) is None and blend.row_of(1) == 0


def test_empty_slots_are_marked(popularity):
    assert popularity.recommend([1], n=4).tolist() == [[30, 40, NO_RECOMMENDATION, NO_RECOMMENDATION]]


@pytest.mark.parametrize("name", ["blend", "popularity"])
def test_state_round_trip_gives_the_same_recommendations(name, request, tmp_path):
    recommender = request.getfixturevalue(name)
    users = [1, 2, 3, 4, 5, 6, 7, 999]

    path = recommender.save_state(str(tmp_path / name))
    restored = type(recommender).load_state(path)

    assert np.array_equal(restored.recommend(users, n=3), recommender.recommend(users, n=3))
    assert restored.model_type == name


def test_blend_state_keeps_the_model_parameters(blend, tmp_path):
    restored = BlendRecommender.load_state(blend.save_state(str(tmp_path / "blend")))

    assert restored.svd.blend_weight == 0.5
    assert np.array_equal(restored.svd.item_factors, blend.svd.item_factors)
    assert (restored.liked != blend.liked).nnz == 0


def test_predict_accepts_the_pyfunc_inputs(blend):
    import pandas as pd

    expected = blend.recommend([1, 999], n=2)

    assert np.array_equal(blend.predict(None, [1, 999], params={"n": 2}), expected)
    assert np.array_equal(blend.predict(None, np.array([1, 999]), params={"n": 2}), expected)
    assert np.array_equal(blend.predict(None, pd.DataFrame({"userId": [1, 999]}), params={"n": 2}), expected)
    assert blend.predict(None, [1]).shape == (1, 10)
