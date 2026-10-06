import numpy as np
from scipy import sparse

from src import baseline
from src.recommender import NO_RECOMMENDATION
from tests.conftest import CUTOFF, DAY, ratings_frame

ITEM_IDS = np.array([10, 20, 30, 40])


def test_window_is_the_90_days_before_the_cutoff():
    assert baseline.window_start(CUTOFF) == CUTOFF - 90 * DAY
    # 2019-06-01 minus 90 days is 2019-03-03.
    assert baseline.window_start(CUTOFF) == 1551571200


def test_ranking_counts_only_liked_ratings_inside_the_window():
    inside = CUTOFF - 10 * DAY
    train = ratings_frame(rows=[
        # movie 10: liked five times, but all before the window -> counts 0
        *[(user, 10, 5.0, CUTOFF - 91 * DAY) for user in range(1, 6)],
        # movie 20: liked twice inside the window
        (1, 20, 4.0, inside), (2, 20, 5.0, CUTOFF - 1),
        # movie 30: liked once inside; the first second of the window counts
        (3, 30, 4.5, CUTOFF - 90 * DAY),
        # movie 30 and 40: rated inside the window but not liked -> not counted
        (4, 30, 3.5, inside), (5, 40, 2.0, inside), (6, 40, 1.0, inside), (7, 40, 3.0, inside),
    ])

    ranking = baseline.popularity_ranking(train, ITEM_IDS, CUTOFF, positive_threshold=4.0)

    # Columns: movie 20 (2 likes), movie 30 (1), then the zero-count movies 10 and 40 in column order.
    assert ranking.tolist() == [1, 2, 0, 3]


def test_recommend_takes_the_most_popular_unrated_movies():
    ranking = np.array([1, 2, 0, 3])
    seen = sparse.csr_matrix(np.array([
        [0, 0, 0, 0],   # rated nothing
        [0, 1, 0, 0],   # rated the most popular movie
        [1, 1, 1, 0],   # only one movie left
    ], dtype=np.int8))

    recs = baseline.recommend(ranking, seen, n=2)

    assert recs.tolist() == [[1, 2], [2, 0], [3, NO_RECOMMENDATION]]


def test_liked_counts_are_what_the_ranking_sorts():
    inside = CUTOFF - 10 * DAY
    train = ratings_frame(rows=[(1, 20, 4.0, inside), (2, 20, 5.0, inside), (3, 30, 4.5, inside),
                                (4, 10, 5.0, CUTOFF - 200 * DAY), (5, 40, 1.0, inside)])

    assert baseline.liked_counts(train, ITEM_IDS, CUTOFF, positive_threshold=4.0).tolist() == [0, 2, 1, 0]
