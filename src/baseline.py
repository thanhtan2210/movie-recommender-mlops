"""Popularity baseline: the movies most liked in the last 90 days of training.

There is nothing to tune: the window length and the threshold are fixed.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from src.data import build_interactions
from src.recommender import NO_RECOMMENDATION, Recommender

WINDOW_DAYS = 90


def window_start(cutoff_timestamp: int) -> int:
    return cutoff_timestamp - WINDOW_DAYS * 86400


def liked_counts(train: pd.DataFrame, item_ids: np.ndarray, cutoff_timestamp: int,
                 positive_threshold: float) -> np.ndarray:
    """Per movie column: the number of liked ratings in [cutoff - 90 days, cutoff)."""
    timestamps = train["timestamp"].to_numpy()
    recent = (timestamps >= window_start(cutoff_timestamp)) & (timestamps < cutoff_timestamp)
    liked = train["rating"].to_numpy() >= positive_threshold
    columns = np.searchsorted(item_ids, train["movieId"].to_numpy()[recent & liked])
    return np.bincount(columns, minlength=len(item_ids))


def popularity_ranking(train: pd.DataFrame, item_ids: np.ndarray, cutoff_timestamp: int,
                       positive_threshold: float) -> np.ndarray:
    """Movie columns ordered by the number of liked ratings in [cutoff - 90 days, cutoff).

    Ties, including the movies with no liked rating in the window, go to the lower column.
    """
    counts = liked_counts(train, item_ids, cutoff_timestamp, positive_threshold)
    return np.argsort(-counts, kind="stable")


def recommend(ranking: np.ndarray, seen_rows: sparse.csr_matrix, n: int = 10) -> np.ndarray:
    """For each user, the first n movies of the ranking that the user has not rated."""
    out = np.full((seen_rows.shape[0], n), NO_RECOMMENDATION, dtype=np.int64)
    for user in range(seen_rows.shape[0]):
        seen = seen_rows.indices[seen_rows.indptr[user]:seen_rows.indptr[user + 1]]
        # At most len(seen) of the leading movies can be excluded.
        head = ranking[:n + len(seen)]
        picks = head[~np.isin(head, seen)][:n]
        out[user, :len(picks)] = picks
    return out


class PopularityRecommender(Recommender):
    """The popularity ranking, minus what each user has already rated."""
    model_type = "popularity"

    @classmethod
    def fit(cls, train: pd.DataFrame, cutoff_timestamp: int, positive_threshold: float) -> "PopularityRecommender":
        interactions = build_interactions(train, positive_threshold)
        ranking = popularity_ranking(train, interactions.item_ids, cutoff_timestamp, positive_threshold)
        return cls(user_ids=interactions.user_ids, item_ids=interactions.item_ids, seen=interactions.seen,
                   popularity_ranking=ranking)

    def _recommend_rows(self, rows: np.ndarray, n: int) -> np.ndarray:
        return recommend(self.popularity_ranking, self.seen[rows], n)
