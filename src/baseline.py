"""Popularity baseline: the movies most liked in the last 90 days of training.

There is nothing to tune: the window length and the threshold are fixed.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from src.model import NO_RECOMMENDATION

WINDOW_DAYS = 90


def window_start(cutoff_timestamp: int) -> int:
    return cutoff_timestamp - WINDOW_DAYS * 86400


def popularity_ranking(train: pd.DataFrame, item_ids: np.ndarray, cutoff_timestamp: int,
                       positive_threshold: float) -> np.ndarray:
    """Movie columns ordered by the number of liked ratings in [cutoff - 90 days, cutoff).

    Ties, including the movies with no liked rating in the window, go to the lower column.
    """
    timestamps = train["timestamp"].to_numpy()
    recent = (timestamps >= window_start(cutoff_timestamp)) & (timestamps < cutoff_timestamp)
    liked = train["rating"].to_numpy() >= positive_threshold
    columns = np.searchsorted(item_ids, train["movieId"].to_numpy()[recent & liked])
    counts = np.bincount(columns, minlength=len(item_ids))
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
