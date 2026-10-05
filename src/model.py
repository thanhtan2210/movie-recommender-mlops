"""The model: PureSVD blended with recent popularity.

PureSVD is a truncated SVD of the binary user x movie matrix (1 where the
user liked the movie). Only the movie factors V are kept; a user is scored by
folding in their row of liked movies: svd(u, .) = (x_u @ V) @ V.T. The final
score adds recent popularity:

    score(u, i) = z_u(svd(u, i)) + blend_weight * z(log1p(pop90(i)))

where z_u standardises a user's SVD scores over all movies and pop90 is the
number of liked ratings in the last 90 days of training. blend_weight = 0 is
plain PureSVD; a large weight approaches the popularity ranking.
"""
import hashlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src import baseline
from src.data import build_interactions, train_window_start
from src.recommender import NO_RECOMMENDATION, Recommender, pack_csr, unpack_csr


class PureSVD:
    def __init__(self, item_factors: np.ndarray, popularity_boost: Optional[np.ndarray] = None,
                 blend_weight: float = 0.0):
        self.item_factors = np.ascontiguousarray(item_factors, dtype=np.float32)  # V: n_items x k
        # z(log1p(pop90)) per movie; only used when blend_weight > 0.
        self.popularity_boost = None if popularity_boost is None else np.asarray(popularity_boost, dtype=np.float32)
        self.blend_weight = float(blend_weight)
        if self.blend_weight and self.popularity_boost is None:
            raise ValueError("blend_weight > 0 needs popularity_boost")

    def with_blend(self, popularity_boost: np.ndarray, blend_weight: float) -> "PureSVD":
        """The same factors with a recent-popularity term added to the score."""
        return PureSVD(self.item_factors, popularity_boost, blend_weight)

    @property
    def k(self) -> int:
        return self.item_factors.shape[1]

    @classmethod
    def fit(cls, X: sparse.csr_matrix, k: int, random_state: int = 42) -> "PureSVD":
        svd = TruncatedSVD(n_components=k, algorithm="randomized", random_state=random_state)
        svd.fit(X)
        return cls(svd.components_.T)

    def score(self, liked_rows: sparse.csr_matrix) -> np.ndarray:
        """Scores of every movie for each user row: (X_u @ V) @ V.T, plus the popularity term if blended."""
        scores = np.asarray((liked_rows @ self.item_factors) @ self.item_factors.T, dtype=np.float32)
        if not self.blend_weight:
            return scores
        return standardise_rows(scores) + self.blend_weight * self.popularity_boost

    def recommend(self, liked_rows: sparse.csr_matrix, seen_rows: sparse.csr_matrix, n: int = 10,
                  batch_size: int = 2048) -> np.ndarray:
        """Top-n movie columns per user, never a movie the user has already rated.

        `liked_rows` are the users' liked movies (the model input); `seen_rows`
        are all movies they rated, liked or not. Returns an (n_users, n) array
        of column indices, best first; NO_RECOMMENDATION where a user has
        fewer than n unrated movies.
        """
        out = np.full((liked_rows.shape[0], n), NO_RECOMMENDATION, dtype=np.int64)
        for start in range(0, liked_rows.shape[0], batch_size):
            stop = start + batch_size
            scores = self.score(liked_rows[start:stop])
            seen_users, seen_items = seen_rows[start:stop].nonzero()
            scores[seen_users, seen_items] = -np.inf
            best = top_n(scores, n)
            out[start:stop, :best.shape[1]] = best  # fewer than n columns when the catalogue is smaller than n
        return out

    def factors_sha256(self) -> str:
        return hashlib.sha256(self.item_factors.tobytes()).hexdigest()


def standardise_rows(scores: np.ndarray) -> np.ndarray:
    """z-score each user's scores over all movies; a constant row (no liked movie) becomes zeros."""
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.where(std > 0, std, 1.0)


def popularity_boost(liked_counts: np.ndarray) -> np.ndarray:
    """z(log1p(count)) over movies: the popularity term of the blended score."""
    logged = np.log1p(np.asarray(liked_counts, dtype=np.float64))
    std = logged.std()
    return ((logged - logged.mean()) / (std if std > 0 else 1.0)).astype(np.float32)


def top_n(scores: np.ndarray, n: int) -> np.ndarray:
    """Column indices of the n highest scores per row, best first; ties go to the lower column."""
    n = min(n, scores.shape[1])
    top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    top_scores = np.take_along_axis(scores, top, axis=1)
    order = np.lexsort((top, -top_scores), axis=1)
    top = np.take_along_axis(top, order, axis=1)
    top[np.take_along_axis(top_scores, order, axis=1) == -np.inf] = NO_RECOMMENDATION
    return top


class BlendRecommender(Recommender):
    """PureSVD + recent popularity, with the training matrices it needs to recommend for known users."""
    model_type = "blend"

    def __init__(self, svd: Optional[PureSVD] = None, liked: Optional[sparse.csr_matrix] = None, **common):
        super().__init__(**common)
        self.svd = svd
        self.liked = liked  # 1 where the user liked the movie inside the training window (the model input)

    @classmethod
    def fit(cls, train: pd.DataFrame, cutoff: str, cutoff_timestamp: int, positive_threshold: float,
            k: int, train_window: str, blend_weight: float, random_state: int = 42) -> "BlendRecommender":
        """Fit on ratings before `cutoff`.

        The SVD sees the liked ratings of the last `train_window` ('all', or
        '<n>y'). The movie index, the already-rated matrix and the 90-day
        popularity always use every rating in `train`.
        """
        interactions = build_interactions(train, positive_threshold,
                                          liked_since=train_window_start(cutoff, train_window))
        counts = baseline.liked_counts(train, interactions.item_ids, cutoff_timestamp, positive_threshold)
        svd = PureSVD.fit(interactions.liked, k, random_state)
        if blend_weight:
            svd = svd.with_blend(popularity_boost(counts), blend_weight)
        return cls(svd=svd, liked=interactions.liked, user_ids=interactions.user_ids,
                   item_ids=interactions.item_ids, seen=interactions.seen,
                   popularity_ranking=np.argsort(-counts, kind="stable"))

    def _recommend_rows(self, rows: np.ndarray, n: int) -> np.ndarray:
        return self.svd.recommend(self.liked[rows], self.seen[rows], n)

    def personalises(self, user_id: int) -> bool:
        # A known user with no liked rating inside the training window has constant SVD
        # scores, so only the popularity term ranks their movies.
        row = self.row_of(user_id)
        return row is not None and bool(self.liked.indptr[row + 1] > self.liked.indptr[row])

    def _fold_in(self, columns: np.ndarray, n: int) -> np.ndarray:
        # The same scoring as for a known user whose liked (and rated) movies are exactly `columns`.
        shape = (1, len(self.item_ids))
        index = (np.zeros(len(columns), dtype=np.int64), columns)
        liked = sparse.csr_matrix((np.ones(len(columns), dtype=np.float32), index), shape=shape)
        seen = sparse.csr_matrix((np.ones(len(columns), dtype=np.int8), index), shape=shape)
        return self.svd.recommend(liked, seen, n)[0]

    def _extra_arrays(self) -> Dict[str, np.ndarray]:
        arrays = {"V": self.svd.item_factors, "blend_weight": np.array(self.svd.blend_weight),
                  **pack_csr("liked", self.liked)}
        if self.svd.popularity_boost is not None:
            arrays["popularity_boost"] = self.svd.popularity_boost
        return arrays

    def _restore_extra(self, arrays) -> None:
        boost = arrays["popularity_boost"] if "popularity_boost" in arrays.files else None
        self.svd = PureSVD(arrays["V"], boost, float(arrays["blend_weight"]))
        self.liked = unpack_csr(arrays, "liked", np.float32)
