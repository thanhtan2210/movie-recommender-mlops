"""PureSVD: a truncated SVD of the binary user x movie matrix.

Only the movie factors V are kept. A user is scored by folding in their row
of liked movies: scores = (x_u @ V) @ V.T. No user factors are stored, so
the model also scores users it was not trained on.
"""
import hashlib
import json
import os
from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

MODEL_FILE = "model.npz"
META_FILE = "model_meta.json"
ARTIFACT_DIR = "artifacts"
NO_RECOMMENDATION = -1


def artifact_dir(cutoff: str) -> str:
    return os.path.join(ARTIFACT_DIR, cutoff)


class PureSVD:
    def __init__(self, item_factors: np.ndarray, item_ids: np.ndarray, user_ids: Optional[np.ndarray] = None):
        self.item_factors = np.ascontiguousarray(item_factors, dtype=np.float32)  # V: n_items x k
        self.item_ids = np.asarray(item_ids)
        self.user_ids = None if user_ids is None else np.asarray(user_ids)

    @property
    def k(self) -> int:
        return self.item_factors.shape[1]

    @classmethod
    def fit(cls, X: sparse.csr_matrix, k: int, random_state: int = 42, item_ids=None, user_ids=None) -> "PureSVD":
        svd = TruncatedSVD(n_components=k, algorithm="randomized", random_state=random_state)
        svd.fit(X)
        item_ids = np.arange(X.shape[1]) if item_ids is None else item_ids
        return cls(svd.components_.T, item_ids, user_ids)

    def score(self, liked_rows: sparse.csr_matrix) -> np.ndarray:
        """Scores of every movie for each user row: (X_u @ V) @ V.T."""
        return np.asarray((liked_rows @ self.item_factors) @ self.item_factors.T, dtype=np.float32)

    def recommend(self, liked_rows: sparse.csr_matrix, seen_rows: sparse.csr_matrix, n: int = 10,
                  batch_size: int = 2048) -> np.ndarray:
        """Top-n movie columns per user, never a movie the user has already rated.

        `liked_rows` are the users' liked movies (the model input); `seen_rows`
        are all movies they rated, liked or not. Returns an (n_users, n) array
        of column indices, best first; NO_RECOMMENDATION where a user has
        fewer than n unrated movies.
        """
        out = np.empty((liked_rows.shape[0], n), dtype=np.int64)
        for start in range(0, liked_rows.shape[0], batch_size):
            stop = start + batch_size
            scores = self.score(liked_rows[start:stop])
            seen_users, seen_items = seen_rows[start:stop].nonzero()
            scores[seen_users, seen_items] = -np.inf
            out[start:stop] = top_n(scores, n)
        return out

    # ------------------------------------------------------------ persistence

    def factors_sha256(self) -> str:
        return hashlib.sha256(self.item_factors.tobytes()).hexdigest()

    def save(self, directory: str, meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(directory, exist_ok=True)
        arrays = {"V": self.item_factors, "item_ids": self.item_ids}
        if self.user_ids is not None:
            arrays["user_ids"] = self.user_ids
        np.savez(os.path.join(directory, MODEL_FILE), **arrays)
        meta = {
            "k": self.k,
            "n_items": int(len(self.item_ids)),
            "n_users": None if self.user_ids is None else int(len(self.user_ids)),
            "item_factors_sha256": self.factors_sha256(),
            **(meta or {}),
        }
        with open(os.path.join(directory, META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, directory: str) -> "PureSVD":
        with np.load(os.path.join(directory, MODEL_FILE)) as arrays:
            user_ids = arrays["user_ids"] if "user_ids" in arrays.files else None
            return cls(arrays["V"], arrays["item_ids"], user_ids)


def load_meta(directory: str) -> Dict[str, Any]:
    with open(os.path.join(directory, META_FILE), encoding="utf-8") as f:
        return json.load(f)


def top_n(scores: np.ndarray, n: int) -> np.ndarray:
    """Column indices of the n highest scores per row, best first; ties go to the lower column."""
    n = min(n, scores.shape[1])
    top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    top_scores = np.take_along_axis(scores, top, axis=1)
    order = np.lexsort((top, -top_scores), axis=1)
    top = np.take_along_axis(top, order, axis=1)
    top[np.take_along_axis(top_scores, order, axis=1) == -np.inf] = NO_RECOMMENDATION
    return top
