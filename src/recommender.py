"""Common base of the recommenders, usable directly and as an MLflow pyfunc model.

    recommend(user_ids, n=10) -> (n_users, n) array of movieIds, best first

A movie the user rated in the training data is never returned. A user who is
not in the training data gets the popularity list (cold-start fallback).
NO_RECOMMENDATION fills the slots of a user with fewer than n movies left.
"""
import os
from typing import Any, Dict, Optional

import mlflow.pyfunc
import numpy as np
from scipy import sparse

NO_RECOMMENDATION = -1
STATE_FILE = "state.npz"
STATE_ARTIFACT = "state"


def pack_csr(name: str, matrix: sparse.csr_matrix) -> Dict[str, np.ndarray]:
    """A binary csr matrix as arrays for np.savez (the values are all 1, so they are not stored)."""
    matrix = matrix.tocsr()
    return {f"{name}_indices": matrix.indices.astype(np.int32), f"{name}_indptr": matrix.indptr.astype(np.int64),
            f"{name}_shape": np.array(matrix.shape, dtype=np.int64)}


def unpack_csr(arrays, name: str, dtype) -> sparse.csr_matrix:
    indices, indptr = arrays[f"{name}_indices"], arrays[f"{name}_indptr"]
    return sparse.csr_matrix((np.ones(len(indices), dtype=dtype), indices, indptr), shape=tuple(arrays[f"{name}_shape"]))


class Recommender(mlflow.pyfunc.PythonModel):
    model_type = "base"

    def __init__(self, user_ids: Optional[np.ndarray] = None, item_ids: Optional[np.ndarray] = None,
                 seen: Optional[sparse.csr_matrix] = None, popularity_ranking: Optional[np.ndarray] = None):
        self.user_ids = user_ids  # sorted; row i of the matrices is user_ids[i]
        self.item_ids = item_ids  # sorted; column j is item_ids[j]
        self.seen = seen  # 1 where the user rated the movie in training
        self.popularity_ranking = popularity_ranking  # movie columns, most liked recently first

    # ------------------------------------------------------------ recommending

    def _recommend_rows(self, rows: np.ndarray, n: int) -> np.ndarray:
        """Top-n movie columns for users known from training (rows of the matrices)."""
        raise NotImplementedError

    def recommend_columns(self, user_ids, n: int = 10) -> np.ndarray:
        """Like recommend(), but returns movie columns instead of movieIds."""
        user_ids = np.asarray(user_ids).ravel()
        positions = np.minimum(np.searchsorted(self.user_ids, user_ids), len(self.user_ids) - 1)
        known = self.user_ids[positions] == user_ids
        out = np.full((len(user_ids), n), NO_RECOMMENDATION, dtype=np.int64)
        if known.any():
            out[known] = self._recommend_rows(positions[known], n)
        if not known.all():
            fallback = self.popularity_ranking[:n]
            out[~known, :len(fallback)] = fallback
        return out

    def recommend(self, user_ids, n: int = 10) -> np.ndarray:
        columns = self.recommend_columns(user_ids, n)
        return np.where(columns == NO_RECOMMENDATION, NO_RECOMMENDATION, self.item_ids[np.maximum(columns, 0)])

    # ------------------------------------------------------------ state on disk

    def _extra_arrays(self) -> Dict[str, np.ndarray]:
        return {}

    def _restore_extra(self, arrays) -> None:
        pass

    def save_state(self, directory: str) -> str:
        """Write everything needed to recommend into one .npz file; returns its path."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, STATE_FILE)
        np.savez_compressed(path, user_ids=self.user_ids, item_ids=self.item_ids,
                            popularity_ranking=self.popularity_ranking, **pack_csr("seen", self.seen),
                            **self._extra_arrays())
        return path

    def restore(self, path: str) -> "Recommender":
        with np.load(path) as arrays:
            self.user_ids, self.item_ids = arrays["user_ids"], arrays["item_ids"]
            self.popularity_ranking = arrays["popularity_ranking"]
            self.seen = unpack_csr(arrays, "seen", np.int8)
            self._restore_extra(arrays)
        return self

    @classmethod
    def load_state(cls, path: str) -> "Recommender":
        return cls().restore(path)

    # ------------------------------------------------------------ MLflow pyfunc

    def __getstate__(self) -> Dict[str, Any]:
        # The pickled pyfunc object carries no data; load_context() reads the state artifact.
        return {}

    def load_context(self, context) -> None:
        self.restore(context.artifacts[STATE_ARTIFACT])

    def predict(self, context, model_input, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """model_input: user ids (list, array, or a DataFrame whose first column holds them); params: {"n": 10}."""
        if hasattr(model_input, "iloc"):
            model_input = model_input.iloc[:, 0].to_numpy()
        return self.recommend(model_input, int((params or {}).get("n", 10)))
