import os
from typing import Any, Dict, Iterable, List, Optional

import lancedb
import numpy as np
import pandas as pd

METADATA_COLUMNS = ["movieId", "title", "genres", "overview", "poster_path", "avg_rating", "rating_count"]


class Reranker:
    @staticmethod
    def rerank(candidates: List[Dict[str, Any]], sim_weight=0.6, pop_weight=0.3, qual_weight=0.1) -> List[Dict[str, Any]]:
        """Cân bằng lại độ tương đồng (ngách) với độ phổ biến và chất lượng chung"""
        if not candidates:
            return []

        # Max values for normalization
        max_pop = max([c['rating_count'] for c in candidates]) or 1
        max_qual = 5.0

        for c in candidates:
            sim_score = c.get('similarity_score', 0.5)
            pop_score = c['rating_count'] / max_pop
            qual_score = c['avg_rating'] / max_qual

            c['final_score'] = (sim_score * sim_weight) + (pop_score * pop_weight) + (qual_score * qual_weight)

        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)

class SemanticSearchEngine:
    def __init__(self,
                 lancedb_uri: str = "lancedb_movies",
                 table_name: str = "movies",
                 model=None):
        self.lancedb_uri = lancedb_uri
        self.table_name = table_name
        # The embedding model is only needed for text queries; it is loaded
        # on first use (or explicitly with load_model()).
        self.model = model

        print(f"Đang kết nối tới LanceDB tại: {self.lancedb_uri}")
        if not self.lancedb_uri.startswith("s3://"):
             os.makedirs(self.lancedb_uri, exist_ok=True)

        self.db = lancedb.connect(self.lancedb_uri)

        self.table = None
        # Filled once by load_table(): metadata indexed by movieId, the
        # vectors as one matrix, and movieId -> row position in that matrix.
        self.catalog: Optional[pd.DataFrame] = None
        self._vectors: Optional[np.ndarray] = None
        self._row_of: Dict[int, int] = {}

    def load_model(self):
        if self.model is None:
            print("Đang tải Embedding Model mới...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model

    def load_table(self):
        try:
            self.table = self.db.open_table(self.table_name)
            print(f"Đã load bảng '{self.table_name}' chứa {self.table.count_rows()} bản ghi.")
        except Exception:
            raise ValueError(f"Bảng '{self.table_name}' không tồn tại trong {self.lancedb_uri}.")

        # Read the table into memory once; every lookup by movieId or title
        # afterwards uses this copy instead of calling to_pandas() again.
        df = self.table.to_pandas()
        self._vectors = np.vstack(df["vector"].to_numpy()).astype(np.float32)
        self._row_of = {int(movie_id): row for row, movie_id in enumerate(df["movieId"])}
        self.catalog = df.drop(columns=["vector"]).set_index("movieId", drop=False)

    def _ensure_loaded(self):
        if self.table is None or self.catalog is None:
            self.load_table()

    def _format_result(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "movie_id": int(row.get("movieId", 0)),
            "title": str(row.get("title", "Unknown")),
            "genres": str(row.get("genres", "")),
            "overview": str(row.get("overview", "")),
            "poster_path": str(row.get("poster_path", "")),
            "avg_rating": float(row.get("avg_rating", 0.0)),
            "rating_count": int(row.get("rating_count", 0)),
            "similarity_score": round(1.0 - row.get("_distance", 0.5), 4) if "_distance" in row else 0.5
        }

    def _validate_candidates(self, candidates: List[Dict[str, Any]], schema_type: str = "record") -> List[Dict[str, Any]]:
        if not candidates:
            return candidates
        try:
            from src.serving.schemas import MovieRecordSchema, RerankerOutputSchema
            df = pd.DataFrame(candidates)
            if schema_type == "rerank":
                RerankerOutputSchema.validate(df)
            else:
                MovieRecordSchema.validate(df)
        except Exception as e:
            print(f"⚠️ [Data Quality Warning] Schema validation failed: {e}")
        return candidates

    def _vector_of(self, movie_id: int) -> np.ndarray:
        row = self._row_of.get(int(movie_id))
        if row is None:
            raise ValueError(f"Không tìm thấy phim với ID: {movie_id}")
        return self._vectors[row]

    def _find_by_title(self, title: str) -> int:
        """movieId of the first movie whose title contains `title` (case-insensitive)."""
        titles = self.catalog["title"].str.lower()
        # regex=False: titles contain parentheses, e.g. "Toy Story (1995)".
        matches = self.catalog[titles.str.contains(title.lower(), regex=False, na=False)]
        if matches.empty:
            raise ValueError(f"Không tìm thấy phim với tên: {title}")
        return int(matches.iloc[0]["movieId"])

    # ================= RETRIEVAL =================
    def embed_query(self, query_text: str) -> np.ndarray:
        query_vector = np.asarray(self.load_model().encode(query_text), dtype=np.float32).flatten()
        if query_vector.size == 0:
            raise ValueError("Vector sai shape: embedding rỗng")
        return query_vector

    def retrieve_candidates(self, query_vector, n_candidates: int,
                            exclude_ids: Optional[Iterable[int]] = None) -> List[Dict[str, Any]]:
        """Nearest movies by cosine distance, without the movies in `exclude_ids`."""
        self._ensure_loaded()
        exclude = {int(movie_id) for movie_id in (exclude_ids or [])}
        query = np.asarray(query_vector, dtype=np.float32).flatten().tolist()
        if len(query) == 0:
            raise ValueError("Vector truy vấn bị rỗng")

        # Ask for enough rows to still have n_candidates after the exclusion.
        results_df = (
            self.table.search(query)
            .metric("cosine")
            .select(METADATA_COLUMNS)
            .limit(n_candidates + len(exclude))
            .to_pandas()
        )
        if exclude:
            results_df = results_df[~results_df["movieId"].isin(exclude)]
        return [self._format_result(row) for _, row in results_df.head(n_candidates).iterrows()]

    def _finish(self, candidates: List[Dict[str, Any]], top_k: int, use_reranker: bool, **weights) -> List[Dict[str, Any]]:
        if use_reranker:
            res = Reranker.rerank(candidates, **weights)[:top_k]
            return self._validate_candidates(res, "rerank")
        return self._validate_candidates(candidates[:top_k], "record")

    # ================= LEVEL 1 =================
    def search_by_description(self, query_text: str, top_k: int = 10, use_reranker: bool = True) -> List[Dict[str, Any]]:
        # KIỂM TRA ĐẦU VÀO:
        if not query_text or not query_text.strip():
            return []

        fetch_k = top_k * 2 if use_reranker else top_k
        candidates = self.retrieve_candidates(self.embed_query(query_text), fetch_k)
        return self._finish(candidates, top_k, use_reranker)

    def search_similar_movies(self, movie_id: int, top_k: int = 5, use_reranker: bool = True) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        fetch_k = top_k * 2 if use_reranker else top_k
        candidates = self.retrieve_candidates(self._vector_of(movie_id), fetch_k, exclude_ids=[movie_id])
        return self._finish(candidates, top_k, use_reranker)

    def search_similar_movies_by_title(self, title: str, top_k: int = 5, use_reranker: bool = True) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return self.search_similar_movies(self._find_by_title(title), top_k=top_k, use_reranker=use_reranker)

    def get_trending_by_rating(self, min_rating: float, min_votes: int, top_k: int = 5) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        df = self.catalog

        df_filtered = df[(df['avg_rating'] >= min_rating) & (df['rating_count'] >= min_votes)]
        df_sorted = df_filtered.sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_k)

        return [self._format_result(row) for _, row in df_sorted.iterrows()]

    # ================= LEVEL 2 =================
    def get_user_vector(self, user_movie_ratings: Dict[int, float]) -> np.ndarray:
        self._ensure_loaded()

        rows = []
        weights = []
        for mid, rating in user_movie_ratings.items():
            row = self._row_of.get(int(mid))
            if row is not None:
                rows.append(row)
                weights.append(rating)

        if not rows:
            raise ValueError("Không tìm thấy dữ liệu các phim đã xem.")

        user_vec = np.average(self._vectors[rows], weights=weights, axis=0)
        return user_vec / np.linalg.norm(user_vec)

    def personalized_recommend(self, user_vec: np.ndarray, top_k: int = 10,
                               exclude_ids: Optional[Iterable[int]] = None,
                               n_candidates: Optional[int] = None, **weights) -> List[Dict[str, Any]]:
        """Top-k for a user vector. `exclude_ids` are movies the user already picked or rated.

        `n_candidates` is the size of the pool that gets reranked (default 2 x top_k);
        `weights` are passed to the reranker (sim_weight, pop_weight, qual_weight).
        """
        candidates = self.retrieve_candidates(user_vec, n_candidates or top_k * 2, exclude_ids=exclude_ids)
        return self._finish(candidates, top_k, True, **weights)
