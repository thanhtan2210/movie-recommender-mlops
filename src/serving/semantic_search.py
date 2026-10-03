import os
import pandas as pd
import numpy as np
import lancedb
from typing import List, Dict, Any


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
        self.model = model
        
        print(f"Đang kết nối tới LanceDB tại: {self.lancedb_uri}")
        if not self.lancedb_uri.startswith("s3://"):
             os.makedirs(self.lancedb_uri, exist_ok=True)
             
        self.db = lancedb.connect(self.lancedb_uri)
        
        if self.model is None:
            print("Đang tải Embedding Model mới...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.table = None

    def load_table(self):
        try:
            self.table = self.db.open_table(self.table_name)
            print(f"Đã load bảng '{self.table_name}' chứa {self.table.count_rows()} bản ghi.")
        except Exception:
            raise ValueError(f"Bảng '{self.table_name}' không tồn tại trong {self.lancedb_uri}.")

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

    # ================= LEVEL 1 =================
    def search_by_description(self, query_text: str, top_k: int = 10, use_reranker: bool = True) -> List[Dict[str, Any]]:
        if self.table is None:
            self.load_table()
        
        # KIỂM TRA ĐẦU VÀO:
        if not query_text or not query_text.strip():
            return []
            
        raw_vec = self.model.encode(query_text)
        
        # Đảm bảo đúng format list 1-D
        query_vector = np.array(raw_vec).flatten().tolist()
        
        if len(query_vector) == 0:
            raise ValueError(f"Vector sai shape: {np.array(raw_vec).shape}")
        
        fetch_k = top_k * 2 if use_reranker else top_k
        results_df = self.table.search(query_vector).metric("cosine").limit(fetch_k).to_pandas()
        
        candidates = [self._format_result(row) for _, row in results_df.iterrows()]
        if use_reranker:
            res = Reranker.rerank(candidates)[:top_k]
            return self._validate_candidates(res, "rerank")
        return self._validate_candidates(candidates[:top_k], "record")

    def search_similar_movies(self, movie_id: int, top_k: int = 5, use_reranker: bool = True) -> List[Dict[str, Any]]:
        if self.table is None:
            self.load_table()
        
        df = self.table.to_pandas()
        query_res = df[df["movieId"] == movie_id]
        
        if query_res.empty:
            raise ValueError(f"Không tìm thấy phim với ID: {movie_id}")
            
        source_vector = query_res.iloc[0]["vector"]
        
        # VALIDATE: Đảm bảo format list 1-D
        query_vector = np.array(source_vector).flatten().tolist()
        
        if len(query_vector) == 0:
            raise ValueError("Vector phim trong DB bị rỗng")
        
        fetch_k = (top_k * 2) + 1 if use_reranker else top_k + 1
        results_df = self.table.search(query_vector).metric("cosine").limit(fetch_k).to_pandas()
        filtered_df = results_df[results_df["movieId"] != movie_id]
        
        candidates = [self._format_result(row) for _, row in filtered_df.iterrows()]
        if use_reranker:
            res = Reranker.rerank(candidates)[:top_k]
            return self._validate_candidates(res, "rerank")
        return self._validate_candidates(candidates[:top_k], "record")

    def search_similar_movies_by_title(self, title: str, top_k: int = 5, use_reranker: bool = True) -> List[Dict[str, Any]]:
        if self.table is None:
            self.load_table()
        df = self.table.to_pandas()
        
        # Tìm phim gốc theo tên
        source_movie = df[df["title"].str.lower().str.contains(title.lower(), na=False)].head(1)
        if source_movie.empty:
            raise ValueError(f"Không tìm thấy phim với tên: {title}")
            
        movie_id = int(source_movie.iloc[0]["movieId"])
        source_vector = source_movie.iloc[0]["vector"]
        query_vector = np.array(source_vector, dtype=np.float32).flatten()
        
        fetch_k = (top_k * 2) + 1 if use_reranker else top_k + 1
        results_df = self.table.search(query_vector).metric("cosine").limit(fetch_k).to_pandas()
        filtered_df = results_df[results_df["movieId"] != movie_id]
        
        candidates = [self._format_result(row) for _, row in filtered_df.iterrows()]
        if use_reranker:
            res = Reranker.rerank(candidates)[:top_k]
            return self._validate_candidates(res, "rerank")
        return self._validate_candidates(candidates[:top_k], "record")

    def get_trending_by_rating(self, min_rating: float, min_votes: int, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.table is None:
            self.load_table()
        df = self.table.to_pandas()
        
        df_filtered = df[(df['avg_rating'] >= min_rating) & (df['rating_count'] >= min_votes)]
        df_sorted = df_filtered.sort_values(by=["avg_rating", "rating_count"], ascending=[False, False]).head(top_k)
        
        return [self._format_result(row) for _, row in df_sorted.iterrows()]

    # ================= LEVEL 2 =================
    def get_user_vector(self, user_movie_ratings: Dict[int, float]) -> np.ndarray:
        if self.table is None:
            self.load_table()
        df = self.table.to_pandas()
        
        vectors = []
        weights = []
        
        for mid, rating in user_movie_ratings.items():
            res = df[df["movieId"] == mid]
            if not res.empty:
                vectors.append(res.iloc[0]["vector"])
                weights.append(rating)
                
        if not vectors:
            raise ValueError("Không tìm thấy dữ liệu các phim đã xem.")
            
        user_vec = np.average(vectors, weights=weights, axis=0)
        return user_vec / np.linalg.norm(user_vec)

    def personalized_recommend(self, user_vec: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.table is None:
            self.load_table()
        
        query_vector = user_vec.tolist()
        
        results_df = self.table.search(query_vector).metric("cosine").limit(top_k * 2).to_pandas()
        candidates = [self._format_result(row) for _, row in results_df.iterrows()]
        res = Reranker.rerank(candidates)[:top_k]
        return self._validate_candidates(res, "rerank")
