import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.serving.semantic_search import SemanticSearchEngine

class DummyModel:
    def encode(self, text: str):
        return [0.1] * 384

class MockTable:
    def __init__(self, df):
        self._df = df
        self._is_search = False

    def count_rows(self):
        return len(self._df)

    def search(self, vector):
        self._is_search = True
        return self

    def metric(self, name):
        return self

    def limit(self, k):
        return self

    def to_pandas(self):
        if self._is_search:
            self._is_search = False
            df_copy = self._df.copy()
            df_copy["_distance"] = 0.1
            return df_copy
        return self._df

@pytest.fixture
def mock_search_engine():
    # Create sample movie data in a pandas DataFrame
    dim = 384
    np.random.seed(42)
    
    data = []
    titles = ["The Matrix", "Inception", "Toy Story", "Pulp Fiction", "The Dark Knight"]
    genres_list = ["Action|Sci-Fi", "Action|Sci-Fi", "Animation|Adventure|Comedy", "Crime|Drama", "Action|Crime|Drama"]
    overviews = ["Neo learns the truth", "Dreams within dreams", "Toys come to life", "Intertwining stories", "Batman battles Joker"]
    
    for i in range(5):
        vec = np.random.rand(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        data.append({
            "movieId": i + 1,
            "title": titles[i],
            "genres": genres_list[i],
            "overview": overviews[i],
            "poster_path": f"/poster_{i+1}.jpg",
            "avg_rating": 4.0 + (i * 0.2),  # 4.0, 4.2, 4.4, 4.6, 4.8
            "rating_count": 100 * (i + 1),   # 100, 200, 300, 400, 500
            "vector": vec.tolist()
        })
        
    df = pd.DataFrame(data)
    mock_table = MockTable(df)
    
    # We patch lancedb.connect to return a mock DB object
    with patch("lancedb.connect") as mock_connect:
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_connect.return_value = mock_db
        
        engine = SemanticSearchEngine(lancedb_uri="mock_uri", model=DummyModel())
        engine.table = mock_table
        yield engine

def test_format_result(mock_search_engine):
    row = pd.Series({
        "movieId": 42,
        "title": "Test Movie",
        "genres": "Sci-Fi",
        "overview": "Overview",
        "poster_path": "/path.jpg",
        "avg_rating": 4.5,
        "rating_count": 120,
        "_distance": 0.3
    })
    
    res = mock_search_engine._format_result(row)
    assert res["movie_id"] == 42
    assert res["title"] == "Test Movie"
    assert res["similarity_score"] == pytest.approx(0.7)

def test_search_by_description_empty(mock_search_engine):
    assert mock_search_engine.search_by_description("") == []
    assert mock_search_engine.search_by_description("   ") == []

def test_search_by_description_success(mock_search_engine):
    results = mock_search_engine.search_by_description("find some cool movie", top_k=2, use_reranker=False)
    
    assert len(results) == 2
    assert "movie_id" in results[0]
    assert "similarity_score" in results[0]
    # Check that schema validation was applied (MovieRecordSchema)
    assert "final_score" not in results[0]

def test_search_by_description_reranker_success(mock_search_engine):
    results = mock_search_engine.search_by_description("find some cool movie", top_k=2, use_reranker=True)
    
    assert len(results) == 2
    # Reranked output must have final_score
    assert "final_score" in results[0]

def test_search_similar_movies_success(mock_search_engine):
    # Find similar to movie ID 1 (The Matrix)
    results = mock_search_engine.search_similar_movies(movie_id=1, top_k=2, use_reranker=False)
    
    assert len(results) == 2
    # Output should not contain the source movie itself
    assert all(r["movie_id"] != 1 for r in results)

def test_search_similar_movies_by_title_success(mock_search_engine):
    results = mock_search_engine.search_similar_movies_by_title(title="matrix", top_k=2, use_reranker=False)
    
    assert len(results) == 2
    assert all(r["title"] != "The Matrix" for r in results)

def test_search_similar_movies_by_title_not_found(mock_search_engine):
    with pytest.raises(ValueError, match="Không tìm thấy phim"):
        mock_search_engine.search_similar_movies_by_title("Avengers")

def test_get_user_vector_and_personalized_recommend(mock_search_engine):
    user_ratings = {1: 4.0, 2: 5.0}
    user_vec = mock_search_engine.get_user_vector(user_ratings)
    
    assert user_vec.shape == (384,)
    assert np.isclose(np.linalg.norm(user_vec), 1.0)
    
    recs = mock_search_engine.personalized_recommend(user_vec, top_k=2)
    assert len(recs) == 2
    # Personalized recommendations are always reranked
    assert "final_score" in recs[0]
