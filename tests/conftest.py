import lancedb
import numpy as np
import pyarrow as pa
import pytest
from typing import List, Dict, Any

from src.serving.semantic_search import SemanticSearchEngine

VECTOR_DIM = 8

# (movieId, title, genres, avg_rating, rating_count)
MOVIES = [
    (1, "Toy Story (1995)", "Animation|Children|Comedy", 3.9, 57309),
    (2, "Jumanji (1995)", "Adventure|Children|Fantasy", 3.3, 24228),
    (6, "Heat (1995)", "Action|Crime|Thriller", 3.9, 24588),
    (32, "Twelve Monkeys (a.k.a. 12 Monkeys) (1995)", "Mystery|Sci-Fi|Thriller", 3.9, 47054),
    (50, "Usual Suspects, The (1995)", "Crime|Mystery|Thriller", 4.3, 55366),
    (296, "Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller", 4.2, 79672),
    (318, "Shawshank Redemption, The (1994)", "Crime|Drama", 4.4, 81482),
    (541, "Blade Runner (1982)", "Action|Sci-Fi|Thriller", 4.1, 34366),
    (2571, "Matrix, The (1999)", "Action|Sci-Fi|Thriller", 4.2, 72674),
    (69757, "(500) Days of Summer (2009)", "Comedy|Drama|Romance", 3.7, 9300),
    (79132, "Inception (2010)", "Action|Crime|Drama|Sci-Fi", 4.2, 38895),
    (100001, "Obscure Indie Film (2012)", "Drama", 3.1, 60),
]


class DummyModel:
    """Stands in for SentenceTransformer: a fixed vector, no download."""

    def encode(self, text: str):
        return np.linspace(0.1, 0.8, VECTOR_DIM, dtype=np.float32)


def make_movie_rows() -> List[Dict[str, Any]]:
    rng = np.random.default_rng(42)
    rows = []
    for movie_id, title, genres, avg_rating, rating_count in MOVIES:
        vector = rng.random(VECTOR_DIM, dtype=np.float32)
        rows.append({
            "movieId": movie_id,
            "title": title,
            "genres": genres,
            "overview": f"Overview of {title}",
            "poster_path": f"/poster_{movie_id}.jpg",
            "avg_rating": avg_rating,
            "rating_count": rating_count,
            "vector": (vector / np.linalg.norm(vector)).tolist(),
        })
    return rows


def create_movies_table(db_dir: str) -> None:
    """A small LanceDB table with the same schema as the real database."""
    schema = pa.schema([
        pa.field("movieId", pa.int32()),
        pa.field("title", pa.string()),
        pa.field("genres", pa.string()),
        pa.field("overview", pa.string()),
        pa.field("poster_path", pa.string()),
        pa.field("avg_rating", pa.float32()),
        pa.field("rating_count", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ])
    lancedb.connect(db_dir).create_table("movies", data=make_movie_rows(), schema=schema, mode="overwrite")


@pytest.fixture
def lancedb_dir(tmp_path) -> str:
    db_dir = str(tmp_path / "lancedb_movies")
    create_movies_table(db_dir)
    return db_dir


@pytest.fixture
def engine(lancedb_dir) -> SemanticSearchEngine:
    engine = SemanticSearchEngine(lancedb_uri=lancedb_dir, model=DummyModel())
    engine.load_table()
    return engine


@pytest.fixture
def mock_movie_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "movie_id": 1,
            "title": "Movie A",
            "genres": "Action|Sci-Fi",
            "overview": "Overview A",
            "poster_path": "/pathA.jpg",
            "avg_rating": 4.5,
            "rating_count": 1000,
            "similarity_score": 0.9
        },
        {
            "movie_id": 2,
            "title": "Movie B",
            "genres": "Drama",
            "overview": "Overview B",
            "poster_path": "/pathB.jpg",
            "avg_rating": 3.0,
            "rating_count": 100,
            "similarity_score": 0.8
        },
        {
            "movie_id": 3,
            "title": "Movie C",
            "genres": "Comedy",
            "overview": "Overview C",
            "poster_path": "/pathC.jpg",
            "avg_rating": 5.0,
            "rating_count": 50,
            "similarity_score": 0.95
        }
    ]
