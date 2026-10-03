import numpy as np
import pandas as pd
import pytest

from tests.conftest import MOVIES, VECTOR_DIM


def test_format_result(engine):
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

    res = engine._format_result(row)
    assert res["movie_id"] == 42
    assert res["title"] == "Test Movie"
    assert res["similarity_score"] == pytest.approx(0.7)


def test_search_by_description_empty(engine):
    assert engine.search_by_description("") == []
    assert engine.search_by_description("   ") == []


def test_search_by_description_without_reranker(engine):
    results = engine.search_by_description("find some cool movie", top_k=2, use_reranker=False)

    assert len(results) == 2
    assert "similarity_score" in results[0]
    assert "final_score" not in results[0]
    # Without the reranker the order is the order of similarity.
    assert results[0]["similarity_score"] >= results[1]["similarity_score"]


def test_search_by_description_with_reranker(engine):
    results = engine.search_by_description("find some cool movie", top_k=2, use_reranker=True)

    assert len(results) == 2
    assert results[0]["final_score"] >= results[1]["final_score"]


def test_search_similar_movies_excludes_the_source(engine):
    results = engine.search_similar_movies(movie_id=1, top_k=3, use_reranker=False)

    assert len(results) == 3
    assert all(r["movie_id"] != 1 for r in results)


def test_search_similar_movies_unknown_id(engine):
    with pytest.raises(ValueError, match="Không tìm thấy phim"):
        engine.search_similar_movies(movie_id=999999)


def test_search_similar_movies_by_title(engine):
    results = engine.search_similar_movies_by_title(title="matrix", top_k=2, use_reranker=False)

    assert len(results) == 2
    assert all(r["movie_id"] != 2571 for r in results)


def test_search_by_title_handles_parentheses(engine):
    """Titles carry a year in parentheses; they must be matched literally, not as a regex."""
    by_full_title = engine.search_similar_movies_by_title("Toy Story (1995)", top_k=2)
    assert all(r["movie_id"] != 1 for r in by_full_title)

    by_leading_parenthesis = engine.search_similar_movies_by_title("(500) Days", top_k=2)
    assert all(r["movie_id"] != 69757 for r in by_leading_parenthesis)


def test_search_similar_movies_by_title_not_found(engine):
    with pytest.raises(ValueError, match="Không tìm thấy phim"):
        engine.search_similar_movies_by_title("Avengers")


def test_get_trending_by_rating(engine):
    results = engine.get_trending_by_rating(min_rating=4.2, min_votes=50000, top_k=5)

    assert [r["title"] for r in results] == [
        "Shawshank Redemption, The (1994)", "Usual Suspects, The (1995)",
        "Pulp Fiction (1994)", "Matrix, The (1999)",
    ]


def test_get_user_vector(engine):
    user_vec = engine.get_user_vector({1: 4.0, 2: 5.0})

    assert user_vec.shape == (VECTOR_DIM,)
    assert np.isclose(np.linalg.norm(user_vec), 1.0)

    expected = 4.0 * engine._vector_of(1) + 5.0 * engine._vector_of(2)
    assert np.allclose(user_vec, expected / np.linalg.norm(expected), atol=1e-6)


def test_get_user_vector_ignores_unknown_movies(engine):
    assert np.allclose(engine.get_user_vector({1: 5.0, 999999: 5.0}), engine.get_user_vector({1: 5.0}))
    with pytest.raises(ValueError):
        engine.get_user_vector({999999: 5.0})


def test_personalized_recommend_excludes_selected_movies(engine):
    """Regression: movies the user picked used to come back as recommendations."""
    liked = [1, 2571]
    user_vec = engine.get_user_vector({movie_id: 5.0 for movie_id in liked})

    without_exclusion = engine.personalized_recommend(user_vec, top_k=5)
    assert set(liked) & {r["movie_id"] for r in without_exclusion}

    recs = engine.personalized_recommend(user_vec, top_k=5, exclude_ids=liked)
    assert len(recs) == 5
    assert not set(liked) & {r["movie_id"] for r in recs}
    assert "final_score" in recs[0]


def test_personalized_recommend_candidate_pool_and_weights(engine):
    user_vec = engine.get_user_vector({1: 5.0})

    # With similarity only, the order is the retrieval order.
    by_similarity = engine.personalized_recommend(
        user_vec, top_k=3, exclude_ids=[1], n_candidates=11, sim_weight=1.0, pop_weight=0.0, qual_weight=0.0)
    retrieved = engine.retrieve_candidates(user_vec, 3, exclude_ids=[1])
    assert [r["movie_id"] for r in by_similarity] == [r["movie_id"] for r in retrieved]

    # With popularity only, the most rated movie of the pool comes first.
    by_popularity = engine.personalized_recommend(
        user_vec, top_k=3, exclude_ids=[1], n_candidates=11, sim_weight=0.0, pop_weight=1.0, qual_weight=0.0)
    assert by_popularity[0]["title"] == "Shawshank Redemption, The (1994)"


def test_retrieve_candidates_returns_requested_number_after_exclusion(engine):
    user_vec = engine.get_user_vector({1: 5.0})
    exclude = [1, 2, 6, 32, 50]

    candidates = engine.retrieve_candidates(user_vec, 4, exclude_ids=exclude)

    assert len(candidates) == 4
    assert not set(exclude) & {c["movie_id"] for c in candidates}


def test_table_is_read_into_memory_only_once(engine, monkeypatch):
    """Regression: every lookup used to call table.to_pandas() again."""
    assert len(engine.catalog) == len(MOVIES)

    calls = []
    table_type = type(engine.table)
    original = table_type.to_pandas
    monkeypatch.setattr(table_type, "to_pandas", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))

    user_vec = engine.get_user_vector({1: 5.0, 2: 4.0})
    engine.search_similar_movies(1, top_k=2)
    engine.search_similar_movies_by_title("heat", top_k=2)
    engine.get_trending_by_rating(4.0, 10000)
    engine.personalized_recommend(user_vec, top_k=3, exclude_ids=[1, 2])

    assert calls == []
