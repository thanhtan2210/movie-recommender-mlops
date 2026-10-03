"""The API against a small blend model in a temporary MODEL_DIR (no network)."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api, tracking
from src.export_champion import write_serving_dir
from src.model import BlendRecommender
from tests.conftest import CUTOFF
from tests.test_recommender import ROWS, ratings_frame

# The fixture data (see tests/test_recommender.py): users 1-3 like movies 10 and 20,
# users 4-6 like 30 and 40, movie 50 is the most liked recently; user 6 has not rated 50.
MOVIES = pd.DataFrame({
    "movieId": [10, 20, 30, 40, 50],
    "title": ["Ten (1990)", "Twenty (1991)", "Thirty (1992)", "Forty (1993)", "Fifty, No Year"],
})
META = {"model_name": "movie-recommender", "model_version": "7", "model_type": "blend", "cutoff": "2019-06-01",
        "trained_on_rows": len(ROWS)}


@pytest.fixture(scope="module")
def blend():
    return BlendRecommender.fit(ratings_frame(rows=ROWS), "2019-06-01", CUTOFF, 4.0, k=2, train_window="1y",
                                blend_weight=0.5)


@pytest.fixture
def client(blend, tmp_path, monkeypatch):
    serving_dir = tmp_path / "serving_model"
    write_serving_dir(str(serving_dir), tracking.save_pyfunc(blend, str(tmp_path / "pyfunc")), MOVIES, META)
    monkeypatch.setenv("MODEL_DIR", str(serving_dir))
    with TestClient(api.app) as test_client:
        yield test_client


def ids(response):
    return [item["movie_id"] for item in response.json()["items"]]


def test_health_reports_the_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_version"] == "7"
    assert body["cutoff"] == "2019-06-01" and body["trained_on_rows"] == len(ROWS)
    assert body["loaded_at"].endswith("Z")
    assert set(body) == {"status", "model_version", "cutoff", "trained_on_rows", "loaded_at"}


def test_known_user_gets_personalized_recommendations_without_rated_movies(client, blend):
    response = client.get("/recommend/6", params={"n": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 6 and body["strategy"] == "personalized"
    assert ids(response) == blend.recommend([6], n=3)[0].tolist()
    assert not {30, 40} & set(ids(response))            # user 6 rated 30 and 40
    assert body["items"][0] == {"movie_id": 50, "title": "Fifty, No Year", "year": None}
    assert {"movie_id": 10, "title": "Ten (1990)", "year": 1990} in body["items"]


def test_unknown_user_gets_the_popularity_fallback(client):
    response = client.get("/recommend/999", params={"n": 3})

    assert response.status_code == 200
    assert response.json()["strategy"] == "popularity_fallback"
    assert ids(response) == [50, 10, 20]


def test_n_defaults_to_ten_and_fewer_items_come_back_when_the_catalogue_runs_out(client):
    response = client.get("/recommend/1")

    assert response.status_code == 200
    assert ids(response) == [30, 40] or sorted(ids(response)) == [30, 40]   # user 1 rated 10, 20 and 50


def test_post_folds_in_the_liked_movies_like_the_model_does(client, blend):
    """User 6 liked (and rated) exactly movies 30 and 40: an anonymous user with the same likes gets the same list."""
    response = client.post("/recommend", json={"liked_movie_ids": [30, 40], "n": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "personalized" and body["ignored_ids"] == []
    assert ids(response) == blend.recommend([6], n=3)[0].tolist()
    assert ids(response) == blend.predict(None, [6], params={"n": 3})[0].tolist()
    assert not {30, 40} & set(ids(response))


def test_post_ignores_unknown_ids_and_reports_them(client):
    response = client.post("/recommend", json={"liked_movie_ids": [30, 123456, 40, 30, 777], "n": 3})

    body = response.json()
    assert response.status_code == 200
    assert body["ignored_ids"] == [123456, 777] and body["strategy"] == "personalized"
    assert ids(response) == ids(client.post("/recommend", json={"liked_movie_ids": [30, 40], "n": 3}))


@pytest.mark.parametrize("liked", [[], [123456, 777]])
def test_post_without_a_usable_id_falls_back_to_popularity(client, liked):
    response = client.post("/recommend", json={"liked_movie_ids": liked, "n": 3})

    body = response.json()
    assert response.status_code == 200
    assert body["strategy"] == "popularity_fallback" and body["ignored_ids"] == liked
    assert ids(response) == [50, 10, 20]


@pytest.mark.parametrize("n", [0, 51, -1, "many"])
def test_n_out_of_range_is_rejected(client, n):
    assert client.get("/recommend/6", params={"n": n}).status_code == 422
    assert client.post("/recommend", json={"liked_movie_ids": [30], "n": n}).status_code == 422


def test_other_invalid_requests_are_rejected(client):
    assert client.get("/recommend/not-a-number").status_code == 422
    assert client.post("/recommend", json={"liked_movie_ids": list(range(101))}).status_code == 422
    assert client.post("/recommend", json={"n": 5}).status_code == 422
    assert client.post("/recommend", json={"liked_movie_ids": ["abc"]}).status_code == 422
    # The limits themselves are accepted.
    assert client.get("/recommend/6", params={"n": 50}).status_code == 200
    assert client.post("/recommend", json={"liked_movie_ids": list(range(100)), "n": 1}).status_code == 200


def test_every_request_writes_one_json_log_line(client, capsys):
    capsys.readouterr()
    client.get("/recommend/6", params={"n": 3})
    client.post("/recommend", json={"liked_movie_ids": [123456], "n": 4})
    client.get("/health")
    client.get("/recommend/6", params={"n": 0})

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]

    assert [(line["method"], line["endpoint"], line["status"], line["strategy"], line["n"]) for line in lines] == [
        ("GET", "/recommend/{user_id}", 200, "personalized", 3),
        ("POST", "/recommend", 200, "popularity_fallback", 4),
        ("GET", "/health", 200, None, None),
        ("GET", "/recommend/{user_id}", 422, None, None),
    ]
    assert all(line["latency_ms"] >= 0 and "time" in line for line in lines)


def test_the_api_refuses_to_start_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="python -m src.export_champion"):
        with TestClient(api.app):
            pass


def test_recommend_for_liked_on_the_recommenders(blend):
    from src.baseline import PopularityRecommender

    movie_ids, ignored, personalised = blend.recommend_for_liked([30, 40, 999], n=2)
    assert personalised and ignored == [999] and len(movie_ids) == 2 and not {30, 40} & set(movie_ids.tolist())

    popularity = PopularityRecommender.fit(ratings_frame(rows=ROWS), CUTOFF, 4.0)
    movie_ids, ignored, personalised = popularity.recommend_for_liked([50, 10], n=2)
    assert not personalised and ignored == [] and movie_ids.tolist() == [20, 30]      # popularity minus the liked ones
    assert isinstance(movie_ids, np.ndarray)
