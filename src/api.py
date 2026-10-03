"""Recommendation API. Three endpoints:

  GET  /health                     model version and when it was loaded
  GET  /recommend/{user_id}?n=10   top-n for a user id
  POST /recommend                  top-n for an anonymous user from the movies they liked

The model is read once at start-up from MODEL_DIR (default serving_model/, see
src.export_champion). Each request writes one JSON line to stdout: time,
endpoint, strategy, n, latency_ms.

Run locally:  uvicorn src.api:app --port 8000
"""
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, Field

from src.baseline import PopularityRecommender
from src.export_champion import META_FILE, MOVIES_FILE, SERVING_DIR, state_path
from src.model import BlendRecommender
from src.recommender import NO_RECOMMENDATION, Recommender

PERSONALIZED = "personalized"
POPULARITY_FALLBACK = "popularity_fallback"
MAX_N = 50
MAX_LIKED = 100
RECOMMENDER_TYPES = {BlendRecommender.model_type: BlendRecommender,
                     PopularityRecommender.model_type: PopularityRecommender}


class Item(BaseModel):
    movie_id: int
    title: Optional[str]
    year: Optional[int]


class Health(BaseModel):
    status: str
    model_version: str
    cutoff: str
    trained_on_rows: int
    loaded_at: str


class UserRecommendations(BaseModel):
    user_id: int
    strategy: str
    items: List[Item]


class LikedMovies(BaseModel):
    liked_movie_ids: List[int] = Field(max_length=MAX_LIKED)
    n: int = Field(default=10, ge=1, le=MAX_N)


class AnonymousRecommendations(BaseModel):
    strategy: str
    items: List[Item]
    ignored_ids: List[int]


class ServingModel:
    """The recommender with the catalogue metadata needed to answer requests."""

    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, META_FILE), encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)
        with open(os.path.join(model_dir, MOVIES_FILE), encoding="utf-8") as f:
            self.movies = {movie["movie_id"]: movie for movie in json.load(f)}
        self.recommender: Recommender = RECOMMENDER_TYPES[self.meta["model_type"]].load_state(state_path(model_dir))
        self.loaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def knows(self, user_id: int) -> bool:
        users = self.recommender.user_ids
        position = min(int(np.searchsorted(users, user_id)), len(users) - 1)
        return bool(users[position] == user_id)

    def items(self, movie_ids) -> List[Item]:
        picked = []
        for movie_id in movie_ids:
            if movie_id == NO_RECOMMENDATION:
                continue
            movie = self.movies.get(int(movie_id), {})
            picked.append(Item(movie_id=int(movie_id), title=movie.get("title"), year=movie.get("year")))
        return picked


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.environ.get("MODEL_DIR", SERVING_DIR)
    if not os.path.exists(os.path.join(model_dir, META_FILE)):
        raise RuntimeError(f"No model in {model_dir!r}. Run: python -m src.export_champion")
    app.state.serving = ServingModel(model_dir)
    yield


app = FastAPI(title="Movie recommender", lifespan=lifespan)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """One JSON line per request on stdout: the basis for monitoring."""
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    print(json.dumps({
        "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "method": request.method,
        "endpoint": getattr(route, "path", request.url.path),
        "status": response.status_code,
        "strategy": getattr(request.state, "strategy", None),
        "n": getattr(request.state, "n", None),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }), flush=True)
    return response


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    serving: ServingModel = request.app.state.serving
    return Health(status="ok", model_version=str(serving.meta["model_version"]), cutoff=serving.meta["cutoff"],
                  trained_on_rows=int(serving.meta["trained_on_rows"]), loaded_at=serving.loaded_at)


@app.get("/recommend/{user_id}", response_model=UserRecommendations)
def recommend_for_user(request: Request, user_id: int, n: int = Query(default=10, ge=1, le=MAX_N)) -> UserRecommendations:
    """Top-n for a user id. A user unknown to the training data gets the popularity list."""
    serving: ServingModel = request.app.state.serving
    strategy = PERSONALIZED if serving.knows(user_id) else POPULARITY_FALLBACK
    request.state.strategy, request.state.n = strategy, n
    movie_ids = serving.recommender.recommend([user_id], n)[0]
    return UserRecommendations(user_id=user_id, strategy=strategy, items=serving.items(movie_ids))


@app.post("/recommend", response_model=AnonymousRecommendations)
def recommend_for_liked_movies(request: Request, body: LikedMovies) -> AnonymousRecommendations:
    """Top-n for an anonymous user, folded into the model from the movies they liked.

    Ids outside the catalogue are ignored and listed in `ignored_ids`; with no
    usable id the popularity list is returned.
    """
    serving: ServingModel = request.app.state.serving
    movie_ids, ignored, personalised = serving.recommender.recommend_for_liked(body.liked_movie_ids, body.n)
    strategy = PERSONALIZED if personalised else POPULARITY_FALLBACK
    request.state.strategy, request.state.n = strategy, body.n
    return AnonymousRecommendations(strategy=strategy, items=serving.items(movie_ids), ignored_ids=ignored)
