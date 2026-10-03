import json
import os

import numpy as np
import pytest
from scipy import sparse

from src.model import (META_FILE, NO_RECOMMENDATION, PureSVD, load_meta, popularity_boost,
                       standardise_rows, top_n)

# Two groups of users with disjoint tastes: users 0-2 like movies 0 and 1,
# users 3-5 like movies 2 and 3. With k = 2 the factors span the two blocks,
# so V @ V.T is 0.5 inside a block and 0 across blocks.
BLOCKS = sparse.csr_matrix(np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
], dtype=np.float32))


def rows(*vectors):
    return sparse.csr_matrix(np.array(vectors, dtype=np.float32))


@pytest.fixture
def model():
    return PureSVD.fit(BLOCKS, k=2, random_state=42, item_ids=np.array([10, 20, 30, 40]), user_ids=np.arange(6))


def test_scores_match_the_hand_computed_projection(model):
    assert model.item_factors.shape == (4, 2)
    assert model.item_factors.dtype == np.float32
    assert np.allclose(model.item_factors @ model.item_factors.T,
                       [[.5, .5, 0, 0], [.5, .5, 0, 0], [0, 0, .5, .5], [0, 0, .5, .5]], atol=1e-5)
    # A user who liked only movie 0, folded in without being part of training.
    assert np.allclose(model.score(rows([1, 0, 0, 0])), [[0.5, 0.5, 0.0, 0.0]], atol=1e-5)


def test_recommend_skips_movies_already_rated(model):
    liked = rows([1, 0, 0, 0])

    # Movie 0 has the top score but was rated; movie 1 is the best unrated one.
    assert model.recommend(liked, seen_rows=rows([1, 0, 0, 0]), n=1).tolist() == [[1]]
    # Movie 1 was rated too (with a low rating, so it is in `seen` but not in `liked`).
    # Movies 2 and 3 both score ~0, so only their membership is checked, not their order.
    assert sorted(model.recommend(liked, seen_rows=rows([1, 1, 0, 0]), n=2)[0].tolist()) == [2, 3]


def test_recommend_marks_empty_slots_when_too_few_movies_are_left(model):
    recs = model.recommend(rows([1, 0, 0, 0]), seen_rows=rows([1, 1, 1, 0]), n=2)

    assert recs.tolist() == [[3, NO_RECOMMENDATION]]


def test_recommend_in_batches_equals_one_pass(model):
    liked = sparse.vstack([BLOCKS, BLOCKS]).tocsr()
    seen = sparse.csr_matrix(liked.shape, dtype=np.int8)

    assert np.array_equal(model.recommend(liked, seen, n=2, batch_size=5), model.recommend(liked, seen, n=2))


def test_top_n_orders_by_score_then_column():
    scores = np.array([[0.1, 0.9, 0.9, 0.3], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    assert top_n(scores, 3).tolist() == [[1, 2, 3], [0, 1, 2]]


def test_same_seed_gives_the_same_factors():
    rng = np.random.default_rng(0)
    X = sparse.csr_matrix((rng.random((60, 30)) < 0.2).astype(np.float32))

    first, second = PureSVD.fit(X, k=5, random_state=42), PureSVD.fit(X, k=5, random_state=42)

    assert np.array_equal(first.item_factors, second.item_factors)
    assert first.factors_sha256() == second.factors_sha256()
    assert first.k == 5


def test_save_and_load_give_the_same_scores(model, tmp_path):
    directory = str(tmp_path / "artifacts" / "2019-06-01")
    model.save(directory, {"train_sha256": "abc", "train_seconds": 1.5})

    loaded = PureSVD.load(directory)
    liked = rows([1, 0, 0, 0], [0, 0, 1, 1])

    assert np.array_equal(loaded.score(liked), model.score(liked))
    assert loaded.item_ids.tolist() == [10, 20, 30, 40]
    assert loaded.user_ids.tolist() == list(range(6))
    meta = load_meta(directory)
    assert meta["k"] == 2 and meta["n_items"] == 4 and meta["n_users"] == 6
    assert meta["train_sha256"] == "abc"
    assert meta["item_factors_sha256"] == model.factors_sha256()
    with open(os.path.join(directory, META_FILE), encoding="utf-8") as f:
        assert json.load(f) == meta


# ---------------------------------------------------------------- blend with recent popularity


def test_standardise_rows():
    scores = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]], dtype=np.float32)

    z = standardise_rows(scores)

    assert np.allclose(z[0].mean(), 0.0, atol=1e-6) and np.allclose(z[0].std(), 1.0, atol=1e-6)
    assert z[0].tolist() == sorted(z[0].tolist())      # order is preserved
    assert z[1].tolist() == [0.0, 0.0, 0.0, 0.0]       # a user with no liked movie: constant scores


def test_popularity_boost_is_the_z_score_of_log_counts():
    counts = np.array([0, 9, 99, 999])

    boost = popularity_boost(counts)

    logged = np.log1p(counts)
    assert np.allclose(boost, (logged - logged.mean()) / logged.std(), atol=1e-6)
    assert popularity_boost(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]


def test_weight_zero_is_plain_pure_svd(model):
    liked = rows([1, 0, 0, 0], [0, 0, 1, 0])
    boost = popularity_boost(np.array([5, 1, 50, 0]))

    assert np.array_equal(model.with_blend(boost, 0.0).score(liked), model.score(liked))


def test_blended_score_matches_the_formula(model):
    liked = rows([1, 0, 0, 0])
    boost = popularity_boost(np.array([5, 1, 50, 0]))

    blended = model.with_blend(boost, 0.5).score(liked)

    assert np.allclose(blended, standardise_rows(model.score(liked)) + 0.5 * boost, atol=1e-6)


def test_a_large_weight_approaches_the_popularity_order(model):
    liked, seen = rows([1, 0, 0, 0]), rows([1, 0, 0, 0])
    boost = popularity_boost(np.array([5, 1, 50, 10]))  # popularity order of the unseen movies: 2, 3, 1

    assert model.recommend(liked, seen, n=3).tolist()[0][0] == 1           # plain: the same-taste movie first
    assert model.with_blend(boost, 100.0).recommend(liked, seen, n=3).tolist() == [[2, 3, 1]]


def test_blend_weight_needs_a_boost(model):
    with pytest.raises(ValueError):
        PureSVD(model.item_factors, model.item_ids, blend_weight=1.0)


def test_blended_model_survives_save_and_load(model, tmp_path):
    boost = popularity_boost(np.array([5, 1, 50, 10]))
    blended = model.with_blend(boost, 0.5)
    directory = str(tmp_path / "blend")

    blended.save(directory, {"train_window": "1y"})
    loaded = PureSVD.load(directory)

    liked = rows([1, 0, 0, 0], [0, 0, 1, 1])
    assert loaded.blend_weight == 0.5
    assert np.array_equal(loaded.score(liked), blended.score(liked))
    assert load_meta(directory)["blend_weight"] == 0.5 and load_meta(directory)["train_window"] == "1y"
