"""Smoke test of the Streamlit app against the fixture database (no network, no keys)."""
import os
import sys
import types

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.serving import chatbot
from tests.conftest import DummyModel, create_movies_table

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    # The app opens ./lancedb_movies; give it the fixture table there.
    monkeypatch.chdir(tmp_path)
    create_movies_table(str(tmp_path / "lancedb_movies"))

    # No model download: the app gets a stand-in embedding model.
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda name: DummyModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    monkeypatch.setattr(chatbot, "client", None)  # no Groq key
    monkeypatch.setattr(chatbot, "_SEARCH_ENGINE", None)
    st.cache_resource.clear()
    yield AppTest.from_file(APP_PATH, default_timeout=120)
    st.cache_resource.clear()


def test_app_opens_with_three_tabs(app):
    app.run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Recommend", "Chat", "Evaluation"]
    # Without a Groq key the Chat tab explains itself instead of crashing.
    assert any("Groq API key" in info.value for info in app.info)


def test_recommend_from_liked_movies_excludes_them(app):
    app.run()
    liked = ["Toy Story (1995)", "Matrix, The (1999)"]

    app.multiselect[0].set_value(liked).run()
    app.button[0].click().run()

    assert not app.exception
    assert "Top 10 for you" in [s.value for s in app.subheader]
    shown = [m.value for m in app.markdown]
    assert "Pulp Fiction (1994)" in shown
    assert not set(liked) & set(shown)


def test_recommend_from_description(app):
    app.run()

    app.text_input[0].set_value("a heist movie").run()
    app.button[1].click().run()

    assert not app.exception
    assert any(c.value == "Matching: a heist movie" for c in app.caption)


def test_app_without_database_shows_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(chatbot, "client", None)
    st.cache_resource.clear()

    at = AppTest.from_file(APP_PATH, default_timeout=120).run()

    assert not at.exception
    assert any("not available" in e.value for e in at.error)
    st.cache_resource.clear()
