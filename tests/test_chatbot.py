"""Chatbot tests. Groq is always mocked; nothing here touches the network."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from groq import RateLimitError

from src.serving import chatbot


def rate_limit_error():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


def reply(content=None, tool_calls=None):
    """A Groq chat completion with one choice."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def tool_call(call_id, name, **arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class FakeClient:
    """Replays a list of outcomes; an exception in the list is raised."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Copy: the chatbot keeps appending to the same list after the call.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def bot(engine, monkeypatch):
    """A chatbot wired to the fixture database, with a single fake key."""
    monkeypatch.setattr(chatbot, "_SEARCH_ENGINE", engine)
    monkeypatch.setattr(chatbot, "API_KEYS", ["key-1"])
    monkeypatch.setattr(chatbot, "current_key_idx", 0)
    monkeypatch.setattr(chatbot, "client", FakeClient([]))
    return chatbot.MovieChatbot()


def test_registry_holds_exactly_the_three_declared_tools():
    declared = {tool["function"]["name"] for tool in chatbot.tools}
    assert declared == set(chatbot.TOOL_REGISTRY) == {
        "search_movies_by_description", "get_recommendations", "get_trending_by_rating"}


def test_plain_answer_without_tools(bot, monkeypatch):
    client = FakeClient([reply(content="Hello!")])
    monkeypatch.setattr(chatbot, "client", client)

    assert bot.chat("hi") == "Hello!"
    assert len(client.calls) == 1
    # The retrieval step put database context into the system prompt.
    assert "THÔNG TIN TỪ DATABASE" in client.calls[0]["messages"][0]["content"]


def test_assistant_message_is_appended_once_for_several_tool_calls(bot, monkeypatch):
    """Regression: the assistant message was appended once per tool call."""
    calls = [
        tool_call("call_1", "get_recommendations", title="Toy Story (1995)"),
        tool_call("call_2", "get_trending_by_rating", min_rating=4.0, min_votes=10000),
    ]
    first = reply(tool_calls=calls)
    client = FakeClient([first, reply(content="Here are some movies.")])
    monkeypatch.setattr(chatbot, "client", client)

    assert bot.chat("recommend something") == "Here are some movies."

    followup = client.calls[1]["messages"]
    assistant_messages = [m for m in followup if m is first.choices[0].message]
    tool_messages = [m for m in followup if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(assistant_messages) == 1
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2"]
    # Assistant message first, then its tool results, in order.
    assert followup[-3] is first.choices[0].message

    recommended = json.loads(tool_messages[0]["content"])
    assert recommended and all(movie["movie_id"] != 1 for movie in recommended)


def test_unknown_tool_is_not_executed(bot, monkeypatch):
    """Regression: any module-level function could be called through globals()."""
    rotate = MagicMock(return_value=True)
    monkeypatch.setattr(chatbot, "rotate_key", rotate)
    client = FakeClient([
        reply(tool_calls=[tool_call("call_1", "rotate_key")]),
        reply(content="Done."),
    ])
    monkeypatch.setattr(chatbot, "client", client)

    assert bot.chat("call something you should not") == "Done."

    rotate.assert_not_called()
    tool_message = client.calls[1]["messages"][-1]
    assert json.loads(tool_message["content"]) == {"error": "Unknown tool: rotate_key"}


def test_bad_tool_arguments_return_an_error_to_the_model(bot):
    assert "error" in json.loads(chatbot.run_tool("get_recommendations", "not json"))
    assert "error" in json.loads(chatbot.run_tool("get_recommendations", json.dumps({"wrong": 1})))


def test_rate_limit_rotates_to_the_next_key_before_falling_back(bot, monkeypatch):
    """Regression: rotate_key() was never called on a RateLimitError."""
    clients = {
        "key-1": FakeClient([rate_limit_error()]),
        "key-2": FakeClient([reply(content="Answer from the second key.")]),
    }
    monkeypatch.setattr(chatbot, "API_KEYS", ["key-1", "key-2"])
    monkeypatch.setattr(chatbot, "Groq", lambda api_key: clients[api_key])
    monkeypatch.setattr(chatbot, "client", clients["key-1"])

    assert bot.chat("a heist movie") == "Answer from the second key."
    assert chatbot.current_key_idx == 1
    assert len(clients["key-1"].calls) == 1
    assert len(clients["key-2"].calls) == 1


def test_fallback_when_every_key_is_rate_limited(bot, monkeypatch):
    clients = {
        "key-1": FakeClient([rate_limit_error()]),
        "key-2": FakeClient([rate_limit_error()]),
    }
    monkeypatch.setattr(chatbot, "API_KEYS", ["key-1", "key-2"])
    monkeypatch.setattr(chatbot, "Groq", lambda api_key: clients[api_key])
    monkeypatch.setattr(chatbot, "client", clients["key-1"])

    answer = bot.chat("a heist movie")

    # Each key was tried exactly once, then the vector-search fallback answered.
    assert len(clients["key-1"].calls) == 1
    assert len(clients["key-2"].calls) == 1
    assert answer.startswith("⚠️")
    assert "1. **" in answer


def test_fallback_with_a_single_key(bot, monkeypatch):
    client = FakeClient([rate_limit_error()])
    monkeypatch.setattr(chatbot, "client", client)

    answer = bot.chat("a heist movie")

    assert len(client.calls) == 1
    assert answer.startswith("⚠️")


def test_chatbot_requires_a_key(monkeypatch):
    monkeypatch.setattr(chatbot, "client", None)
    with pytest.raises(ValueError):
        chatbot.MovieChatbot()
