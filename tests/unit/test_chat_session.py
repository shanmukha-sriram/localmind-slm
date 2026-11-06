from types import SimpleNamespace

import pytest

import app.inference.generator as gen
from app.core.exceptions import OllamaUnavailableError


def _chunk(text="", done=False, **stats):
    return SimpleNamespace(message=SimpleNamespace(content=text), done=done, **stats)


class FakeClient:
    def __init__(self, chunks=None, error=None):
        self.chunks, self.error, self.received = chunks or [], error, []

    def chat(self, **kwargs):
        self.received.append(kwargs["messages"])
        if self.error:
            raise self.error
        return iter(self.chunks)


def _reply(text):
    return [_chunk("Hel"), _chunk(text[3:]), _chunk(done=True, eval_count=10,
            eval_duration=2_000_000_000)]


def test_streaming_collects_text_stats_and_history(monkeypatch):
    client = FakeClient(_reply("Hello!"))
    monkeypatch.setattr(gen, "get_client", lambda: client)
    session = gen.ChatSession(model="m")

    assert "".join(session.send_stream("hi")) == "Hello!"
    assert session.history == [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "Hello!"}]
    assert session.last_stats.tokens_per_second == pytest.approx(5.0)
    assert session.last_stats.ttft_s is not None


def test_history_is_resent_on_second_turn(monkeypatch):
    client = FakeClient(_reply("Hello!"))
    monkeypatch.setattr(gen, "get_client", lambda: client)
    session = gen.ChatSession(model="m")
    session.send("hi")
    client.chunks = _reply("Again!")
    session.send("again")
    assert len(client.received[1]) == 3  # user, assistant, user


def test_server_down_raises_clear_error_and_keeps_history_clean(monkeypatch):
    monkeypatch.setattr(gen, "get_client", lambda: FakeClient(error=ConnectionError("down")))
    session = gen.ChatSession(model="m")
    with pytest.raises(OllamaUnavailableError):
        session.send("hi")
    assert session.history == []


def test_clear(monkeypatch):
    monkeypatch.setattr(gen, "get_client", lambda: FakeClient(_reply("Hello!")))
    session = gen.ChatSession(model="m")
    session.send("hi")
    session.clear()
    assert session.history == [] and session.last_stats is None