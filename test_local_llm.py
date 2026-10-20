"""Judge routing: homelab Ollama first, Haiku fallback. No network — stubs both sides.
Run: pytest test_local_llm.py"""
import json
import socket

import answerer
import parse


class _Resp:
    def __init__(self, content):
        self.body = json.dumps({"message": {"content": content}}).encode()

    def read(self, *a):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _setup(monkeypatch, local_reply=None, local_exc=None, cloud="NO"):
    """local_reply/local_exc drive urlopen; returns lists of captured calls."""
    sent, cloud_calls = [], []
    monkeypatch.setattr(answerer, "OLLAMA_URL", "http://ollama:11434")

    def fake_urlopen(req, timeout):
        sent.append((req.full_url, json.loads(req.data), timeout))
        if local_exc:
            raise local_exc
        return _Resp(local_reply)

    def fake_call(system, user, max_tokens, temperature, model=None):
        cloud_calls.append(model)
        return cloud

    monkeypatch.setattr(answerer.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(answerer, "_call", fake_call)
    return sent, cloud_calls


def test_local_yes(monkeypatch):
    sent, cloud = _setup(monkeypatch, local_reply="YES")
    assert answerer.judge("mitochondrion", "MITOCHONDRIA / POWERHOUSE")
    assert cloud == []
    url, body, timeout = sent[0]
    assert url == "http://ollama:11434/api/chat"
    assert body["stream"] is False and body["think"] is False and body["keep_alive"] == "24h"
    assert body["options"] == {"temperature": 0, "num_ctx": 16384, "num_predict": 4}


def test_local_no_with_punctuation(monkeypatch):
    _, cloud = _setup(monkeypatch, local_reply=" No.\n", cloud="YES")
    assert not answerer.judge("cell membrane", "CELL WALL (DO NOT ACCEPT: CELL MEMBRANE)")
    assert cloud == []


def test_timeout_falls_back(monkeypatch):
    _, cloud = _setup(monkeypatch, local_exc=socket.timeout("timed out"), cloud="YES")
    assert answerer.judge("a", "b")
    assert cloud == [answerer.FAST]


def test_http_error_falls_back(monkeypatch):
    _, cloud = _setup(monkeypatch, local_exc=OSError("connection refused"), cloud="NO")
    assert not answerer.judge("a", "b")
    assert cloud == [answerer.FAST]


def test_bad_output_falls_back(monkeypatch):
    for reply in ("Yes, because", "", "MAYBE"):
        _, cloud = _setup(monkeypatch, local_reply=reply, cloud="YES")
        assert answerer.judge("a", "b")
        assert cloud == [answerer.FAST]


def test_disabled(monkeypatch):
    sent, cloud = _setup(monkeypatch, local_reply="YES", cloud="NO")
    monkeypatch.setattr(answerer, "OLLAMA_URL", "")
    assert not answerer.judge("a", "b")
    assert sent == [] and cloud == [answerer.FAST]


def test_live_path_skips_local(monkeypatch):
    sent, cloud = _setup(monkeypatch, local_reply="YES", cloud="NO")
    assert not answerer.judge("a", "b", local=False)
    assert sent == [] and cloud == [answerer.FAST]


def test_oversized_prompt_skips_local(monkeypatch):
    sent, cloud = _setup(monkeypatch, local_reply="YES", cloud="NO")
    assert not answerer.judge("a", "x" * 60000)
    assert sent == [] and cloud == [answerer.FAST]


def test_clean_answer_keeps_do_not_accept():
    assert parse.clean_answer("CELL WALL (DO NOT ACCEPT: CELL MEMBRANE) [AG]") == \
        "CELL WALL (DO NOT ACCEPT: CELL MEMBRANE)"
    assert parse.clean_answer("conditional instability (do not accept: absolute instability)") == \
        "conditional instability (DO NOT ACCEPT: absolute instability)"
    assert parse.clean_answer("FUSION (REJECT: FISSION)") == "FUSION (DO NOT ACCEPT: FISSION)"
    # accepts still become '/ alt'; prompts still dropped
    assert parse.clean_answer("INJECTIVE (ACCEPT: ONE-TO-ONE) (PROMPT ON: MAP)") == \
        "INJECTIVE / ONE-TO-ONE"
