"""Stale-guess / fresh-judge: the fast path buzzes on the latest completed guess at Jev speed when Jev rates it
against the words heard since, without waiting for Sonnet's re-guess."""
import server


def _req(prefix, history, mode="balanced"):
    return server.AnalyzeReq(prefix=prefix, category="BIOLOGY", fast=True, mode=mode,
                             history=[server._Step(guess=g, mode="recall", words=w) for w, g in history])


def test_stale_guess_buzzes_without_a_fresh_guess(monkeypatch):
    calls = []
    monkeypatch.setattr(server.jev, "qtype_ready", lambda *a, **k: (0.0, 0.0, 0.0))
    monkeypatch.setattr(server.jev, "p_correct", lambda text, cat, traj: calls.append(traj) or 0.97)
    monkeypatch.setattr(server.answerer, "anticipate_fast_confirm",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fresh guess must not be awaited")))
    d = server.analyze(_req("Opsin is activated when what molecule, derived from vitamin",
                            [(3, "rhodopsin"), (6, "retinal"), (9, "retinal")]))
    assert d["gate_buzz"] and d["guess"] == "retinal" and d["stale"]
    assert calls[-1][-1] == (8, "retinal")   # judged the latest guess, trajectory in word order


def test_stale_guess_below_threshold_waits_for_the_fresh_guess(monkeypatch):
    seen = []
    monkeypatch.setattr(server.jev, "qtype_ready", lambda *a, **k: (0.0, 0.0, 0.0))
    monkeypatch.setattr(server.jev, "p_correct", lambda text, cat, traj: seen.append(traj[-1][1]) or 0.5)
    monkeypatch.setattr(server.answerer, "anticipate_fast_confirm", lambda *a, **k: ("retinal", "UNKNOWN", "recall"))
    d = server.analyze(_req("Opsin is activated when what molecule", [(3, "rhodopsin"), (6, "retinal")]))
    assert not d["gate_buzz"] and d["guess"] == "retinal" and "stale" not in d
    assert seen == ["retinal"]   # the fresh guess equals the stale one: Jev's answer is reused, not re-asked


def test_no_history_and_pre_ask_guard(monkeypatch):
    monkeypatch.setattr(server.jev, "qtype_ready", lambda *a, **k: (0.0, 0.0, 0.0))
    monkeypatch.setattr(server.jev, "p_correct", lambda text, cat, traj: 0.9)
    monkeypatch.setattr(server.answerer, "anticipate_fast_confirm", lambda *a, **k: ("corpus callosum", "UNKNOWN", "recall"))
    # nothing asked yet ("The left and right hemispheres of the"): 0.9 is under the pre-ask bar, stale or fresh
    d = server.analyze(_req("The left and right hemispheres of the", [(3, "corpus callosum"), (6, "corpus callosum")]))
    assert not d["gate_buzz"] and d["gate_T"] == server.PRE_ASK_T
    assert server._latest_guess(_req("a b c", [])) is None
    assert server._latest_guess(_req(" ".join("w" * 20), [(3, "old")])) is None   # too far behind to trust
