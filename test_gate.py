import gate


def test_feats_track_runs_and_reset_on_unknown():
    rec = {"seq": [["UNKNOWN", False], ["cycloid", True], ["cycloid", True], ["UNKNOWN", False], ["cycloid", True]],
           "idx": [2, 5, 8, 11, 14], "category": "PHYSICS", "jev": [None, 0.4, 0.9, None, 0.8]}
    has, run, arun, churn, agrees, pj = gate.feats(rec, 2)[:6]
    assert (has, run, arun, churn, agrees, pj) == (1.0, 2.0, 2.0, 1.0, 1.0, 0.9)
    has, run, arun = gate.feats(rec, 3)[:3]
    assert (has, run, arun) == (0.0, 0.0, 0.0)          # an UNKNOWN tick resets both runs
    assert gate.feats(rec, 4)[1:3] == [1.0, 1.0]         # ...so the next guess starts over


def test_p_correct_uses_shipped_weights(monkeypatch):
    rec = {"seq": [["x", True]], "idx": [4], "category": "MATH", "jev": [0.5]}
    monkeypatch.setattr(gate, "_w", {"coef": [0.0] * len(gate.feats(rec, 0)), "intercept": 0.0, "T": 0.7})
    assert gate.p_correct(rec, 0) == (0.5, 0.7)
