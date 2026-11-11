"""The early-buzz gate: only commit after the SAME answer is cross-model-confirmed
BUZZ_RUN ticks in a row. Run: ./botvenv/bin/python -m pytest test_buzzgate.py -q"""
from live import _confirm_run, _norm_answer, BUZZ_RUN


def _feed(ticks):
    """Replay (norm, agrees) ticks; return the word-run at each step."""
    norm, run, runs = None, 0, []
    for n, agr in ticks:
        norm, run = _confirm_run(norm, run, n, agr)
        runs.append(run)
    return runs


def test_confirms_only_on_stable_agreement():
    # same answer, both agreeing, two ticks -> run reaches BUZZ_RUN
    runs = _feed([("caspases", True), ("caspases", True)])
    assert runs == [1, 2] and runs[-1] >= BUZZ_RUN


def test_disagreement_never_accrues():
    assert _feed([("caspases", False), ("caspases", False)]) == [0, 0]


def test_changed_answer_resets_run():
    # a flip-flop between two agreed answers must not reach the buzz threshold
    runs = _feed([("apoptosis", True), ("caspases", True), ("apoptosis", True)])
    assert max(runs) < BUZZ_RUN


def test_unknown_resets():
    runs = _feed([("caspases", True), ("", False), ("caspases", True)])
    assert runs == [1, 0, 1]


def test_norm_absorbs_wording_and_punctuation():
    assert _norm_answer("The Mitochondrion.") == _norm_answer("the  mitochondrion")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
