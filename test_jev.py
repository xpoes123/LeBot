import jev


def test_no_guess_is_zero_and_no_key_falls_back(monkeypatch):
    assert jev.p_correct("This curve", "PHYSICS", []) == 0.0
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert jev.p_correct("This curve", "PHYSICS", [(1, "parabola")]) is None  # server uses plain LR


def test_state_handles_unknown_word_index():
    st = jev.state("This curve", "PHYSICS", [(None, "parabola"), (4, "cycloid")])
    assert "parabola; after 5 words: cycloid" in st and st.endswith("cycloid")
