import answerer


def test_buzz_after_y_detection():
    assert answerer.mc_past_y("Earth and Space multiple choice. Which... w wind duration, x wind speed, y fetch")
    assert answerer.mc_past_y("Which is not a factor? W, wind duration. X, wind speed, why, fetch")
    assert not answerer.mc_past_y("multiple choice. Which... w wind duration, x wind speed, y")   # Y's text not started
    assert not answerer.mc_past_y("Math short answer. If w is 3 and x is 2 find y plus w")        # algebra, not choices


def test_asked_yet_and_spoken_numbers():
    assert not answerer.asked_yet("Edwin is connected to two springs with zero rest length")
    assert answerer.asked_yet("This curve of fastest descent")
    assert answerer.asked_yet("then what is Edwin's minimum potential energy")
    assert answerer.is_computational("constants of two and four newtons per meter, fixed nine meters apart", "PHYSICS")
    assert not answerer.is_computational("the second law of thermodynamics", "PHYSICS")


def test_vote_in_time_respects_the_answer_window():
    import time
    mk = lambda v, d: (lambda: (time.sleep(d), v)[1])
    t = time.time()
    assert answerer._vote_in_time([mk("Y", 0.1), mk("Y", 0.15), mk("Z", 2)], 0.8) == "Y" and time.time() - t < 0.5
    t = time.time()
    assert answerer._vote_in_time([mk("Y", 0.1), mk("Z", 0.15), mk("Y", 2)], 0.8) in ("Y", "Z") and time.time() - t < 1.2
    assert answerer._vote_in_time([mk("UNKNOWN", 0.05)] * 3, 0.8) is None


def test_parrot_guard_catches_phrase_completion():
    stem = "He notices that when plotted on the eightfold"
    assert answerer._parrots_stem("Eightfold Way", stem)
    assert answerer._parrots_stem("cytotoxic T cells", "Granzyme B, secreted by cytotoxic T cells, activates")
    assert not answerer._parrots_stem("Kirkwood gaps", "The gravitational influence of Jupiter results in")
    assert not answerer._parrots_stem("retinal", "Opsin is activated when what molecule")   # single words may recur
    assert not answerer.asked_yet(stem)                     # "when plotted" is a clause, not the ask
    assert answerer.asked_yet("how many seconds does it take")
