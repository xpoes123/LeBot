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
