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
