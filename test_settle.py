from listen import _settle_word


def test_settle():
    # settles early on "collagen" at word 12 and never wavers
    traj = [(3, "UNKNOWN"), (7, "elastin"), (12, "collagen"), (20, "collagen"), (35, "collagen")]
    assert _settle_word(traj) == ("collagen", 12, 35)

    # only locks in at the very end
    traj = [(5, "A"), (10, "B"), (15, "C")]
    assert _settle_word(traj) == ("C", 15, 15)

    # never gave a real answer
    assert _settle_word([(4, "UNKNOWN"), (9, "")]) == (None, 0, 9)

    # UNKNOWN after a real answer doesn't count as the final; last named wins
    traj = [(6, "mitochondria"), (12, "mitochondria"), (18, "UNKNOWN")]
    assert _settle_word(traj) == ("mitochondria", 6, 18)


if __name__ == "__main__":
    test_settle()
    print("ok")
