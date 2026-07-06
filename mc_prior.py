"""Multiple-choice letter-prior de-biasing.

The corpus is not uniform over W/X/Y/Z, and stem cues shift it further. This blends the
model's letter votes with an empirical, stem-adjusted prior so that when the votes are
SPLIT the prior breaks the tie the right way (and when the votes are confident, they win).

Empirical priors from packets/questions_clean.json (n=8,241 four-option MC tossups+bonus):
  base            W .215  X .272  Y .274  Z .239   (X/Y inflated, W depressed)
  CAPS negation   -> answer skews Y/Z ~59%   (NOT / EXCEPT / LEAST)
  "closest to"    -> middle values X/Y ~67%
  numeric sorted  -> middle bias, the largest value is the least-likely answer
"""
import re
from collections import Counter

LETTERS = ["W", "X", "Y", "Z"]
BASE = {"W": 0.215, "X": 0.272, "Y": 0.274, "Z": 0.239}
NEGATION = {"W": 0.205, "X": 0.205, "Y": 0.29, "Z": 0.30}   # ~59% Y/Z
CLOSEST = {"W": 0.15, "X": 0.335, "Y": 0.335, "Z": 0.18}    # ~67% X/Y (middle)
NUMERIC = {"W": 0.17, "X": 0.315, "Y": 0.315, "Z": 0.20}    # middle bias, largest(Z) down


def _normalize(p):
    s = sum(p.values())
    return {k: v / s for k, v in p.items()}


def _is_negated(stem):
    return bool(re.search(r"\b(NOT|EXCEPT|LEAST|INCORRECT|FALSE)\b", stem or "")
                or re.search(r"which of the following.{0,60}\bnot\b", stem or "", re.I))


def _numeric_sorted(options):
    """True if all options parse as numbers and are monotonic (the common ~88% case)."""
    if not options or len(options) < 3:
        return False
    vals = []
    for o in options:
        m = re.search(r"-?\d+\.?\d*", str(o).replace(",", ""))
        if not m:
            return False
        vals.append(float(m.group()))
    return vals == sorted(vals) or vals == sorted(vals, reverse=True)


def stem_prior(stem, options=None):
    """Adjusted prior over W/X/Y/Z given stem cues (checked most-specific first)."""
    if _is_negated(stem):
        return _normalize(NEGATION)
    if re.search(r"closest to", stem or "", re.I):
        return _normalize(CLOSEST)
    if _numeric_sorted(options):
        return _normalize(NUMERIC)
    return _normalize(BASE)


def debias(letters, stem, options=None):
    """Blend model letter votes with the stem-adjusted prior -> (letter, confidence).

    Unanimous votes → that letter wins with high confidence. Split votes → the prior
    breaks the tie. No votes (all NONE) → fall back to the prior's argmax.
    """
    votes = Counter(l for l in letters if l in BASE)
    prior = stem_prior(stem, options)
    if not votes:
        letter = max(prior, key=prior.get)
        return letter, prior[letter]
    total = sum(votes.values())
    # posterior ∝ (vote fraction + small floor) × prior
    post = {L: (votes.get(L, 0) / total + 1e-3) * prior[L] for L in BASE}
    z = sum(post.values())
    post = {L: v / z for L, v in post.items()}
    letter = max(post, key=post.get)
    return letter, round(post[letter], 3)


def demo():
    # prior argmax is X or Y (inflated), never W
    assert max(BASE, key=BASE.get) in ("X", "Y")
    # negation shifts mass to Y/Z
    p = stem_prior("Which of the following is NOT a noble gas?")
    assert p["Y"] + p["Z"] > p["W"] + p["X"]
    # closest-to shifts to the middle X/Y
    p = stem_prior("Which is closest to the speed of sound?", ["100", "200", "340", "500"])
    assert p["X"] + p["Y"] > 0.6
    # numeric sorted -> middle bias, largest (Z) least
    p = stem_prior("What is the value?", ["1", "2", "3", "4"])
    assert p["Z"] < p["X"] and p["Z"] < p["Y"]
    # unanimous votes win regardless of prior
    letter, conf = debias(["W", "W", "W", "W", "W"], "plain stem")
    assert letter == "W" and conf > 0.8, (letter, conf)
    # split W/Z vote, no cue -> prior breaks toward Z (Z prior .239 > W .215)? both low;
    # split X/W -> X wins (higher prior)
    letter, _ = debias(["X", "W"], "plain stem")
    assert letter == "X", letter
    # negation + split Y/W -> Y wins (negation lifts Y)
    letter, _ = debias(["Y", "W"], "Which is NOT true?")
    assert letter == "Y", letter
    # no votes -> prior argmax
    letter, _ = debias(["NONE", "NONE"], "plain stem")
    assert letter in ("X", "Y"), letter
    print("ok")


if __name__ == "__main__":
    demo()
