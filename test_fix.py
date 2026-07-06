"""Which approach reliably recovers a hard recall answer the gut knows but reasoning
derails? Test on the brachistochrone question."""
from collections import Counter

import answerer

FULL = ("What shape is formed when light passes through an object where each "
        "infinitesimally thin layer of the object has an index of refraction that "
        "varies with the sine of the angle that light approaches it divided by the "
        "speed of the light at that layer?")
CAT = "PHYSICS"

# 1. self-consistency: sample the terse gut answer several times, take the vote
votes = answerer.anticipate_sa(FULL, CAT, n=8)
print("1. self-consistency (Sonnet gut x8):", dict(Counter(votes)))

# 2. stronger model, terse
opus = answerer._call(
    answerer.META_SA.format(category=CAT),
    f"SHORT-ANSWER question:\n\"{FULL}\"\n\nReply ONLY the terse answer.",
    max_tokens=20, temperature=0.0, model="claude-opus-4-8")
print("2. Opus terse:", repr(answerer._clean_answer(opus)))

# 3. name candidates then pick the most specific (structured System 2)
cand = answerer._call(
    answerer.META_SA.format(category=CAT),
    f"SHORT-ANSWER question:\n\"{FULL}\"\n\nList 2-3 candidate named answers (one per "
    "line). Then a final line exactly: ANSWER: <the single most specific correct term>.",
    max_tokens=180, temperature=0.3)
print("3. candidates-then-pick:\n   " + cand.replace("\n", "\n   "))
