"""Check the blind list-prior guard: don't answer index-lists before items are shown."""
from answerer import _list_prior_guess

S = "Identify all of the following 3 quantities that must increase if the power"

# blind: guessing all indices before ANY item is revealed
assert _list_prior_guess("1, 2, 3", S)
# blind: references items 2 and 3 but only "1)" has been read
assert _list_prior_guess("2, 3", S + " ... increased: 1) P-value;")
# grounded: every referenced item is visible -> allowed
assert not _list_prior_guess("2, 3", S + ": 1) P-value; 2) Level; 3) Type II error")
# 'none'/0 before the full list is shown -> blind
assert _list_prior_guess("0", "Identify all of the following three quantities that must increase")
# 'none'/0 with the full list visible -> allowed
assert not _list_prior_guess("0", "following three quantities: 1) a; 2) b; 3) c")
# real numeric recall answer, no list framing -> NEVER guarded
assert not _list_prior_guess("22", "how many digits does 3 to the power of 46 have?")
assert not _list_prior_guess("2.5", "what is the current in the circuit at t = 10?")
# normal word answer -> not a list at all
assert not _list_prior_guess("Transducin", S)

print("ok")
