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

# --- exclusion resolution: "all except X" -> indices, before all items are read ---
from answerer import resolve_exclusion

P = "Identify all of the following 3 senses whose signals relay through the thalamus: 1) Vision, 2) Olfaction,"
# heard olfaction (item 2), item 3 not yet read -> "all except olfaction" = 1, 3
assert resolve_exclusion("all except olfaction", P) == ("1, 3", True)
assert resolve_exclusion("all but 2", P) == ("1, 3", True)
# "none except" -> only the named ones
assert resolve_exclusion("none except olfaction", P) == ("2", True)
# a plain index answer is NOT an exclusion -> passthrough
assert resolve_exclusion("1, 3", P) == ("1, 3", False)
assert resolve_exclusion("mitochondria", P) == ("mitochondria", False)
# can't identify the exception -> don't fabricate
assert resolve_exclusion("all except taste", P) == ("all except taste", False)

print("ok")

# --- anti-parrot: don't answer with a phrase read verbatim in the stem ---
from answerer import _parrots_stem
S2 = "Granzyme B, secreted by cytotoxic T cells, activates the executioner variety of what"
assert _parrots_stem("cytotoxic T cells", S2)          # phrase lifted from the stem
assert _parrots_stem("Granzyme B", S2)
assert not _parrots_stem("caspases", S2)                # real answer, not in stem
assert not _parrots_stem("apoptosis", S2)               # single word -> not guarded
assert not _parrots_stem("UNKNOWN", S2)
print("parrot ok")
