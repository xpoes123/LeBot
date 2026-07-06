"""Check _clean_answer catches reasoning leaks without nuking valid terse answers."""
from answerer import _clean_answer

# reasoning leaks -> UNKNOWN
assert _clean_answer("For n = 3:") == "UNKNOWN"
assert _clean_answer("In the island biogeography model") == "UNKNOWN"
assert _clean_answer("Since the power increases") == "UNKNOWN"
assert _clean_answer("First, we compute the") == "UNKNOWN"
assert _clean_answer("Given the quantum number is") == "UNKNOWN"
assert _clean_answer("When the reservoir cools") == "UNKNOWN"

# valid terse answers -> preserved
assert _clean_answer("ATP synthase") == "ATP synthase"
assert _clean_answer("Eddington limit") == "Eddington limit"
assert _clean_answer("Transducin") == "Transducin"
assert _clean_answer("3, 1, 2") == "3, 1, 2"
assert _clean_answer("Newton's second law") == "Newton's second law"
assert _clean_answer("Indium") == "Indium"          # starts with "In" but not "in the"
assert _clean_answer("Fermat's principle") == "Fermat's principle"  # not "For..."
assert _clean_answer("phosphorescence") == "phosphorescence"
assert _clean_answer("UNKNOWN") == "UNKNOWN"

print("ok")
