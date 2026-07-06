"""Mechanical ordering solver: for "order/rank the following by <axis>" questions whose
axis is a fixed canonical sequence or a lookup table, compute the index permutation with
ZERO LLM calls. Conservative — returns None (→ LLM fallback) unless every item maps
cleanly to the matched axis.

solve_ordering(stem) -> "3, 1, 2" | None
"""
import re

# --- Reference data (accurate; common Science Bowl elements/minerals/bands) -----------

# Pauling electronegativity
EN = {"h": 2.20, "li": 0.98, "be": 1.57, "b": 2.04, "c": 2.55, "n": 3.04, "o": 3.44,
      "f": 3.98, "na": 0.93, "mg": 1.31, "al": 1.61, "si": 1.90, "p": 2.19, "s": 2.58,
      "cl": 3.16, "k": 0.82, "ca": 1.00, "br": 2.96, "i": 2.66, "fe": 1.83, "cu": 1.90,
      "zn": 1.65, "ge": 2.01, "as": 2.18, "se": 2.55, "kr": 3.00, "rb": 0.82, "sr": 0.95}

# Covalent radius (pm) — atomic size
RADIUS = {"h": 31, "li": 128, "be": 96, "b": 84, "c": 76, "n": 71, "o": 66, "f": 57,
          "ne": 58, "na": 166, "mg": 141, "al": 121, "si": 111, "p": 107, "s": 105,
          "cl": 102, "ar": 106, "k": 203, "ca": 176, "br": 120, "i": 139, "kr": 116,
          "rb": 220, "sr": 195, "fe": 132, "cu": 132, "zn": 122}

# First ionization energy (kJ/mol)
IE = {"h": 1312, "li": 520, "be": 899, "b": 801, "c": 1086, "n": 1402, "o": 1314,
      "f": 1681, "ne": 2081, "na": 496, "mg": 738, "al": 578, "si": 786, "p": 1012,
      "s": 1000, "cl": 1251, "ar": 1521, "k": 419, "ca": 590, "br": 1140, "i": 1008,
      "kr": 1351, "rb": 403, "sr": 549, "fe": 762, "cu": 745, "zn": 906}

ELEMENT_NAME = {
    "hydrogen": "h", "lithium": "li", "beryllium": "be", "boron": "b", "carbon": "c",
    "nitrogen": "n", "oxygen": "o", "fluorine": "f", "neon": "ne", "sodium": "na",
    "magnesium": "mg", "aluminum": "al", "aluminium": "al", "silicon": "si",
    "phosphorus": "p", "sulfur": "s", "sulphur": "s", "chlorine": "cl", "argon": "ar",
    "potassium": "k", "calcium": "ca", "bromine": "br", "iodine": "i", "krypton": "kr",
    "rubidium": "rb", "strontium": "sr", "iron": "fe", "copper": "cu", "zinc": "zn",
    "germanium": "ge", "arsenic": "as", "selenium": "se"}

# Ordered canonical sequences: rank index = position in the list (0 = "least")
MOHS = ["talc", "gypsum", "calcite", "fluorite", "apatite", "orthoclase", "feldspar",
        "quartz", "topaz", "corundum", "diamond"]
METAMORPHIC = ["slate", "phyllite", "schist", "gneiss", "migmatite"]  # increasing grade
TAXONOMY = ["domain", "kingdom", "phylum", "division", "class", "order", "family",
            "genus", "species"]  # broad -> specific
PLANETS = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
EM = ["radio", "microwave", "infrared", "visible", "ultraviolet", "x-ray", "xray",
      "gamma"]  # increasing frequency/energy, decreasing wavelength
COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]  # inc freq/energy

# --- Axis detection -------------------------------------------------------------------
# Each axis: (list of trigger phrases, value-function). Value-function returns a sortable
# number for an item name, or None if the item isn't on this axis. Larger value = later in
# the "increasing" direction.

def _norm(s):
    return re.sub(r"[^\w\s-]", "", s.strip().lower()).strip()


def _elem_key(name):
    n = _norm(name)
    if n in ELEMENT_NAME:
        return ELEMENT_NAME[n]
    if n in EN or n in RADIUS or n in IE:  # already a symbol
        return n
    # symbol like "Cl" possibly with charge/subscript stripped
    m = re.match(r"^([a-z]{1,2})\b", n)
    return m.group(1) if m else None


def _table_val(table):
    def f(item):
        k = _elem_key(item)
        return table.get(k) if k else None
    return f


def _seq_val(seq):
    def f(item):
        n = _norm(item)
        for i, entry in enumerate(seq):
            if entry == n or entry in n.split() or n in entry:
                return i
        return None
    return f


# (trigger regex, value fn, is_element_table) — element tables are neutral-atom/first-IE
AXES = [
    (r"electronegativit", _table_val(EN), True),
    (r"first ionization energ|ionization energ|ionisation energ", _table_val(IE), True),
    (r"atomic radi|atomic size|covalent radi", _table_val(RADIUS), True),
    (r"mohs|hardness", _seq_val(MOHS), False),
    (r"metamorphic grade", _seq_val(METAMORPHIC), False),
    (r"taxonomic|classification rank|taxonomy", _seq_val(TAXONOMY), False),
    (r"distance from the sun|from the sun", _seq_val(PLANETS), False),
    (r"wavelength", _seq_val(EM + COLORS), False),
    (r"frequency|photon energy", _seq_val(EM + COLORS), False),
]

# axes where the stem property DECREASES as you go later in the canonical list under
# "increasing" (e.g. increasing wavelength = radio->gamma reversed)
_INVERSE_UNDER_INCREASING = {"wavelength"}


def _direction(stem):
    """+1 if the question wants ascending (least->greatest), -1 if descending."""
    s = stem.lower()
    # explicit "A to B" phrasings win (the FROM end sets the direction)
    asc_from = r"(least|lowest|smallest|shortest|softest|weakest|youngest|most inclusive|" \
               r"broadest|most general|coarsest|finest)\b.{0,30}\bto\b"
    desc_from = r"(greatest|highest|largest|longest|hardest|strongest|oldest|most massive|" \
                r"most exclusive|most specific)\b.{0,30}\bto\b"
    if re.search(asc_from, s):
        return 1
    if re.search(desc_from, s):
        return -1
    if re.search(r"increas|from low|softest|most inclusive|broadest|most general", s):
        return 1
    if re.search(r"decreas|from high|hardest|most exclusive|most specific|most massive|"
                 r"greatest|highest|largest|longest|strongest|oldest", s):
        return -1
    return 1  # default ascending


def _items(stem):
    """Parse numbered items '1) X; 2) Y; 3) Z' -> {1: 'X', 2: 'Y', ...}."""
    out = {}
    for m in re.finditer(r"(\d+)\)\s*([^;:\n]+?)(?=\s*\d+\)|[;:\n]|$)", stem):
        out[int(m.group(1))] = m.group(2).strip()
    return out


def solve_ordering(stem):
    """-> index-permutation string like '3, 1, 2', or None if not confidently solvable."""
    if not re.search(r"\b(order|rank|arrange)\b", stem, re.I):
        return None
    items = _items(stem)
    if len(items) < 2:  # numbered items are what make it a list question
        return None
    # only fire once ALL declared items are present (live stems arrive word-by-word)
    m = re.search(r"following\s+(two|three|four|five|six|\d+)", stem, re.I)
    if m:
        w = m.group(1).lower()
        declared = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}.get(w)
        declared = declared or (int(w) if w.isdigit() else None)
        if declared and declared != len(items):
            return None

    low = stem.lower()
    # element property tables are for NEUTRAL atoms + FIRST ionization only
    ionic = re.search(r"\b(ion|ions|anion|cation|anions|cations)\b|[+-]\s*\d?\s*$", low) \
        or re.search(r"\b\w+\s*[+-]\b", stem)
    higher_ie = re.search(r"\b(second|third|fourth|2nd|3rd|4th)\s+ionization", low)

    for trig, valfn, is_elem_table in AXES:
        if not re.search(trig, low):
            continue
        if is_elem_table and (ionic or higher_ie):
            return None  # neutral-atom first-IE tables don't apply to ions / higher IE
        vals = {i: valfn(name) for i, name in items.items()}
        if any(v is None for v in vals.values()):
            return None  # some item off the axis -> let the LLM handle it
        if len(set(vals.values())) != len(vals):
            return None  # ties -> ambiguous, bail

        direction = _direction(stem)
        # wavelength is inverse: increasing wavelength = decreasing EM/color position
        axis_word = re.search(trig, low).group(0)
        if any(w in axis_word for w in _INVERSE_UNDER_INCREASING):
            direction *= -1

        order = sorted(vals, key=lambda i: vals[i] * direction)
        return ", ".join(str(i) for i in order)
    return None


def demo():
    # electronegativity increasing: Na(0.93) < Al(1.61) < Cl(3.16) -> 3,1,2 given order Na,Al,Cl
    q1 = ("Rank the following three elements by increasing electronegativity: "
          "1) Sodium; 2) Aluminum; 3) Chlorine")
    assert solve_ordering(q1) == "1, 2, 3", solve_ordering(q1)
    # atomic radius decreasing across a period: Na(166)>Si(111)>Cl(102) -> 1,2,3
    q2 = ("Order the following by decreasing atomic radius: 1) Sodium; 2) Silicon; 3) Chlorine")
    assert solve_ordering(q2) == "1, 2, 3", solve_ordering(q2)
    # increasing wavelength: gamma < visible < radio -> gamma shortest -> 3,2,1? items: radio,visible,gamma
    q3 = ("Rank by increasing wavelength: 1) Radio waves; 2) Visible light; 3) Gamma rays")
    assert solve_ordering(q3) == "3, 2, 1", solve_ordering(q3)
    # Mohs increasing hardness: talc<quartz<diamond given order quartz,talc,diamond -> 2,1,3
    q4 = ("Order these minerals by increasing hardness on the Mohs scale: "
          "1) Quartz; 2) Talc; 3) Diamond")
    assert solve_ordering(q4) == "2, 1, 3", solve_ordering(q4)
    # taxonomy broad->specific: family, phylum, genus -> phylum(2) < family(5) < genus(7): 2,1,3
    q5 = ("Arrange the following taxonomic ranks from broadest to most specific: "
          "1) Family; 2) Phylum; 3) Genus")
    assert solve_ordering(q5) == "2, 1, 3", solve_ordering(q5)
    # non-canonical axis (viscosity) -> None
    q6 = ("Rank the following liquids by increasing viscosity: 1) Water; 2) Honey; 3) Oil")
    assert solve_ordering(q6) is None, solve_ordering(q6)
    # not an ordering question -> None
    assert solve_ordering("What is the powerhouse of the cell?") is None
    print("ok")


if __name__ == "__main__":
    demo()
