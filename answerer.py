"""The brain. Stem-first, the way an elite Science Bowl player actually plays:

  1. predict the answer from the STEM ALONE (blind to the options) -> commits to
     knowledge instead of pattern-matching whichever option looks best.
  2. map that prediction onto W/X/Y/Z.

Confidence = self-consistency on the *letter*: sample the stem-answer N times, map
each to a letter, fraction agreeing = confidence. Anthropic exposes no logprobs, so
sampling is the only calibrated signal. Letter-level agreement (not phrase-level)
absorbs harmless wording differences ("mitochondria" vs "the mitochondrion").
"""

import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import anthropic

import sequences

LETTERS = ["W", "X", "Y", "Z"]
MODEL = os.environ.get("LEBOT_MODEL", "claude-sonnet-4-6")       # System 2: thinking
FAST = os.environ.get("LEBOT_FAST", "claude-haiku-4-5-20251001")  # System 1: subconscious
_client = None  # lazy: so consensus()/LETTERS import without an API key (--dry)


def _get_client():
    global _client
    if _client is None:
        # max_retries rides out transient 429/5xx/529 overloads with the SDK's
        # built-in exponential backoff (overnight batch runs hit 529 constantly).
        _client = anthropic.Anthropic(max_retries=8)  # reads ANTHROPIC_API_KEY
    return _client

SB_CONTEXT = """The U.S. National Science Bowl is a prestigious academic quiz \
competition for high-school teams; its questions are rigorous and routinely reach \
early-college material (modern physics, organic chemistry, multivariable calculus, \
biochemistry, named theorems and phenomena). Every question has ONE specific intended \
answer — a precise scientific term, name, law, constant, value, or named phenomenon \
(e.g. "tetrahedral", "Avogadro constant", "allosteric inhibition"), never a vague \
description. Reach for the specific advanced concept, not a generic one."""

# The Science Bowl meta, imparted to the model. Tier 1 = how you win; Tier 2 =
# last-resort tells used ONLY when the science is unknown.
META = SB_CONTEXT + """ You are an elite player answering a MULTIPLE-CHOICE \
toss-up worth 4 points, in a live buzz race against other experts. The moderator \
reads the question stem, then four options aloud in order: W, X, Y, Z. You may buzz \
the instant you know the answer.

How elite players win:
- KNOW IT FROM THE STEM. Most non-computational questions ("...is called what?", \
"which structure does X?") have one answer you can name before any option is read. \
Decide from the stem, then buzz the moment the matching option is read, even if it \
is the first one.
- COMPUTE ONLY WHEN FORCED. If the question needs a calculation, hear it fully.
- NEGATION FLIPS EVERYTHING. Watch for NOT / EXCEPT / LEAST / INCORRECT; the answer \
is the odd one out.
- "BEST describes/explains" means several options are plausible; choose the most \
precise and complete, not merely true.

Last-resort tells, ONLY when you do not know the science:
- Options are usually structured: numbers in order (watch units and orders of \
magnitude), or one textbook-correct term among plausible-sounding distractors.
- Prefer the most specific, technically precise option; eliminate options that are \
true statements but do not answer THIS question.
- When you DO know the science, ignore option structure and answer from knowledge.

Category for this question: {category}."""


def _norm(s):
    return s.strip().strip(".").lower()


def subconscious(prefix, category):
    """System 1: fast gut answer on a partial question. Haiku, temp 0, ~250ms.

    Returns a short normalized phrase, or None if it can't tell yet. Run this on
    every word tick; the answer holding stable across ticks IS the confidence.
    """
    text = _call(
        "You are an elite Science Bowl player's gut instinct. Answer fast.",
        f"Category: {category}. Partial Science Bowl question (may be cut off):\n"
        f"{prefix}\n\nYour single best 1-3 word answer, or '?' if you cannot tell yet.",
        max_tokens=16, temperature=0.0, model=FAST,
    )
    n = _norm(text)
    return None if not n or n.startswith("?") else n


def gate(prefix, options, category):
    """The BUZZER (System 1). Not an answerer: judges 'is this gettable yet?'.

    Haiku, every word, temp 0. Options are known up front; only the stem streams in.
    -> True if a top player could already commit to an answer. v0 = prompt; v1 = a
    classifier trained on buzzes.csv word_index.
    """
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    text = _call(
        "You are a National Science Bowl buzzer reflex. You do NOT answer; you only "
        "decide whether there is now enough information to buzz.",
        f"Category: {category}. The four options are known:\n{opts}\n\n"
        f"Question stem heard SO FAR (more words may follow):\n{prefix}\n\n"
        "Could a top player already pick the correct option with confidence? "
        "Reply YES or NO only.",
        max_tokens=8, temperature=0.0, model=FAST,
    )
    return text.strip().upper().startswith("Y")


def anticipate(prefix, options, category, n=5):
    """Question anticipation (the core idea). Given a CUT-OFF stem, predict where the
    question is going and which option it will land on. Sampled n times.

    Agreement across samples = confidence: once the identifying clue is heard, every
    plausible completion converges on one option -> buzz, even mid-sentence.
    Returns n letters (each W/X/Y/Z or 'NONE'). Uses the fast model (runs every word).
    """
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    system = META.format(category=category)

    def one(_):
        letter = _call(
            system,
            f"This Science Bowl question is CUT OFF mid-reading:\n\"{prefix}...\"\n\n"
            f"Options (known in advance):\n{opts}\n\n"
            "Anticipate how the question will finish and which option it is heading "
            "toward. If the answer is already determined, give its letter; if it is "
            "still ambiguous, reply NONE. Reply with ONLY W, X, Y, Z, or NONE.",
            max_tokens=16, temperature=0.8, model=FAST,
        ).upper()
        return letter[0] if letter and letter[0] in LETTERS else "NONE"

    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(one, range(n)))


META_SA = SB_CONTEXT + """ You are an elite player answering a SHORT-ANSWER toss-up in \
a live buzz race. Toss-ups are pyramidal: early words give general clues that narrow to \
one answer, so commit as soon as the answer is determined, before the question finishes. \
Recall questions are often answerable from the key concept alone; computational \
questions need the full setup. There is a POINT PENALTY for a wrong interrupt, so only \
commit when genuinely confident. Category: {category}.

CRITICAL — numbered-list questions: when the question presents a numbered list \
(e.g., "1) Azimuthal; 2) Magnetic; 3) Spin") and asks you to ORDER or IDENTIFY items, \
give ONLY the item numbers in your answer (e.g., "3, 1, 2" or "1, 3"), never the names. \
This is the required Science Bowl answer format for these questions.

CRITICAL — "must"/"always"/"necessarily" questions: for "identify all that MUST/ALWAYS \
[do X]" questions, include an item ONLY if the property is GUARANTEED, not merely \
possible. A relationship that CAN hold does not mean it MUST (e.g. raising significance \
raises power, but power can also rise via sample size, so significance need not increase). \
If none of the items are guaranteed, the answer is "0" (none) — this is a valid and \
common answer; do not force-pick an item just because the format lists several.

CRITICAL — multi-select ("identify all") questions: evaluate EACH numbered item \
INDEPENDENTLY as true/false, then report the COMPLETE set that qualifies (most answers are \
a PAIR; "0"/none and all are equally valid — do not force a nonzero answer). While items \
are STILL being read, do NOT commit a partial set: either answer by exclusion (below) or \
reply UNKNOWN until you can give the whole answer. Once EVERY item has been read, always \
give your best complete set — never abstain on a fully-read question.
ANSWER EARLY BY EXCLUSION — the key skill for "identify all" questions with a GENERAL \
RULE and a famous EXCEPTION. The stem states the item count, so you do NOT need to hear \
every item:
- Before any items have been read: reply UNKNOWN (you cannot name an exception you have \
not heard).
- The MOMENT you hear the EXCEPTION item (one you know breaks the general rule), answer \
EXACTLY "all except <that item's name>" — every other item follows the rule, so you commit \
without hearing the rest. (Use "none except <name>" if instead only the exceptions qualify.) \
NEVER write index numbers for items you have not yet heard.
- Only once ALL items have been read aloud may you answer with an index list like "1, 3".
Example: "identify all 3 amino acids that are chiral: 1) Alanine, 2) Glycine, 3) ..." — \
glycine is the sole achiral amino acid, so the instant you hear glycine answer "all except \
glycine", before item 3 is read. Only use exclusion when you are CERTAIN of the general rule.

ANSWER FORM (Science Bowl scoring): give the NAME of a concept, not a description \
("Newton's second law", not "F=ma"; canonical symbols like "c" are fine). For a person, \
LAST NAME ONLY ("Einstein"). OMIT units unless the question demands them. Numbers in \
exact simplest form — no scientific notation, no repeating decimals (use fractions). \
A chemical formula is only acceptable when the compound has no isomers (use the name \
otherwise). Keep it to the bare answer term — never a sentence."""


def _clean_answer(r):
    """Force terse: a buzz answer is a term/value, never a sentence/explanation."""
    r = r.strip().strip('"').strip("#* ").strip()
    if not r or r.upper().startswith("UNKNOWN"):
        return "UNKNOWN"
    r = r.split("\n")[0]
    # drop a lead-in like "The answer is" / "Answer:" the model sometimes prepends
    r = re.sub(r"(?i)^(the\s+)?answer\s+is\s*:?\s*|^answer\s*:\s*", "", r).strip()
    r = re.split(r"\s+(?:\(|—|–|-\s|,\s*which\b|because\b|since\b|is\b|are\b|refers\b)", r)[0]
    r = r.strip().strip(".").strip()
    if not r:
        return "UNKNOWN"
    # reasoning leaks: a trailing colon leads into an explanation ("For n = 3:"); a
    # sentence opener followed by more words is a truncated explanation, not a term.
    if r.endswith(":"):
        return "UNKNOWN"
    if re.match(r"(?i)(for|in the|since|because|if|when|as the|according|assuming|"
                r"we|let|first|note that|given|thus|therefore|so the)\b[\s,]+\S+", r):
        return "UNKNOWN"
    # a dangling connective/preposition at the end = truncated mid-phrase, not a clean
    # term (excludes "a"/"an" so "vitamin A", "plan B" survive)
    if re.search(r"(?i)\s(and|or|of|the|to|with|for|by|in|on|at|that|which|as)$", r):
        return "UNKNOWN"
    # a real short answer is brief; a long string means it explained -> not a clean commit
    return "UNKNOWN" if len(r.split()) > 7 or not r else r


def anticipate_sa(prefix, category, n=1):
    """Short-answer anticipation: predict the free-text answer from a cut-off stem.
    Returns n predictions (each a TERSE answer, or 'UNKNOWN')."""
    sys = META_SA.format(category=category)

    def one(_):
        r = _call(
            sys,
            f"This SHORT-ANSWER question is cut off mid-reading:\n\"{prefix}...\"\n\n"
            "Give ONLY the answer itself — a single term, name, or value, at most a few "
            "words (e.g. \"mitochondria\", \"contact metamorphism\", \"36 mph\"). "
            "NO sentence, NO explanation, do not write \"X is...\". "
            "If not yet determinable, reply UNKNOWN.\nAnswer:",
            max_tokens=12, temperature=0.7,
        )
        return _clean_answer(r)

    if n == 1:
        return [one(0)]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(one, range(n)))


import math
from fractions import Fraction

# whitelisted namespace for executing model-written calculation code (no builtins)
_SAFE = {"Fraction": Fraction, "math": math, "sqrt": math.sqrt, "pi": math.pi,
         "e": math.e, "log": math.log, "log2": math.log2, "log10": math.log10,
         "exp": math.exp, "factorial": math.factorial, "comb": math.comb,
         "perm": math.perm, "sin": math.sin, "cos": math.cos, "tan": math.tan,
         "radians": math.radians, "degrees": math.degrees, "gcd": math.gcd,
         "abs": abs, "round": round, "sum": sum, "min": min, "max": max,
         "pow": pow, "range": range, "len": len, "int": int, "float": float}


def _fmt(a):
    if isinstance(a, Fraction):
        return str(a.numerator) if a.denominator == 1 else f"{a.numerator}/{a.denominator}"
    if isinstance(a, float):
        return str(int(a)) if a == int(a) else f"{a:.4g}"
    return str(a)


_ALLOWED_MODS = {"math", "fractions", "itertools", "statistics", "cmath", "decimal"}


def _safe_import(name, *a, **k):
    if name.split(".")[0] in _ALLOWED_MODS:
        return __import__(name, *a, **k)
    raise ImportError(name)


def _calc(code):
    # ponytail: sandbox = restricted builtins (only safe math imports) + whitelist ns
    safe_builtins = {"__import__": _safe_import, "print": lambda *a, **k: None,
                     "range": range, "len": len, "int": int, "float": float,
                     "abs": abs, "round": round, "sum": sum, "min": min, "max": max,
                     "pow": pow, "enumerate": enumerate, "list": list, "map": map,
                     "zip": zip, "sorted": sorted, "set": set, "dict": dict, "tuple": tuple}
    ns = {"__builtins__": safe_builtins}
    ns.update(_SAFE)
    exec(code, ns)
    return ns.get("answer")


def solve(prefix, category, use_opus=False):
    """If `prefix` is a computational question with all numbers present, COMPUTE the
    answer (model writes Python -> we execute it). Returns a formatted answer, or None
    if it's not computational / numbers missing / code failed.

    use_opus routes the setup to Opus for harder reasoning; the calculator does the
    arithmetic either way so the answer is exact, not eyeballed.
    """
    model = "claude-opus-4-8" if use_opus else MODEL
    r = _call(
        "You solve National Science Bowl computational questions by writing Python. "
        f"Category: {category}.",
        f"Question (may be cut off mid-reading):\n\"{prefix}\"\n\n"
        "If this is a COMPUTATIONAL question AND every number needed is already present "
        "(numbers may be mangled like '2/3'->'23' — reconstruct sensible fractions if "
        "obvious), write Python that assigns the result to `answer`, using "
        "fractions.Fraction for exact ratios. Output ONLY a ```python code block. "
        "If it is not computational, or required numbers are missing/ambiguous, output "
        "exactly NOTCOMPUTABLE.",
        max_tokens=500, temperature=0.0, model=model,
    )
    if "NOTCOMPUTABLE" in r.upper():
        return None
    m = re.search(r"```(?:python)?\s*(.*?)```", r, re.DOTALL)
    code = m.group(1) if m else r
    try:
        a = _calc(code)
    except Exception:
        return None
    return _fmt(a) if a is not None else None


def anticipate_sa_verbose(prefix, category):
    """Same anticipation, but capture the reasoning too (for walkthroughs).
    -> (reasoning_sentence, terse_answer)."""
    raw = _call(
        META_SA.format(category=category),
        f"This SHORT-ANSWER question is cut off mid-reading:\n\"{prefix}...\"\n\n"
        "FIRST line, exactly: ANSWER: <a terse answer, or UNKNOWN>\n"
        "SECOND line: one short sentence explaining your reasoning.",
        max_tokens=120, temperature=0.7,
    )
    if "ANSWER:" in raw.upper():
        after = raw[raw.upper().index("ANSWER:") + 7:]
        parts = after.split("\n", 1)
        reasoning = parts[1].strip() if len(parts) > 1 else ""
        ans = _clean_answer(parts[0])
        resolved, is_excl = resolve_exclusion(ans, prefix)
        if is_excl:  # "all except X" -> grounded exclusion, bypass the blind guard
            return reasoning, resolved
        if _list_prior_guess(resolved, prefix):  # blind index-guess before items shown
            return reasoning, "UNKNOWN"
        return reasoning, resolved
    return raw.strip(), "UNKNOWN"  # no clean ANSWER line -> don't leak reasoning as answer


def _vote(cands):
    """Majority vote over terse answers (gut self-consistency). Reasoning can derail
    the model off an answer its gut knows; voting recovers it (brachistochrone: 7/8)."""
    counts, rep = Counter(), {}
    for a in cands:
        if a.upper() == "UNKNOWN":
            continue
        k = _norm(a)
        counts[k] += 1
        rep.setdefault(k, a)
    return rep[counts.most_common(1)[0][0]] if counts else "UNKNOWN"


_NUMWORD = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _list_prior_guess(ans, prefix):
    """True if `ans` is a numbered-list answer the model CAN'T actually know yet:
    a bare index-list (e.g. '1, 2, 3', '2, 3') or 'none'/0 whose referenced items
    aren't revealed in the stem so far. This is the blind format-prior — buzzing
    '1, 2, 3' just because the question says "identify all of the following", before
    seeing what 1/2/3 even are. Gated on the list cue ("following") so real numeric
    recall answers ("22 digits") are never touched."""
    a = (ans or "").strip().lower()
    if not re.fullmatch(r"\d+(\s*,\s*\d+)*", a):
        return False
    low = prefix.lower()
    if "following" not in low:                       # not a numbered-list question
        return False
    markers = set(re.findall(r"(\d+)\)", prefix))    # '1)', '2)' items revealed so far
    nums = [n for n in re.findall(r"\d+", a) if n != "0"]
    if not nums:                                     # '0' = none-of-them: need the WHOLE list
        m = re.search(r"following\s+(\w+)", low)
        g = m.group(1) if m else ""
        want = _NUMWORD.get(g) or (int(g) if g.isdigit() else None)
        return want is None or len(markers) < want
    return any(n not in markers for n in nums)       # references an item not yet shown


def _is_list_q(prefix):
    """A numbered-list (order/identify-all/rank) question — always answered by RECALL
    as indices, never the calculator (which would return raw values or a list repr)."""
    return bool(re.search(r"\d\)", prefix)) or any(
        w in prefix.lower() for w in ("identify all", "order the", "rank the"))


def _declared_count(prefix):
    m = re.search(r"following\s+(\w+|\d+)", prefix.lower())
    if not m:
        return None
    w = m.group(1)
    return _NUMWORD.get(w) or (int(w) if w.isdigit() else None)


def _list_items(prefix):
    """{index: item text lowercased} parsed from 'N) ...' in the stem so far."""
    return {int(i): t.strip().lower()
            for i, t in re.findall(r"(\d+)\)\s*([^;:\n]+?)(?=\s*\d+\)|[;:\n]|$)", prefix)}


def resolve_exclusion(ans, prefix):
    """Convert an EXCLUSION answer ('all except olfaction', 'all but 2', 'none except 3')
    into an index list, using the declared item count + the numbered items heard so far.
    This is how you answer a multi-select before all items are read — you only need the
    exception, since every other item satisfies the general rule. Returns (indices_str,
    True) when it resolves an exclusion, else (ans, False)."""
    a = (ans or "").strip().lower().rstrip(".")
    m = re.match(r"^(all|none)\s+(?:but|except|besides|other than)\s+(.+)$", a)
    if not m:
        return ans, False
    base, exc = m.group(1), m.group(2)
    declared = _declared_count(prefix)
    items = _list_items(prefix)
    if not declared:
        declared = max(items) if items else None
    if not declared:
        return ans, False
    excluded = set()
    for tok in re.split(r"\s*(?:,|and)\s*", exc):
        tok = tok.strip().strip(".").strip()
        if tok.isdigit():
            excluded.add(int(tok))
        else:
            for idx, name in items.items():
                if tok and (tok in name or name in tok):
                    excluded.add(idx)
    if not excluded:
        return ans, False  # couldn't identify the exception -> don't fabricate
    if base == "all":
        result = [i for i in range(1, declared + 1) if i not in excluded]
    else:  # "none except X" -> only the named ones qualify
        result = sorted(excluded)
    return (", ".join(map(str, result)) if result else "0"), True


def anticipate_best(prefix, category, n=3):
    """Router: numbers present -> COMPUTE (calculator, exact); else recall via
    majority-voted gut answers. -> (answer, mode='calc'|'recall'). The calculator
    returns None until enough numbers are present, so it waits for the operative ones."""
    seq = sequences.solve_ordering(prefix)  # deterministic canonical-sequence ordering
    if seq is not None:
        return seq, "seq"
    if not _is_list_q(prefix) and any(ch.isdigit() for ch in prefix):
        v = solve(prefix, category)
        if v is not None:
            return v, "calc"
    ans = _vote(anticipate_sa(prefix, category, n=n))
    resolved, is_excl = resolve_exclusion(ans, prefix)
    if is_excl:
        return resolved, "recall"  # grounded exclusion ("all except X") -> bypass blind guard
    if _list_prior_guess(resolved, prefix):
        resolved = "UNKNOWN"
    return resolved, "recall"


def stream_answer(prefix, category):
    """Stream the verbose call. Yields ('answer', str) as soon as the ANSWER: line
    is complete (~0.4s), then ('reasoning', str) when the stream finishes (~1.5s).
    This cuts effective latency from 1.5s to 0.4s — ~1 word instead of ~4."""
    system = META_SA.format(category=category)
    user = (
        f"This SHORT-ANSWER question is cut off mid-reading:\n\"{prefix}...\"\n\n"
        "FIRST line, exactly: ANSWER: <a terse answer, or UNKNOWN>\n"
        "SECOND line: one short sentence explaining your reasoning."
    )
    buf = ""
    answer_sent = False
    with _get_client().messages.stream(
        model=MODEL,
        max_tokens=120,
        temperature=0.7,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for chunk in stream.text_stream:
            buf += chunk
            if not answer_sent and "ANSWER:" in buf.upper():
                after = buf[buf.upper().index("ANSWER:") + 7:]
                if "\n" in after:
                    answer_sent = True
                    yield "answer", _clean_answer(after.split("\n")[0])
    # full buffer — extract reasoning from second line
    if "ANSWER:" in buf.upper():
        after = buf[buf.upper().index("ANSWER:") + 7:]
        parts = after.split("\n", 1)
        reasoning = parts[1].strip() if len(parts) > 1 else ""
    else:
        reasoning = ""
    if not answer_sent:
        yield "answer", _clean_answer(buf)
    yield "reasoning", reasoning


def judge(pred, gold):
    """Semantic correctness for short answer (synonyms/equivalent forms count)."""
    if not pred or pred.upper() == "UNKNOWN":
        return False
    if _norm(pred) == _norm(gold):
        return True
    # rescue pdftotext-mangled fractions: gold "4/5" was scanned as "45"
    dp, dg = re.sub(r"\D", "", pred), re.sub(r"\D", "", gold)
    if dp and dp == dg and (("/" in pred) != ("/" in gold)):
        return True
    r = _call(
        "You judge National Science Bowl short answers. Equivalent forms, synonyms, "
        "and correct values (any units) count as correct; be reasonably lenient.",
        f"Expected answer: {gold}\nContestant said: {pred}\n\nIs it correct? YES or NO.",
        max_tokens=4, temperature=0.0, model=FAST,
    )
    return r.strip().upper().startswith("Y")


def blind(options, category, n=5, model=None):
    """Control arm: guess from the options + category with NO stem at all.

    Measures the pure MC option-prior. If anticipation only matches this, the stem
    added nothing — the bot is just exploiting test-taking priors, not anticipating.
    Pass the SAME model used for the stem-guess so the control is fair. Returns n letters.
    """
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))

    def one(_):
        letter = _call(
            "You are guessing a multiple-choice answer with NO question, only options.",
            f"Category: {category}. Options:\n{opts}\n\nYou have NOT heard the "
            "question. Guess the most likely correct option. Reply ONLY W/X/Y/Z.",
            max_tokens=16, temperature=0.8, model=model or FAST,
        ).upper()
        return letter[0] if letter and letter[0] in LETTERS else "NONE"

    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(one, range(n)))


def guess(prefix, options, category, n=5):
    """The GUESSER (System 2): which letter, from a partial stem + known options."""
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    system = META.format(category=category)

    def one(_):
        letter = _call(
            system,
            f"Question stem so far:\n{prefix}\n\nOptions:\n{opts}\n\n"
            "Which option is correct? Reply with ONLY W, X, Y, Z, or NONE.",
            max_tokens=5, temperature=0.7,
        ).upper()
        return letter[0] if letter and letter[0] in LETTERS else "NONE"

    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(one, range(n)))


def _once(m, system, user, max_tokens, temperature):
    kw = dict(
        model=m, max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    if not m.startswith("claude-opus-4-8"):  # opus 4.8 deprecates temperature
        kw["temperature"] = temperature
    return _get_client().messages.create(**kw)


def _text(msg):
    for b in msg.content:  # skip non-text blocks; tolerate empty content
        if getattr(b, "type", None) == "text":
            return b.text.strip()
    return ""


def _call(system, user, max_tokens, temperature, model=None):
    m = model or MODEL
    msg = _once(m, system, user, max_tokens, temperature)
    # Sonnet over-refuses benign science (pathogens, toxins) with stop_reason=refusal
    # and empty content. Haiku refuses far less — fall back so we don't silently abstain.
    if (msg.stop_reason == "refusal" or not _text(msg)) and m != FAST:
        msg = _once(FAST, system, user, max_tokens, temperature)
    return _text(msg)


def _predict_then_locate(stem, options, category):
    """One sample: answer the stem blind, then map to a letter (or NONE)."""
    system = META.format(category=category)
    phrase = _call(
        system,
        f"Question stem (options NOT yet revealed):\n{stem}\n\n"
        "Answer the question in a few words. If it requires a calculation you cannot "
        "do yet, or you cannot answer without seeing the options, reply NEEDOPTIONS.",
        max_tokens=30, temperature=0.8,
    )
    if "NEEDOPTIONS" in phrase.upper():
        return "NONE"
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    letter = _call(
        system,
        f"Question: {stem}\n\nOptions:\n{opts}\n\nYour intended answer: {phrase}\n\n"
        "Which option letter matches your answer? Reply with ONLY W, X, Y, Z, or NONE.",
        max_tokens=16, temperature=0.0,
    ).upper()
    return letter[0] if letter and letter[0] in LETTERS else "NONE"


def predict_letters(stem, options, category, n=5):
    """-> list of n letters (each in W/X/Y/Z or 'NONE')."""
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(
            lambda _: _predict_then_locate(stem, options, category), range(n)))


def consensus(letters):
    """-> (modal_letter or None, confidence). NONE votes count against confidence."""
    counts = Counter(x for x in letters if x != "NONE")
    if not counts:
        return None, 0.0
    letter, k = counts.most_common(1)[0]
    return letter, k / len(letters)


def consensus_debiased(letters, stem=None, options=None):
    """Like consensus() but blends the votes with the empirical, stem-adjusted MC letter
    prior (X/Y inflated; negation->Y/Z; closest-to->X/Y; largest-numeric down). Unanimous
    votes still win; split votes get broken by the prior. -> (letter or None, confidence)."""
    import mc_prior
    if not any(l in LETTERS for l in letters):
        return None, 0.0
    return mc_prior.debias(letters, stem or "", options)


if __name__ == "__main__":
    ls = predict_letters(
        "What is the powerhouse of the cell?",
        ["Nucleus", "Mitochondrion", "Ribosome", "Golgi apparatus"],
        "BIOLOGY", n=5,
    )
    letter, conf = consensus(ls)
    print(f"letters={ls} -> {letter} @ {conf}")
    assert letter == "X", f"expected X (Mitochondrion), got {letter}"
    print("ok")
