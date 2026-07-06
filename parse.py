"""Parse pdftotext'd Science Bowl packets into structured questions (our schema).

Yields dicts: tournament, packet, question_type (TOSSUP/BONUS), question_style
(SHORT_ANSWER/MULTIPLE_CHOICE), category, question_text, options[], correct_answer.

Run: python parse.py  -> writes packets/questions.json + prints stats/samples.
"""
import glob
import json
import os
import re
from collections import Counter

TXT_DIR = os.path.join(os.path.dirname(__file__), "packets", "txt")
OUT = os.path.join(os.path.dirname(__file__), "packets", "questions.json")

STYLE = {"SHORT ANSWER": "SHORT_ANSWER", "MULTIPLE CHOICE": "MULTIPLE_CHOICE"}


def norm_cat(c):
    u = re.sub(r"\s+", " ", c.strip().upper())
    if u.startswith("EARTH"):
        return "EARTH_SPACE"
    for key, pref in (("MATH", "MATH"), ("BIO", "BIOLOGY"), ("CHEM", "CHEMISTRY"),
                      ("PHYS", "PHYSICS"), ("ENERG", "ENERGY")):
        if u.startswith(key):
            return pref
    return "OTHER"


# A question header: optional TOSS-UP/BONUS, number), CATEGORY [- ] STYLE, then body up
# to the next header. Category captured loosely (handles "Energy - Short Answer",
# "EARTH AND SPACE Multiple Choice", "X-Risk - ..."); normalized via norm_cat.
QSTART = re.compile(
    r"(?:(TOSS[\s-]?UP|BONUS)\s+)?(\d+)\)\s+([A-Za-z][A-Za-z &/-]{1,28}?)\s*[-–—]?\s*"
    r"(Short Answer|Multiple Choice)\b",
    re.IGNORECASE)
OPT = re.compile(r"\b([WXYZ])\)\s*(.+?)(?=\s*\b[WXYZ]\)|\s*ANSWER:|\Z)", re.IGNORECASE | re.DOTALL)


def _strip_noise(text):
    """Drop repeated header/footer lines (page furniture repeats across pages)."""
    lines = text.split("\n")
    freq = Counter(l.strip() for l in lines if l.strip())
    out = []
    for l in lines:
        s = l.strip()
        if not s:
            continue
        # a short line repeated many times is page furniture, not content
        # (but keep TOSS-UP/BONUS markers — they repeat yet carry question_type)
        if (freq[s] >= 4 and len(s) < 60 and not QSTART.match(s)
                and not re.match(r"(?i)^(TOSS-?UP|BONUS)$", s)):
            continue
        out.append(s)
    return " ".join(out)


def clean_answer(s):
    """Clean a short-answer gold: drop writer initials [XX] and 'End of Questions',
    keep ACCEPT-alternatives as '/ alt' (the judge should honor them), drop
    'do not accept'/'prompt' notes."""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\[[^\]]*\]", "", s)                                  # [AG] initials
    s = re.sub(r"(?i)\((?:accept|or)[:\s]+([^)]*)\)", r" / \1", s)        # accepts -> / alt
    s = re.sub(r"(?i)\s*\((?:do not accept|prompt|reject)[^)]*\)", "", s)  # drop these
    s = re.sub(r"(?i)\bend of questions?\b.*$", "", s)
    return re.sub(r"\s+", " ", s).strip(" .;:/").strip()


def parse_file(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    text = _strip_noise(raw)
    rel = os.path.relpath(path, TXT_DIR)
    tournament = rel.split(os.sep)[0]
    packet = os.path.basename(rel).replace(".pdf.txt", "")

    matches = list(QSTART.finditer(text))
    out = []
    for i, m in enumerate(matches):
        body = text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        am = re.search(r"ANSWER:\s*(.+?)(?:\[[^\]]*\])?\s*$", body, re.DOTALL)
        if not am:
            continue
        answer_raw = am.group(1).strip()
        stem_and_opts = body[: am.start()].strip()
        style = STYLE[m.group(4).upper()]
        rec = {
            "tournament": tournament, "packet": packet,
            "question_type": "BONUS" if (m.group(1) or "").upper().startswith("BONUS") else "TOSSUP",
            "question_style": style,
            "category": norm_cat(m.group(3)),
            "options": [],
        }
        if style == "MULTIPLE_CHOICE":
            opts = OPT.findall(stem_and_opts)
            if len(opts) < 2:
                continue  # malformed MC
            first_opt = stem_and_opts.find(opts[0][0] + ")")
            rec["question_text"] = stem_and_opts[:first_opt].strip()
            rec["options"] = [re.sub(r"\s+", " ", o[1]).strip() for o in opts]
            lm = re.match(r"\s*([WXYZ])\)", answer_raw, re.IGNORECASE)
            rec["correct_answer"] = lm.group(1).upper() if lm else answer_raw
        else:
            rec["question_text"] = re.sub(r"\s+", " ", stem_and_opts).strip()
            rec["correct_answer"] = clean_answer(answer_raw)
        if rec["question_text"]:
            out.append(rec)
    return out


if __name__ == "__main__":
    files = glob.glob(os.path.join(TXT_DIR, "**", "*.txt"), recursive=True)
    allq = []
    for f in files:
        allq += parse_file(f)
    json.dump(allq, open(OUT, "w"))
    print(f"parsed {len(files)} packets -> {len(allq)} questions")
    print("by style:", dict(Counter(q["question_style"] for q in allq)))
    print("by type:", dict(Counter(q["question_type"] for q in allq)))
    print("by category:", dict(Counter(q["category"] for q in allq)))
    mc = [q for q in allq if q["question_style"] == "MULTIPLE_CHOICE"]
    bad = [q for q in mc if q["correct_answer"] not in ("W", "X", "Y", "Z")]
    print(f"MC: {len(mc)}  | MC with unparsed answer: {len(bad)}")
    print("\n--- 3 sample MC ---")
    for q in mc[:3]:
        print(f"[{q['category']}/{q['question_type']}] {q['question_text'][:90]}")
        print(f"    opts={[o[:18] for o in q['options']]} ans={q['correct_answer']}")
