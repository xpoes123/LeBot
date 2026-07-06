"""Load Stanford Science Bowl questions + human buzzes from the scibowl-org CSV facts.

Everything we need lives in two files under <facts_dir>:
  questions_meta.csv  -- stem, options, correct answer, category (keyed by packet_checksum,question_id)
  buzzes.csv          -- every human buzz with location + result (same key)

No DB, no pandas. ponytail: stdlib csv handles 1.5MB fine.
"""

import csv
import json
import os

csv.field_size_limit(10_000_000)  # question_text can be long

LETTERS = ["W", "X", "Y", "Z"]


def _key(row):
    return (row["packet_checksum"], row["question_id"])


def load_questions(facts_dir, mc_only=True, tossup_only=True):
    """-> dict[(checksum, qid)] = question dict."""
    out = {}
    with open(os.path.join(facts_dir, "questions_meta.csv")) as f:
        for row in csv.DictReader(f):
            if mc_only and row["question_style"] != "MULTIPLE_CHOICE":
                continue
            if tossup_only and row["question_type"] != "TOSSUP":
                continue
            answer = json.loads(row["correct_answer"])  # e.g. "Z"
            if answer not in LETTERS:
                continue  # skip IDENTIFY_ALL/RANK etc.
            out[_key(row)] = {
                "checksum": row["packet_checksum"],
                "qid": row["question_id"],
                "category": row["category"],
                "stem": row["question_text_stripped"],
                "options": json.loads(row["options_json"]),
                "answer": answer,
                "answer_idx": LETTERS.index(answer),
            }
    return out


def human_best_step(buzzes_for_q):
    """Earliest CORRECT human buzz as a reveal step (lower = earlier).

    stem buzz -> -1 (before options; MC humans rarely do this)
    option k  ->  k   (0..3)
    end       ->  4
    no correct buzz -> inf (nobody got it; bot correct = free points)
    """
    best = float("inf")
    for b in buzzes_for_q:
        if b["result"] != "correct":
            continue
        kind = b["location_kind"]
        if kind == "question":
            step = -1
        elif kind == "option":
            step = int(b["option_index"])
        else:  # end
            step = 4
        best = min(best, step)
    return best


def human_best_word(buzzes_for_q, nwords):
    """Earliest CORRECT human buzz on a unified WORD axis (lower = earlier).

    stem buzz -> its word_index (this is the early-buzz battlefield)
    option buzz -> nwords + option_index   (after the stem finished)
    end -> nwords + 10
    no correct buzz -> inf
    """
    best = float("inf")
    for b in buzzes_for_q:
        if b["result"] != "correct":
            continue
        kind = b["location_kind"]
        if kind == "question":
            step = int(b["word_index"])
        elif kind == "option":
            step = nwords + int(b["option_index"])
        else:
            step = nwords + 10
        best = min(best, step)
    return best


def load_buzzes(facts_dir):
    """-> dict[(checksum, qid)] = list of buzz rows."""
    out = {}
    with open(os.path.join(facts_dir, "buzzes.csv")) as f:
        for row in csv.DictReader(f):
            out.setdefault(_key(row), []).append(row)
    return out


if __name__ == "__main__":
    # smoke test against the real Stanford facts
    fd = os.path.expanduser(
        "~/code/scibowl-org/stats/stanford-science-bowl/facts"
    )
    qs = load_questions(fd)
    bz = load_buzzes(fd)
    print(f"{len(qs)} MC tossups, {sum(len(v) for v in bz.values())} buzz rows")
    k = next(iter(qs))
    q = qs[k]
    print(f"\nsample {k}: [{q['category']}] {q['stem'][:70]}...")
    print(f"  answer={q['answer']}  human_best_step={human_best_step(bz.get(k, []))}")
