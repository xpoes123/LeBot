"""Generate labeled (question, prefix) -> correct? data for the short-answer buzzer.

For ~300 short-answer tossups spread across tournaments, anticipate the answer at
strided stem prefixes (n=1) and judge each prediction semantically. Judgments are
deduped by (gold, pred) so we don't re-judge repeats. Idempotent per question.

Output: sa_labels.json  ->  list of {id, tournament, category, nwords, gold,
                                      preds: [[word_idx, pred, correct], ...]}
"""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import answerer

CLEAN = os.path.join(os.path.dirname(__file__), "packets", "questions_clean.json")
OUT = os.path.join(os.path.dirname(__file__), "sa_labels.json")
STRIDE = 3
PER_TOURNAMENT = 25   # cap to spread the ~300 across many tournaments
TARGET = 300


def _id(q):
    return hashlib.md5((q["tournament"] + q["question_text"]).encode()).hexdigest()[:10]


def pick_questions():
    qs = [q for q in json.load(open(CLEAN))
          if q["question_style"] == "SHORT_ANSWER" and q["question_type"] == "TOSSUP"
          and 8 <= len(q["question_text"].split()) <= 70]
    by_t, picked = {}, []
    for q in qs:
        by_t.setdefault(q["tournament"], []).append(q)
    # round-robin across tournaments for diversity
    while len(picked) < TARGET and any(by_t.values()):
        for t in list(by_t):
            if by_t[t] and sum(1 for p in picked if p["tournament"] == t) < PER_TOURNAMENT:
                picked.append(by_t[t].pop())
            if len(picked) >= TARGET:
                break
    return picked[:TARGET]


def process(q):
    words = q["question_text"].split()
    idx = sorted(set(list(range(0, len(words), STRIDE)) + [len(words) - 1]))
    prefixes = [" ".join(words[: j + 1]) for j in idx]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(
            lambda p: answerer.anticipate_best(p, q["category"]), prefixes))
    judged = {}  # dedup judging within this question
    recs = []
    for j, (pred, mode) in zip(idx, results):
        if pred not in judged:
            judged[pred] = answerer.judge(pred, q["correct_answer"])
        recs.append([j, pred, judged[pred], mode])
    return _id(q), {
        "id": _id(q), "tournament": q["tournament"], "category": q["category"],
        "nwords": len(words), "gold": q["correct_answer"], "preds": recs,
    }


if __name__ == "__main__":
    import threading
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    lock = threading.Lock()
    qs = [q for q in pick_questions() if _id(q) not in out]
    print(f"{len(qs)} short-answer tossups to (re)generate across "
          f"{len(set(q['tournament'] for q in qs))} tournaments")
    done = [0]

    def work(q):
        qid, rec = process(q)
        with lock:
            out[qid] = rec
            done[0] += 1
            json.dump(out, open(OUT, "w"))
            if done[0] % 10 == 0:
                print(f"  {done[0]}/{len(qs)}")

    with ThreadPoolExecutor(max_workers=4) as ex:  # 4 questions x 6 prefixes concurrent
        list(ex.map(work, qs))
    print(f"wrote {OUT} ({len(out)} questions)")
