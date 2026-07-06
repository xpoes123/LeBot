"""Remap sa_labels.json gold answers to the cleaned golds, and re-judge predictions
for any question whose gold changed. One-time fixup after improving parse.clean_answer.

Run: python fix_golds.py
"""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import answerer

LABELS = os.path.join(os.path.dirname(__file__), "sa_labels.json")
CLEAN = os.path.join(os.path.dirname(__file__), "packets", "questions_clean.json")


def _id(tournament, text):
    return hashlib.md5((tournament + text).encode()).hexdigest()[:10]


if __name__ == "__main__":
    clean_gold = {_id(q["tournament"], q["question_text"]): q["correct_answer"]
                  for q in json.load(open(CLEAN))}
    data = json.load(open(LABELS))

    changed = [rec for rec in data.values()
               if rec["id"] in clean_gold and clean_gold[rec["id"]] != rec["gold"]]
    print(f"{len(changed)}/{len(data)} questions have a cleaned gold -> re-judging those")

    def rejudge(rec):
        gold = clean_gold[rec["id"]]
        rec["gold"] = gold
        cache = {}
        for r in rec["preds"]:
            pred = r[1]
            if pred not in cache:
                cache[pred] = answerer.judge(pred, gold)
            r[2] = cache[pred]
        return rec

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(rejudge, changed))

    json.dump(data, open(LABELS, "w"))
    print(f"updated {LABELS}")
