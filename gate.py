"""The LIVE buzz gate, trained on the live pipeline's own trajectories.

buzzer.py learned from the offline verbose pipeline with the TRUE question length — live
has neither. This learns P(current fast guess is correct) from what live actually sees:
the fast Sonnet lean + Haiku check per tick (measure.py's cache), Jev's P, and words heard
so far (an absolute count, no length). Validated leave-one-tournament-out against the
human buzzes: every number is on a tournament the gate never trained on.

  python gate.py            # LOTO report (judge calls cached to .cache_judge.json)
"""
import json
import math
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import data
import measure
from buzzer import CATS

SETS = ["stanford-science-bowl", "johns-hopkins-invitational", "dasoni-2", "stanford-college-plus-science-bowl"]
JUDGE_CACHE = os.path.join(os.path.dirname(__file__), ".cache_judge.json")


def feats(rec, i):
    """Features at tick i, from ticks 0..i only (everything live knows at that moment)."""
    prev, run, arun, seen = None, 0, 0, set()
    for (g, agrees), _j in list(zip(rec["seq"], rec["idx"]))[: i + 1]:
        unk = not g or g.upper() == "UNKNOWN"
        ng = None if unk else measure._norm_answer(g)
        run = 0 if unk else (run + 1 if ng == prev else 1)
        arun = arun + 1 if (agrees and ng == prev) else (1 if agrees else 0)
        prev = ng
        if not unk:
            seen.add(ng)
    words = rec["idx"][i] + 1
    pj = (rec.get("jev") or [None] * len(rec["seq"]))[i] or 0.0
    return [0.0 if unk else 1.0, float(run), float(arun), float(len(seen)), float(bool(agrees)),
            pj, math.log1p(words)] + [1.0 if rec["category"] == c else 0.0 for c in CATS]


def load(judge):
    """-> {set: [(rec, human_word, X, y)]} over cached trajectories."""
    cache = json.load(open(measure.CACHE))
    out = {}
    for name in SETS:
        facts = os.path.expanduser(f"~/code/scibowl-org/stats/{name}/facts")
        qs, bz = data.load_sa_questions(facts), data.load_buzzes(facts)
        out[name] = []
        for key in qs:
            rec = cache.get(measure._ck(key, measure.STRIDE)) or cache.get(measure._ck(key, measure.STRIDE, 1))
            if not rec or "jev" not in rec:
                continue
            X = [feats(rec, i) for i in range(len(rec["seq"]))]
            y = [int(judge(rec["gold"], g)) for g, _a in rec["seq"]]
            out[name].append((rec, data.human_best_word(bz.get(key, []), rec["nwords"]), X, y))
    return out


def fit(rows):
    X = np.array([x for r in rows for x in r[2]])
    y = np.array([v for r in rows for v in r[3]])
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)


def judge_cached():
    import answerer
    cache = json.load(open(JUDGE_CACHE)) if os.path.exists(JUDGE_CACHE) else {}

    def judge(gold, pred):
        if not pred or pred.upper() == "UNKNOWN":
            return False
        k = f"{gold}\x1f{pred}"
        if k not in cache:
            cache[k] = bool(answerer.judge(pred, gold))
            json.dump(cache, open(JUDGE_CACHE, "w"))
        return cache[k]
    return judge


if __name__ == "__main__":
    judge = judge_cached()
    by_set = load(judge)
    for held in SETS:
        train = [r for s in SETS if s != held for r in by_set[s]]
        clf = fit(train)
        for rows in (train, by_set[held]):
            for rec, _h, X, _y in rows:
                rec["_p"] = list(clf.predict_proba(np.array(X))[:, 1])
        # tune T on the TRAINING tournaments, report on the held-out one
        tr_items = [(r[0], r[1]) for r in train]
        T = max((0.5, 0.6, 0.7, 0.8, 0.9), key=lambda t: measure.evaluate(tr_items, t, judge)[f"P>={t}"]["ev"])
        arms = measure.evaluate([(r[0], r[1]) for r in by_set[held]], T, judge)
        measure.print_report(f"HELD OUT {held} (n={len(by_set[held])}), gate trained on the other two, T={T}", arms)
