"""Does the live SHORT-ANSWER early-buzz gate beat humans on WHEN to answer?

Replays the exact live gate over real SA tossups that carry human buzz logs. At strided
stem prefixes it calls the SAME fast confirm the live server uses
(answerer.anticipate_fast_confirm: Sonnet lean + independent Haiku vote), then commits
when the answer is cross-model-confirmed BUZZ_RUN ticks running (live._confirm_run) — and
compares the buzz word + correctness to the fastest CORRECT human on that question.

Two arms, same trajectories:
  early   — the new gate: buzz mid-read the moment it's confirmed.
  end     — the old behavior: buzz only after the whole stem (word = nwords).

  build (costs tokens, cached to .cache_live.json):  ./botvenv/bin/python measure.py --limit 40
  re-score from cache (free):                        ./botvenv/bin/python measure.py --limit 40
  self-check, no API:                                ./botvenv/bin/python measure.py --dry

Scoring (word axis; a wrong MID-STEM buzz is an interrupt, penalized like a real match):
  correct & buzz_word <= human_best_word -> +4   (bot first)
  correct & human earlier                ->  0
  wrong & buzzed before the stem ended   -> -4   (interrupt penalty)
  wrong at the end, or never buzzed      ->  0
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import data
from live import _confirm_run, _norm_answer, BUZZ_RUN

CACHE = os.path.join(os.path.dirname(__file__), ".cache_live.json")
STRIDE = 3   # anticipate every 3 words (+ the last) — mirrors live's THINK_STEP cadence


def _ck(key, stride, n=3):
    return "|".join(key) + f"|s{stride}" + ("" if n == 3 else f"|n{n}")


def build(qs, limit, stride=STRIDE, n=3):
    """Per question: (guess, agrees) at each strided stem prefix, via the real fast confirm."""
    import answerer
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    items = list(qs.items())[:limit] if limit else list(qs.items())
    for i, (key, q) in enumerate(items):
        ck = _ck(key, stride, n)
        if ck in cache or _ck(key, stride) in cache:   # one trajectory per question
            continue
        words = q["stem"].split()
        idx = sorted(set(range(0, len(words), stride)) | {len(words) - 1})
        prefixes = [" ".join(words[: j + 1]) for j in idx]
        def timed(p):
            t = time.time()
            return answerer.anticipate_fast_confirm(p, q["category"], n=n), time.time() - t
        with ThreadPoolExecutor(max_workers=4) as ex:
            out = list(ex.map(timed, prefixes))
        res, lat = [r for r, _ in out], [round(t, 2) for _, t in out]
        seq = []
        for g, h, _mode in res:
            named = bool(g and g.upper() != "UNKNOWN" and h and h.upper() != "UNKNOWN")
            agrees = named and _norm_answer(g) == _norm_answer(h)
            seq.append([g, agrees])
        cache[ck] = {"idx": idx, "nwords": len(words), "seq": seq, "lat": lat, "words": words,
                     "gold": q["gold"], "category": q["category"]}
        json.dump(cache, open(CACHE, "w"))   # checkpoint each (tokens are precious)
        print(f"  [{i+1}/{len(items)}] {q['category']:<11} gold={q['gold'][:28]!r}")
    return cache


def gate_buzz(rec):
    """Replay the live gate. -> (buzz_word_idx, committed_guess) or (None, None)."""
    norm, run = None, 0
    for (g, agrees), j in zip(rec["seq"], rec["idx"]):
        ng = _norm_answer(g) if (g and g.upper() != "UNKNOWN" and agrees) else ""
        norm, run = _confirm_run(norm, run, ng, agrees)
        if run >= BUZZ_RUN and j + 1 >= 3:      # words>=3, as in live
            return j, g
    return None, None


def end_answer(rec):
    """The old behavior: the answer after the whole stem is read (last named guess)."""
    for g, _a in reversed(rec["seq"]):
        if g and g.upper() != "UNKNOWN":
            return g
    return None


def score(bword, correct, hword, nwords):
    if bword is None:
        return 0
    if correct:
        return 4 if bword <= hword else 0
    return -4 if bword < nwords else 0        # wrong mid-stem = interrupt


WPS = 2.7        # moderator reading speed, words/s (NSB reads fast); latency -> words late
LAT_S = 1.5      # fallback per-tick latency for cache entries built before latency was logged
JEV_S = 0.15     # extra latency of the Jev call in the P-gate arms


def add_jev(cache):
    """Jev P(correct) per strided prefix, given the guess trajectory so far (cached in rec['jev'])."""
    import jev
    todo = []
    for ck, rec in cache.items():
        if "jev" in rec or "words" not in rec:
            continue
        traj, jobs = [], []
        for (g, _a), j in zip(rec["seq"], rec["idx"]):
            if g and g.upper() != "UNKNOWN":
                traj.append((j, g))
                jobs.append(jev.state(" ".join(rec["words"][: j + 1]), rec["category"], list(traj)))
            else:
                jobs.append(None)
        todo.append((rec, jobs))
    if todo:
        print(f"Jev pass over {len(todo)} questions")

    def one(st):
        try:
            return 0.0 if st is None else round(jev.ask(st), 3)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rec, jobs in todo:
            rec["jev"] = list(ex.map(one, jobs))
    json.dump(cache, open(CACHE, "w"))


def p_seq(rec):
    """LR+Jev P(buzz is correct) per prefix, features exactly as the server builds them live
    (total words unknown live -> live.est_total)."""
    import numpy as np
    import server
    from live import est_total
    out, prev, run, seen = [], None, 0, set()
    for i, ((g, _a), j) in enumerate(zip(rec["seq"], rec["idx"])):
        unk = not g or g.upper() == "UNKNOWN"
        ng = None if unk else _norm_answer(g)
        run = 0 if unk else (run + 1 if ng == prev else 1)
        prev = ng
        if not unk:
            seen.add(ng)
        total = est_total(j + 1)
        feats = np.array([[(j + 1) / total, 0.0 if unk else 1.0, float(run), float(len(seen)),
                           np.log1p(total), 0.0] + [1.0 if rec["category"] == c else 0.0 for c in server.CATS]])
        pj = (rec.get("jev") or [None] * len(rec["seq"]))[i]
        clf, f = (server._clf, feats) if pj is None else (server._clf_jev, np.append(feats, [[pj]], axis=1))
        out.append(float(clf.predict_proba(f)[0, 1]))
    return out


def first_fire(rec, fire, extra_s=0.0):
    """Earliest tick where fire(i) holds -> (buzz_word, guess), the buzz landing late by the
    tick's own latency at reading speed (capped at the end of the stem)."""
    lat = rec.get("lat") or [LAT_S] * len(rec["seq"])
    for i, ((g, _a), j) in enumerate(zip(rec["seq"], rec["idx"])):
        if j + 1 >= 3 and g and g.upper() != "UNKNOWN" and fire(i):
            return min(rec["nwords"], j + math.ceil((lat[i] + extra_s) * WPS)), g
    return None, None


def gates(rec, T):
    """Arm name -> (buzz_word, guess). `T` thresholds the LR+Jev P."""
    p = rec["_p"]
    agree = [a for _g, a in rec["seq"]]
    norm, run, runs = None, 0, []
    for g, a in rec["seq"]:
        norm, run = _confirm_run(norm, run, _norm_answer(g) if (g and a) else "", a)
        runs.append(run)
    eg = end_answer(rec)
    return {"agree x2 (live now)": first_fire(rec, lambda i: runs[i] >= BUZZ_RUN),
            "agree x1": first_fire(rec, lambda i: agree[i]),
            f"P>={T}": first_fire(rec, lambda i: p[i] >= T, JEV_S),
            f"P>={T} & agree": first_fire(rec, lambda i: p[i] >= T and agree[i], JEV_S),
            "end of stem": (rec["nwords"], eg) if eg else (None, None)}


def evaluate(items, T, is_correct):
    """items = [(rec, human_word)] -> {arm: metrics}."""
    arms = {}
    for rec, hword in items:
        n = rec["nwords"]
        for name, (bw, g) in gates(rec, T).items():
            a = arms.setdefault(name, {"n": 0, "buzzed": 0, "correct": 0, "ev": 0, "wins": 0,
                                       "fracs": [], "vs_h": []})
            a["n"] += 1
            if bw is None:
                continue
            ok = is_correct(rec["gold"], g)
            pts = score(bw, ok, hword, n)
            a["buzzed"] += 1
            a["correct"] += ok
            a["ev"] += pts
            a["wins"] += pts == 4
            a["fracs"].append(bw / n)
            if ok and hword < n:   # both rang in correctly during the stem: how far apart?
                a["vs_h"].append((bw - hword) / n)
    return arms


def print_report(title, arms):
    print(f"\n=== {title} ===")
    print(f"{'arm':>20} {'buzz%':>6} {'acc':>5} {'buzz@':>6} {'vs human':>9} {'win%':>5} {'ev/q':>6}")
    for name, a in arms.items():
        n, b = a["n"], a["buzzed"]
        vh = sorted(a["vs_h"])
        med = f"{vh[len(vh) // 2]:+.0%}" if vh else "-"
        print(f"{name:>20} {b / n:>6.0%} {a['correct'] / b if b else 0:>5.0%} "
              f"{sum(a['fracs']) / b if b else float('nan'):>6.0%} {med:>9} "
              f"{a['wins'] / n:>5.0%} {a['ev'] / n:>+6.2f}")


def demo():
    # gate: two confirmed ticks on the same answer -> buzz at that word index
    rec = {"idx": [0, 3, 6, 9], "nwords": 10,
           "seq": [["UNKNOWN", False], ["caspases", True], ["caspases", True], ["caspases", True]]}
    assert gate_buzz(rec) == (6, "caspases"), gate_buzz(rec)
    # never agrees -> never buzzes
    assert gate_buzz({"idx": [0, 3, 6], "nwords": 7,
                      "seq": [["x", False], ["x", False], ["x", False]]}) == (None, None)
    # flip-flop -> never reaches a run of 2
    assert gate_buzz({"idx": [0, 3, 6, 9], "nwords": 10,
                      "seq": [["a", True], ["b", True], ["a", True], ["b", True]]}) == (None, None)
    # end answer = last named guess
    assert end_answer(rec) == "caspases"
    assert end_answer({"idx": [0], "nwords": 3, "seq": [["UNKNOWN", False]]}) is None
    # scoring boundaries
    assert score(3, True, 5, 10) == 4       # correct, before human
    assert score(6, True, 5, 10) == 0       # correct, human earlier
    assert score(3, False, 5, 10) == -4     # wrong, mid-stem interrupt
    assert score(10, False, 5, 10) == 0     # wrong at the end, no penalty
    assert score(None, False, 5, 10) == 0   # silent
    assert score(3, True, float("inf"), 10) == 4   # nobody buzzed -> free points
    print("demo ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default="stanford-science-bowl,johns-hopkins-invitational",
                   help="scibowl-org stats datasets; the FIRST tunes T, the rest are held-out tests")
    p.add_argument("--limit", type=int, default=40, help="per set, 0 = all")
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--dry", action="store_true", help="self-check, no API")
    args = p.parse_args()

    if args.dry:
        demo()
        raise SystemExit

    import answerer
    judged = {}

    def is_correct(gold, pred):
        if not pred or pred.upper() == "UNKNOWN":
            return False
        if (gold, pred) not in judged:
            judged[(gold, pred)] = answerer.judge(pred, gold)
        return judged[(gold, pred)]

    per_set = {}
    for name in args.sets.split(","):
        facts = os.path.expanduser(f"~/code/scibowl-org/stats/{name}/facts")
        qs = data.load_sa_questions(facts)
        cache = build(qs, args.limit, args.stride)
        for key, q in qs.items():   # backfill stem words for entries cached before they were stored
            if _ck(key, args.stride) in cache:
                cache[_ck(key, args.stride)].setdefault("words", q["stem"].split())
        add_jev(cache)
        bz = data.load_buzzes(facts)
        items = []
        for key in list(qs)[: args.limit or None]:
            rec = cache.get(_ck(key, args.stride))
            if rec:
                rec["_p"] = p_seq(rec)
                items.append((rec, data.human_best_word(bz.get(key, []), rec["nwords"])))
        per_set[name] = items

    tune, *tests = list(per_set)
    best = max((0.4, 0.5, 0.6, 0.7, 0.8, 0.9), key=lambda T: max(
        a["ev"] for k, a in evaluate(per_set[tune], T, is_correct).items() if k.startswith("P")))
    print(f"\nT tuned on {tune}: {best}   (reading speed {WPS} words/s, measured per-call latency)")
    print_report(f"TUNE  {tune} (n={len(per_set[tune])})", evaluate(per_set[tune], best, is_correct))
    for t in tests:
        print_report(f"TEST  {t} (n={len(per_set[t])}, unseen)", evaluate(per_set[t], best, is_correct))
    print("\nbuzz@ = stem fraction heard when the buzz LANDS (latency included) · vs human = median "
          "(bot - fastest correct human) buzz point, as a fraction of the stem, on questions both "
          "got in-stem (negative = bot earlier) · win% = bot correct AND first · ev/q: mid-stem wrong = -4.")
