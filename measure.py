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
import os
from concurrent.futures import ThreadPoolExecutor

import data
from live import _confirm_run, _norm_answer, BUZZ_RUN

CACHE = os.path.join(os.path.dirname(__file__), ".cache_live.json")
STRIDE = 3   # anticipate every 3 words (+ the last) — mirrors live's THINK_STEP cadence


def _ck(key, stride):
    return "|".join(key) + f"|s{stride}"


def build(qs, limit, stride=STRIDE):
    """Per question: (guess, agrees) at each strided stem prefix, via the real fast confirm."""
    import answerer
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    items = list(qs.items())[:limit] if limit else list(qs.items())
    for i, (key, q) in enumerate(items):
        ck = _ck(key, stride)
        if ck in cache:
            continue
        words = q["stem"].split()
        idx = sorted(set(range(0, len(words), stride)) | {len(words) - 1})
        prefixes = [" ".join(words[: j + 1]) for j in idx]
        with ThreadPoolExecutor(max_workers=4) as ex:
            res = list(ex.map(
                lambda p: answerer.anticipate_fast_confirm(p, q["category"]), prefixes))
        seq = []
        for g, h, _mode in res:
            named = bool(g and g.upper() != "UNKNOWN" and h and h.upper() != "UNKNOWN")
            agrees = named and _norm_answer(g) == _norm_answer(h)
            seq.append([g, agrees])
        cache[ck] = {"idx": idx, "nwords": len(words), "seq": seq,
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


def run_eval(facts_dir, cache, stride=STRIDE):
    import answerer
    bz = data.load_buzzes(facts_dir)
    judged = {}   # (gold, pred) -> bool, dedup the judge calls

    def is_correct(gold, pred):
        if not pred or pred.upper() == "UNKNOWN":
            return False
        k = (gold, pred)
        if k not in judged:
            judged[k] = answerer.judge(pred, gold)
        return judged[k]

    arms = {"early": {"ev": 0, "correct": 0, "buzzed": 0, "wins": 0, "fracs": [], "earlier": 0},
            "end":   {"ev": 0, "correct": 0, "buzzed": 0, "wins": 0, "fracs": [], "earlier": 0}}
    ngraded = human_gettable = 0
    for key, q in _sa_items(facts_dir):
        ck = _ck(key, stride)
        if ck not in cache:
            continue
        rec = cache[ck]
        nwords = rec["nwords"]
        hword = data.human_best_word(bz.get(key, []), nwords)
        ngraded += 1
        human_gettable += hword != float("inf")

        # early arm
        bi, guess = gate_buzz(rec)
        if bi is not None:
            a = arms["early"]
            a["buzzed"] += 1
            a["fracs"].append(bi / nwords)
            a["earlier"] += bi <= hword
            ok = is_correct(rec["gold"], guess)
            a["correct"] += ok
            pts = score(bi, ok, hword, nwords)
            a["ev"] += pts
            a["wins"] += pts == 4

        # end arm (buzz at nwords with the final answer)
        eg = end_answer(rec)
        if eg is not None:
            a = arms["end"]
            a["buzzed"] += 1
            a["fracs"].append(1.0)
            a["earlier"] += nwords <= hword
            ok = is_correct(rec["gold"], eg)
            a["correct"] += ok
            pts = score(nwords, ok, hword, nwords)
            a["ev"] += pts
            a["wins"] += pts == 4

    return ngraded, human_gettable, arms


def _sa_items(facts_dir):
    return data.load_sa_questions(facts_dir).items()


def print_report(ngraded, human_gettable, arms):
    print(f"\nGraded {ngraded} short-answer tossups "
          f"({human_gettable} had a correct human buzz).\n")
    print(f"{'arm':>6} {'buzz%':>6} {'acc':>5} {'buzz@':>6} {'earlier%':>8} {'win%':>5} {'ev/q':>6}")
    for name in ("early", "end"):
        a = arms[name]
        n = a["buzzed"]
        print(f"{name:>6} {n/ngraded if ngraded else 0:>6.0%} "
              f"{a['correct']/n if n else 0:>5.0%} "
              f"{(sum(a['fracs'])/len(a['fracs'])) if a['fracs'] else float('nan'):>6.0%} "
              f"{a['earlier']/ngraded if ngraded else 0:>8.0%} "
              f"{a['wins']/ngraded if ngraded else 0:>5.0%} "
              f"{a['ev']/ngraded if ngraded else 0:>+6.2f}")
    print("\nbuzz%=how often it commits · acc=correct when it does · buzz@=avg fraction of the "
          "stem heard at buzz · earlier%=buzzed at/before the fastest human · win%=+4 rate · "
          "ev/q=expected points/question (mid-stem wrong=-4).")


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
    p.add_argument("--facts", default=os.path.expanduser(
        "~/code/scibowl-org/stats/stanford-science-bowl/facts"))
    p.add_argument("--limit", type=int, default=40, help="0 = all")
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--dry", action="store_true", help="self-check, no API")
    args = p.parse_args()

    if args.dry:
        demo()
        raise SystemExit

    qs = data.load_sa_questions(args.facts)
    cache = build(qs, args.limit, args.stride)
    print_report(*run_eval(args.facts, cache, args.stride))
