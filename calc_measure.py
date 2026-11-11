"""Early buzz on COMPUTATION questions: does the calculator, run mid-read, beat humans?

The live fast path is recall-only, so on a calc tossup it says UNKNOWN until the reader
stops. Here: from the first number on, every 3 words run answerer.solve(anticipate=True)
and buzz when the SAME computed value comes back on 2 consecutive ticks (exact answers:
stability is the confidence signal). Latency at reading speed, scored like measure.py.

  python calc_measure.py stanford-science-bowl johns-hopkins-invitational
"""
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import answerer
import data
import measure

CACHE = os.path.join(os.path.dirname(__file__), ".cache_calc.json")


def calc_questions(name):
    qs = data.load_sa_questions(os.path.expanduser(f"~/code/scibowl-org/stats/{name}/facts"))
    return {k: q for k, q in qs.items() if re.search(r"\d", q["stem"]) and re.match(r"^\s*[-−]?[\d.,/ ]+", q["gold"])
            and q["category"] in ("MATH", "PHYSICS", "CHEMISTRY", "EARTH_SPACE", "ENERGY")}


def ticks(stem):
    words = stem.split()
    first = next(i for i, w in enumerate(words) if re.search(r"\d", w))
    return sorted(set(range(first, len(words), 3)) | {len(words) - 1}), words


def run(key, q):
    idx, words = ticks(q["stem"])

    def one(j):
        t = time.time()
        try:
            v = answerer.solve(" ".join(words[: j + 1]), q["category"], anticipate=True)
        except Exception:
            v = None
        return v, round(time.time() - t, 2)
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(one, idx))
    return {"idx": idx, "nwords": len(words), "vals": [v for v, _ in out], "lat": [t for _, t in out]}


def calc_buzz(rec, need=2):
    """First tick where the same non-None value has come back `need` ticks running -> (word, value)."""
    prev, run_ = None, 0
    for v, j, lat in zip(rec["vals"], rec["idx"], rec["lat"]):
        run_ = run_ + 1 if (v is not None and v == prev) else (1 if v is not None else 0)
        prev = v
        if run_ >= need:
            return min(rec["nwords"], j + math.ceil(lat * measure.WPS)), v
    return None, None


if __name__ == "__main__":
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    judged = {}
    for name in sys.argv[1:]:
        qs = calc_questions(name)
        bz = data.load_buzzes(os.path.expanduser(f"~/code/scibowl-org/stats/{name}/facts"))
        todo = [(k, q) for k, q in qs.items() if "|".join(k) not in cache]
        for i, (k, q) in enumerate(todo):
            cache["|".join(k)] = run(k, q)
            json.dump(cache, open(CACHE, "w"))
            print(f"  [{i + 1}/{len(todo)}] gold={q['gold'][:20]!r} vals={cache['|'.join(k)]['vals'][-3:]}", flush=True)
        for need in (1, 2):
            n = buzzed = correct = ev = wins = 0
            fr, vs = [], []
            for k, q in qs.items():
                rec = cache["|".join(k)]
                h = data.human_best_word(bz.get(k, []), rec["nwords"])
                w, v = calc_buzz(rec, need)
                n += 1
                if w is None:
                    continue
                if (q["gold"], v) not in judged:
                    judged[(q["gold"], v)] = answerer.judge(v, q["gold"])
                ok = judged[(q["gold"], v)]
                pts = measure.score(w, ok, h, rec["nwords"])
                buzzed += 1; correct += ok; ev += pts; wins += pts == 4
                fr.append(w / rec["nwords"])
                if ok and h < rec["nwords"]:
                    vs.append((w - h) / rec["nwords"])
            vs.sort()
            print(f"{name[:26]:26} same value x{need}: n={n} buzz {buzzed / n:.0%} acc {correct / max(buzzed, 1):.0%} "
                  f"buzz@ {sum(fr) / max(buzzed, 1):.0%} vs fastest human {vs[len(vs) // 2] if vs else 0:+.0%} "
                  f"win {wins / n:.0%} ev/q {ev / n:+.2f}")
