"""Show buzzes from a cached --antic run, focusing on where the bot beat the humans.

Usage: python show.py [facts_dir] [n] [stride] [S] [T]
Defaults to the held-out dasoni-2 test set with the locked config.
"""
import json
import os
import sys

import answerer
import data
import run

facts = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/code/scibowl-org/stats/dasoni-2/facts")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
S = int(sys.argv[4]) if len(sys.argv) > 4 else 1
T = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6

cache = json.load(open(run.CACHE_ANTIC))
qs = data.load_questions(facts)
bz = data.load_buzzes(facts)
LET = answerer.LETTERS


def human_desc(buzzes, nwords):
    """Where did the fastest CORRECT human buzz, in words / options?"""
    best = None
    for b in buzzes:
        if b["result"] != "correct":
            continue
        k = b["location_kind"]
        if k == "question":
            pos, txt = int(b["word_index"]), f"stem word {b['word_index']}"
        elif k == "option":
            pos, txt = nwords + int(b["option_index"]), f"option {LET[int(b['option_index'])]}"
        else:
            pos, txt = nwords + 10, "end of question"
        if best is None or pos < best[0]:
            best = (pos, txt)
    return best  # (sortable_pos, human-readable) or None


wins = []
for key, q in qs.items():
    ck = run._ck(key, n, stride)
    if ck not in cache or "blind" not in cache[ck]:
        continue
    rec = cache[ck]
    widx = rec.get("word_idx") or list(range(len(rec["per_word"])))
    bmodal = answerer.consensus(rec["blind"])[0]
    bi, letter = run.find_buzz(rec["per_word"], T, S, blind_modal=bmodal, stability_prior=S + 1)
    if bi is None or letter != q["answer"]:
        continue
    bot_word = widx[bi]
    hd = human_desc(bz.get(key, []), rec["nwords"])
    human_pos = hd[0] if hd else float("inf")
    if bot_word <= human_pos:  # a win
        wins.append((bot_word / rec["nwords"], key, q, bot_word, rec["nwords"], letter, hd, bmodal))

wins.sort()  # earliest (as fraction of stem) first
print(f"\n{len(wins)} winning buzzes on {os.path.basename(facts.rstrip('/').replace('/facts',''))} "
      f"(config S={S} T={T}, stride {stride}, n {n})\n" + "=" * 78)

for frac, key, q, bw, nw, letter, hd, bmodal in wins:
    words = q["stem"].split()
    marked = " ".join(words[: bw + 1]) + "  ⟨🔔 BUZZ⟩  " + " ".join(words[bw + 1:])
    human = hd[1] if hd else "nobody got it"
    flag = "  (diverged from prior!)" if letter != bmodal else "  (matched prior)"
    print(f"\n[{q['category']}]  answer = {letter}. {q['options'][LET.index(letter)][:60]}")
    print(f"  {marked}")
    print(f"  options: " + " | ".join(
        f"{LET[i]}={o[:22]}" for i, o in enumerate(q['options'])))
    print(f"  BOT buzzed at stem word {bw}/{nw} ({frac:.0%} in){flag}")
    print(f"  fastest correct human: {human}")
