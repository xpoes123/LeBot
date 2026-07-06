"""Offline harness: can the brain + a threshold buzzer beat the human field?

Pipeline per MC tossup:
  1. reveal options one at a time; at each step ask the answerer (N samples) ->
     store (modal_letter, confidence). LLM runs ONCE per step; cached to disk.
  2. sweep threshold T and a latency handicap purely in Python (no extra tokens):
     bot buzzes at the first step where confidence>=T and the modal answer is
     stable vs the previous step.
  3. score EV vs the fastest correct human buzz.

Scoring (NSB toss-up, bot vs the field's fastest correct human):
  bot correct, buzzes at/<= human best step -> +4   (bot first)
  bot correct, but human was earlier         ->  0
  bot wrong AND interrupts (heard <4 options) -> -4   (interrupt penalty)
  bot wrong after hearing all options         ->  0   (no penalty at end)
  bot stays silent                            ->  0

Usage:
  python run.py --facts <dir> --limit 30 --n 5          # builds trajectories (costs tokens)
  python run.py --facts <dir>                           # re-sweep from cache, free
  python run.py --dry                                   # scoring self-check, no API
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import data

CACHE = os.path.join(os.path.dirname(__file__), ".cache_traj.json")
INTERRUPT_LIMIT = 3  # buzzing before option index 3 = question not fully read


def build_trajectories(facts_dir, limit, n):
    """For each question, store the N stem-first sampled letters."""
    import answerer  # imported here so --dry needs no API key

    qs = data.load_questions(facts_dir)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    items = list(qs.items())[:limit] if limit else list(qs.items())
    for i, (key, q) in enumerate(items):
        ck = "|".join(key) + f"|n{n}"
        if ck in cache:
            continue
        letters = answerer.predict_letters(q["stem"], q["options"], q["category"], n=n)
        cache[ck] = letters
        modal, conf = answerer.consensus(letters)
        print(f"  [{i+1}/{len(items)}] {q['category']:<11} ans={q['answer']} "
              f"-> {modal}@{conf:.1f} {letters}")
        json.dump(cache, open(CACHE, "w"))  # checkpoint after each (tokens are precious)
    return cache


def bot_buzz(letters, threshold, delay, stem=None, options=None, debias=False):
    """Stem-first buzz: if confident, buzz when the matched option is read.

    Buzz step = the position of the modal letter (W=0..Z=3), shifted by latency.
    -> (step, letter) or (None, None) if not confident enough.

    debias=True blends votes with the empirical MC letter prior (mc_prior); default off
    keeps the reproducible raw-consensus semantics used by the existing experiments.
    """
    import answerer
    if debias:
        modal, conf = answerer.consensus_debiased(letters, stem, options)
    else:
        modal, conf = answerer.consensus(letters)
    if modal is None or conf < threshold:
        return None, None
    return answerer.LETTERS.index(modal) + delay, modal


def score(bot_step, bot_letter, q, human_step):
    if bot_step is None:
        return 0
    correct = bot_letter == q["answer"]
    if correct:
        return 4 if bot_step <= human_step else 0
    return -4 if bot_step < INTERRUPT_LIMIT else 0


def evaluate(facts_dir, cache, n, thresholds, delays):
    qs = data.load_questions(facts_dir)
    bz = data.load_buzzes(facts_dir)
    rows = []
    for delay in delays:
        for T in thresholds:
            ev = correct = buzzed = wins = 0
            steps = []
            for key, q in qs.items():
                ck = "|".join(key) + f"|n{n}"
                if ck not in cache:
                    continue
                traj = cache[ck]
                hstep = data.human_best_step(bz.get(key, []))
                bstep, bletter = bot_buzz(traj, T, delay)
                pts = score(bstep, bletter, q, hstep)
                ev += pts
                if bstep is not None:
                    buzzed += 1
                    steps.append(bstep)
                    if bletter == q["answer"]:
                        correct += 1
                    if pts == 4:
                        wins += 1
            ngraded = sum(1 for key in qs if "|".join(key) + f"|n{n}" in cache)
            rows.append({
                "delay": delay, "T": T, "n": ngraded,
                "buzz%": buzzed / ngraded if ngraded else 0,
                "acc": correct / buzzed if buzzed else 0,
                "avg_step": sum(steps) / len(steps) if steps else float("nan"),
                "ev/q": ev / ngraded if ngraded else 0,
                "win%": wins / ngraded if ngraded else 0,
            })
    return rows


def print_table(rows):
    print(f"\n{'delay':>5} {'T':>5} {'n':>4} {'buzz%':>6} {'acc':>5} {'avg_step':>8} {'ev/q':>6} {'win%':>5}")
    for r in rows:
        print(f"{r['delay']:>5} {r['T']:>5.2f} {r['n']:>4} {r['buzz%']:>6.0%} "
              f"{r['acc']:>5.0%} {r['avg_step']:>8.2f} {r['ev/q']:>6.2f} {r['win%']:>5.0%}")


CACHE_DUAL = os.path.join(os.path.dirname(__file__), ".cache_dual.json")


def lock_point(guesses, stability):
    """First tick index where the gut answer has held stable for `stability` ticks.

    -> word index of the lock, or None if it never settles.
    """
    run = 0
    prev = None
    for i, g in enumerate(guesses):
        if g is not None and g == prev:
            run += 1
        else:
            run = 1 if g is not None else 0
        prev = g
        if run >= stability:
            return i
    return None


def build_dual(facts_dir, limit, n):
    """Per question: subconscious guess at every stem-word prefix + a thinking pass."""
    import answerer

    qs = data.load_questions(facts_dir)
    cache = json.load(open(CACHE_DUAL)) if os.path.exists(CACHE_DUAL) else {}
    items = list(qs.items())[:limit] if limit else list(qs.items())
    for i, (key, q) in enumerate(items):
        ck = "|".join(key) + f"|n{n}"
        if ck in cache:
            continue
        words = q["stem"].split()
        prefixes = [" ".join(words[: j + 1]) for j in range(len(words))]
        with ThreadPoolExecutor(max_workers=8) as ex:
            guesses = list(ex.map(
                lambda p: answerer.subconscious(p, q["category"]), prefixes))
        letters = answerer.predict_letters(q["stem"], q["options"], q["category"], n=n)
        cache[ck] = {"guesses": guesses, "letters": letters, "nwords": len(words)}
        modal, conf = answerer.consensus(letters)
        lk = lock_point(guesses, 2)
        print(f"  [{i+1}/{len(items)}] {q['category']:<11} ans={q['answer']} "
              f"think={modal}@{conf:.1f} lock@{lk}/{len(words)} gut={guesses}")
        json.dump(cache, open(CACHE_DUAL, "w"))
    return cache


def evaluate_dual(facts_dir, cache, n, stabilities, thresholds):
    import answerer

    qs = data.load_questions(facts_dir)
    bz = data.load_buzzes(facts_dir)
    rows = []
    for S in stabilities:
        for T in thresholds:
            ev = correct = buzzed = wins = 0
            locks = []
            ngraded = 0
            for key, q in qs.items():
                ck = "|".join(key) + f"|n{n}"
                if ck not in cache:
                    continue
                ngraded += 1
                rec = cache[ck]
                hstep = data.human_best_step(bz.get(key, []))
                lk = lock_point(rec["guesses"], S)
                modal, conf = answerer.consensus(rec["letters"])
                # buzz only if the gut locked AND thinking is confident.
                # thinking overlapped stem reading -> act at the matched option, delay 0.
                if lk is None or modal is None or conf < T:
                    continue
                buzzed += 1
                locks.append(lk / rec["nwords"])  # fraction of stem when we locked
                bstep = answerer.LETTERS.index(modal)
                pts = score(bstep, modal, q, hstep)
                ev += pts
                if modal == q["answer"]:
                    correct += 1
                if pts == 4:
                    wins += 1
            rows.append({
                "S": S, "T": T, "n": ngraded,
                "buzz%": buzzed / ngraded if ngraded else 0,
                "acc": correct / buzzed if buzzed else 0,
                "lock@": sum(locks) / len(locks) if locks else float("nan"),
                "ev/q": ev / ngraded if ngraded else 0,
                "win%": wins / ngraded if ngraded else 0,
            })
    return rows


def print_dual(rows):
    print(f"\n{'S':>2} {'T':>5} {'n':>4} {'buzz%':>6} {'acc':>5} {'lock@':>6} {'ev/q':>6} {'win%':>5}")
    for r in rows:
        print(f"{r['S']:>2} {r['T']:>5.2f} {r['n']:>4} {r['buzz%']:>6.0%} "
              f"{r['acc']:>5.0%} {r['lock@']:>6.0%} {r['ev/q']:>6.2f} {r['win%']:>5.0%}")


CACHE_ANTIC = os.path.join(os.path.dirname(__file__), ".cache_antic.json")


def find_buzz(per_word_letters, threshold, stability,
              blind_modal=None, stability_prior=None):
    """First stem word where anticipations agree (conf>=T) and hold for `stability`
    consecutive words. -> (word_index, letter) or (None, None).

    Prior-aware: if blind_modal is given, an answer that merely echoes the blind
    option-prior must hold for `stability_prior` words (>= stability) before we trust
    it — a divergent answer is real stem evidence and buzzes at `stability`.
    """
    import answerer
    s_prior = stability if stability_prior is None else stability_prior
    prev = None
    run = 0
    for j, letters in enumerate(per_word_letters):
        modal, conf = answerer.consensus(letters)
        active = modal is not None and conf >= threshold
        if active and modal == prev:
            run += 1
        elif active:
            run = 1
        else:
            run, prev = 0, None
            continue
        prev = modal
        need = s_prior if (blind_modal is not None and modal == blind_modal) else stability
        if run >= need:
            return j, modal
    return None, None


def _ck(key, n, stride):
    import answerer
    return "|".join(key) + f"|n{n}|s{stride}|{answerer.FAST}"


def build_antic(facts_dir, limit, n, stride=1):
    """Per question: anticipate the answer at strided stem-word prefixes.

    stride>1 anticipates every `stride` words (plus always the full stem) to cut
    cost — mirrors live (you won't run an LLM every single word anyway). Stores the
    actual word index of each computed point so buzz timing stays exact.
    """
    import answerer

    qs = data.load_questions(facts_dir)
    cache = json.load(open(CACHE_ANTIC)) if os.path.exists(CACHE_ANTIC) else {}
    items = list(qs.items())[:limit] if limit else list(qs.items())
    for i, (key, q) in enumerate(items):
        ck = _ck(key, n, stride)
        rec = cache.get(ck)
        if rec is not None and "blind" in rec:
            continue
        words = q["stem"].split()
        if rec is None:
            idx = sorted(set(list(range(0, len(words), stride)) + [len(words) - 1]))
            prefixes = [" ".join(words[: j + 1]) for j in idx]
            with ThreadPoolExecutor(max_workers=8) as ex:
                per_word = list(ex.map(
                    lambda p: answerer.anticipate(p, q["options"], q["category"], n),
                    prefixes))
            rec = {"per_word": per_word, "word_idx": idx, "nwords": len(words)}
        rec["blind"] = answerer.blind(q["options"], q["category"], n)  # control arm
        cache[ck] = rec
        bi, letter = find_buzz(rec["per_word"], 0.8, 2)
        loc = f"{rec['word_idx'][bi]}/{len(words)}" if bi is not None else f"-/{len(words)}"
        bmodal, _ = answerer.consensus(rec["blind"])
        print(f"  [{i+1}/{len(items)}] {q['category']:<11} ans={q['answer']} "
              f"antic-buzz {letter} @ word {loc}  blind-prior={bmodal}")
        json.dump(cache, open(CACHE_ANTIC, "w"))
    return cache


def _score_set(items, cache, n, bz, S, T, stride=1, prior_aware=True):
    """Score the anticipation buzzer over a set of (key, q) items. When prior_aware,
    answers that echo the blind prior need one extra confirming word to buzz."""
    import answerer
    ev = correct = buzzed = wins = ngraded = 0
    fracs = []
    for key, q in items:
        ck = _ck(key, n, stride)
        if ck not in cache:
            continue
        ngraded += 1
        rec = cache[ck]
        widx = rec.get("word_idx") or list(range(len(rec["per_word"])))
        hword = data.human_best_word(bz.get(key, []), rec["nwords"])
        bmodal = answerer.consensus(rec["blind"])[0] if prior_aware and "blind" in rec else None
        bi, letter = find_buzz(rec["per_word"], T, S,
                               blind_modal=bmodal, stability_prior=S + 1)
        if bi is None:
            continue
        j = widx[bi]  # actual stem word index of the buzz
        buzzed += 1
        fracs.append(j / rec["nwords"])
        ok = letter == q["answer"]
        correct += ok
        pts = (4 if j <= hword else 0) if ok else -4  # mid-stem buzz: wrong = interrupt
        wins += pts == 4
        ev += pts
    return {
        "S": S, "T": T, "n": ngraded,
        "buzz%": buzzed / ngraded if ngraded else 0,
        "acc": correct / buzzed if buzzed else 0,
        "buzz@": sum(fracs) / len(fracs) if fracs else float("nan"),
        "ev/q": ev / ngraded if ngraded else 0,
        "win%": wins / ngraded if ngraded else 0,
    }


def _bucket(facts_dir, cache, n, stride):
    """Split a dataset's questions into all / prior-wrong, and score the blind control.
    A question is included only if its anticipation cache exists."""
    import answerer
    qs = data.load_questions(facts_dir)
    all_items, prior_wrong = [], []
    blind_ev = blind_n = blind_hits = 0
    for key, q in qs.items():
        ck = _ck(key, n, stride)
        if ck not in cache or "blind" not in cache[ck]:
            continue
        all_items.append((key, q))
        bmodal, _ = answerer.consensus(cache[ck]["blind"])
        prior_correct = bmodal == q["answer"]
        if not prior_correct:
            prior_wrong.append((key, q))
        if bmodal is not None:  # control: buzz the prior at word 0 (always an interrupt)
            blind_n += 1
            blind_hits += prior_correct
            blind_ev += 4 if prior_correct else -4
    control = {
        "n": blind_n,
        "acc": blind_hits / blind_n if blind_n else 0,
        "ev/q": blind_ev / blind_n if blind_n else 0,
        "prior_wrong": len(prior_wrong),
    }
    return all_items, prior_wrong, control


def evaluate_antic(facts_dir, cache, n, thresholds, stabilities, stride=1):
    """De-confounded: overall, the prior-WRONG subset (honest test), blind control."""
    bz = data.load_buzzes(facts_dir)
    all_items, prior_wrong, control = _bucket(facts_dir, cache, n, stride)
    overall = [_score_set(all_items, cache, n, bz, S, T, stride)
               for S in stabilities for T in thresholds]
    subset = [_score_set(prior_wrong, cache, n, bz, S, T, stride)
              for S in stabilities for T in thresholds]
    return overall, subset, control


def train_test(train_facts, test_facts, n, stride, limit):
    """Train = pick the buzzer config (S,T) that maximizes OVERALL ev/q on the train
    set; then report that single locked config on the held-out test set."""
    cache = build_antic(train_facts, limit, n, stride)
    cache = build_antic(test_facts, limit, n, stride)

    tr_all, tr_pw, tr_ctrl = _bucket(train_facts, cache, n, stride)
    bz_tr = data.load_buzzes(train_facts)
    configs = [(S, T) for S in (1, 2, 3) for T in (0.6, 0.8, 1.0)]
    scored = [((S, T), _score_set(tr_all, cache, n, bz_tr, S, T, stride))
              for (S, T) in configs]
    (bS, bT), tr_best = max(scored, key=lambda x: x[1]["ev/q"])

    te_all, te_pw, te_ctrl = _bucket(test_facts, cache, n, stride)
    bz_te = data.load_buzzes(test_facts)
    te_overall = _score_set(te_all, cache, n, bz_te, bS, bT, stride)
    te_subset = _score_set(te_pw, cache, n, bz_te, bS, bT, stride)

    print(f"\n===== TRAIN: {os.path.basename(train_facts.rstrip('/').replace('/facts',''))} "
          f"(n={tr_ctrl['n']}, prior wrong {tr_ctrl['prior_wrong']}) =====")
    _print_rows("config sweep (OVERALL ev/q is the selection metric):",
                [s for _, s in scored])
    print(f"\n>>> LOCKED config from train: S={bS} T={bT}  "
          f"(train overall ev/q={tr_best['ev/q']:+.2f})")
    print(f"\n===== TEST: {os.path.basename(test_facts.rstrip('/').replace('/facts',''))} "
          f"(n={te_ctrl['n']}, prior wrong {te_ctrl['prior_wrong']}, "
          f"blind control ev/q={te_ctrl['ev/q']:+.2f}) =====")
    _print_rows("HELD-OUT overall:", [te_overall])
    _print_rows("HELD-OUT prior-wrong subset (honest):", [te_subset])


def _print_rows(title, rows):
    print(f"\n{title}")
    print(f"{'S':>2} {'T':>5} {'n':>4} {'buzz%':>6} {'acc':>5} {'buzz@':>6} {'ev/q':>6} {'win%':>5}")
    for r in rows:
        print(f"{r['S']:>2} {r['T']:>5.2f} {r['n']:>4} {r['buzz%']:>6.0%} "
              f"{r['acc']:>5.0%} {r['buzz@']:>6.0%} {r['ev/q']:>6.2f} {r['win%']:>5.0%}")


def print_antic(result):
    overall, subset, control = result
    print(f"\nBLIND CONTROL (buzz option-prior at word 0): "
          f"acc={control['acc']:.0%} ev/q={control['ev/q']:+.2f} over n={control['n']}  "
          f"| prior was WRONG on {control['prior_wrong']}/{control['n']} questions")
    _print_rows("OVERALL (all questions — still prior-contaminated):", overall)
    _print_rows("PRIOR-WRONG SUBSET (honest anticipation: stem must do the work):", subset)


def demo():
    """Scoring self-check, no API. Asserts the EV logic at the boundaries."""
    q = {"answer": "X"}
    # bot correct and first -> +4
    assert score(1, "X", q, human_step=2) == 4
    # bot correct but human earlier -> 0
    assert score(3, "X", q, human_step=1) == 0
    # bot correct, ties human step -> bot wins +4
    assert score(2, "X", q, human_step=2) == 4
    # bot wrong, interrupts (step<3) -> -4
    assert score(1, "W", q, human_step=2) == -4
    # bot wrong but heard all options (step>=3) -> 0
    assert score(3, "W", q, human_step=2) == 0
    # silent -> 0
    assert score(None, None, q, human_step=2) == 0
    # nobody got it (human inf), bot correct -> +4
    assert score(2, "X", q, human_step=float("inf")) == 4

    # buzzer: stem-first, buzz at the matched option's position, latency-shifted
    letters = ["X", "X", "W", "NONE", "X"]  # modal X, conf 3/5=0.6, X is option index 1
    assert bot_buzz(letters, 0.5, 0) == (1, "X")
    assert bot_buzz(letters, 0.5, 1) == (2, "X")  # +1 latency
    assert bot_buzz(letters, 0.7, 0) == (None, None)  # conf .6 < .7
    assert bot_buzz(["NONE"] * 5, 0.1, 0) == (None, None)  # no commitment

    # dual-process lock: gut answer must hold stable for `stability` ticks
    g = [None, "foo", "anisotropy", "anisotropy", "anisotropy"]
    assert lock_point(g, 2) == 3   # stable from idx2; 2-in-a-row satisfied at idx3
    assert lock_point(g, 3) == 4
    assert lock_point(g, 4) is None  # never holds 4 in a row
    assert lock_point([None, "a", "b", "a"], 2) is None  # flip-flop never locks

    # anticipation buzz: fire when N anticipations agree and hold `stability` words
    pw = [["NONE"] * 4, ["W", "X", "NONE", "X"], ["X", "X", "X", "NONE"],
          ["X", "X", "X", "X"]]  # word3 conf .75 X, word4 conf 1.0 X -> stable@word3
    assert find_buzz(pw, 0.7, 2) == (3, "X")
    assert find_buzz(pw, 0.7, 1) == (2, "X")  # word2 alone clears .7 (.75)
    assert find_buzz(pw, 0.9, 1) == (3, "X")  # only word3 hits .9
    assert find_buzz([["NONE"] * 4] * 3, 0.5, 1) == (None, None)
    print("demo ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--facts", default=os.path.expanduser(
        "~/code/scibowl-org/stats/stanford-science-bowl/facts"))
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--n", type=int, default=5, help="self-consistency samples")
    p.add_argument("--dry", action="store_true", help="scoring self-check only")
    p.add_argument("--dual", action="store_true", help="System 1/2 dual-process")
    p.add_argument("--antic", action="store_true", help="question-anticipation buzzer")
    p.add_argument("--stride", type=int, default=1, help="anticipate every N words (cost)")
    p.add_argument("--train", help="facts dir to TUNE the buzzer config on")
    p.add_argument("--test", help="facts dir to evaluate the locked config on (held out)")
    args = p.parse_args()

    if args.dry:
        demo()
        raise SystemExit

    if args.train and args.test:
        train_test(args.train, args.test, args.n, args.stride, args.limit)
        raise SystemExit

    if args.dual:
        cache = build_dual(args.facts, args.limit, args.n)
        print_dual(evaluate_dual(args.facts, cache, args.n,
                                 stabilities=[2, 3], thresholds=[0.6, 1.0]))
        raise SystemExit

    if args.antic:
        cache = build_antic(args.facts, args.limit, args.n, args.stride)
        print_antic(evaluate_antic(args.facts, cache, args.n,
                                   thresholds=[0.6, 0.8, 1.0], stabilities=[1, 2],
                                   stride=args.stride))
        raise SystemExit

    cache = build_trajectories(args.facts, args.limit, args.n)
    thresholds = [0.4, 0.6, 0.8, 1.0]
    delays = [0, 1, 2]
    print_table(evaluate(args.facts, cache, args.n, thresholds, delays))
