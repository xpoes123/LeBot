"""The ML buzzer: a CALIBRATED answerability classifier.

features(partial stem so far) -> P(committing now is correct), learned from the
replayed (question, prefix) -> correct labels in sa_labels.json. This REPLACES the
LLM's self-reported confidence. We then evaluate, on held-out questions:
  - calibration (reliability curve + ECE) — does P mean what it says?
  - a buzz policy (buzz at earliest prefix with P>=T) vs a naive stability heuristic:
    lead time (how early), accuracy when buzzing, and EV with the neg penalty.

Run: python buzzer.py
"""
import json
import os
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

LABELS = os.path.join(os.path.dirname(__file__), "sa_labels.json")
CATS = ["BIOLOGY", "CHEMISTRY", "PHYSICS", "EARTH_SPACE", "MATH", "ENERGY", "OTHER"]
GAIN, NEG = 4, -4  # short-answer toss-up: +4 correct, neg penalty for wrong interrupt


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def featurize(rec):
    """-> list of (features, label, frac, qid) over the question's prefixes, in order."""
    rows = []
    prev = None
    run = 0
    seen = set()
    for p in rec["preds"]:
        widx, pred, correct = p[0], p[1], p[2]
        is_calc = 1.0 if (len(p) > 3 and p[3] == "calc") else 0.0
        np_ = _norm(pred)
        is_unk = (pred.upper() == "UNKNOWN") or not np_
        if not is_unk and np_ == prev:
            run += 1
        elif not is_unk:
            run = 1
        else:
            run = 0
        prev = None if is_unk else np_
        if not is_unk:
            seen.add(np_)
        frac = widx / max(1, rec["nwords"] - 1)
        cat = [1.0 if rec["category"] == c else 0.0 for c in CATS]
        feats = [
            frac,                      # how much of the question heard
            0.0 if is_unk else 1.0,    # does it have a guess at all
            float(run),                # consecutive prefixes holding this answer (stability)
            float(len(seen)),          # how many distinct answers it has floated (churn)
            np.log1p(rec["nwords"]),   # question length
            is_calc,                   # answered by the calculator (exact) vs eyeballed
        ] + cat
        rows.append((feats, int(correct), frac, rec["id"]))
    return rows


def load_rows():
    data = json.load(open(LABELS))
    rows = []
    for rec in data.values():
        rows += featurize(rec)
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    frac = np.array([r[2] for r in rows])
    groups = np.array([r[3] for r in rows])
    return X, y, frac, groups


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    print("  predicted -> empirical (n)")
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            emp = y[m].mean()
            ece += m.sum() / len(p) * abs(emp - p[m].mean())
            print(f"   {lo:.1f}-{hi:.1f}: {p[m].mean():.2f} -> {emp:.2f}  (n={m.sum()})")
    print(f"  ECE = {ece:.3f}  (lower = better calibrated)")


def buzz_eval(test_by_q, T):
    """Buzz at the earliest prefix with P>=T. -> dict of metrics over test questions."""
    ev = buzzed = correct = 0
    fracs = []
    for rows in test_by_q.values():
        rows = sorted(rows, key=lambda r: r[0])  # by frac
        fired = next(((f, c) for f, c, p in rows if p >= T), None)
        if fired is None:
            continue
        f, c = fired
        buzzed += 1
        fracs.append(f)
        correct += c
        ev += GAIN if c else NEG
    n = len(test_by_q)
    return {"T": T, "buzz%": buzzed / n, "acc": correct / buzzed if buzzed else 0,
            "buzz@": np.mean(fracs) if fracs else float("nan"), "ev/q": ev / n}


def naive_eval(test_by_q, stability):
    """Baseline: buzz when the answer has held stable for `stability` prefixes
    (the old LLM-confidence-free heuristic, no learned calibration)."""
    ev = buzzed = correct = 0
    fracs = []
    for rows in test_by_q.values():
        rows = sorted(rows, key=lambda r: r[1])  # by frac (index 1 here = frac)
        fired = next((r for r in rows if r[2] >= stability), None)
        if fired:
            buzzed += 1
            fracs.append(rows and fired[1])
            correct += fired[3]
            ev += GAIN if fired[3] else NEG
    n = len(test_by_q)
    return {"stab": stability, "buzz%": buzzed / n, "acc": correct / buzzed if buzzed else 0,
            "buzz@": np.mean(fracs) if fracs else float("nan"), "ev/q": ev / n}


if __name__ == "__main__":
    X, y, frac, groups = load_rows()
    print(f"{len(X)} prefix-examples from {len(set(groups))} questions  "
          f"| base correct rate {y.mean():.2f}")

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                  .split(X, y, groups))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
    p_te = clf.predict_proba(X[te])[:, 1]

    print("\n=== CALIBRATION (held-out questions) ===")
    reliability(p_te, y[te])

    # group held-out rows by question for the buzz policies
    test_by_q = {}
    for pos, i in enumerate(te):
        test_by_q.setdefault(groups[i], []).append((frac[i], y[i], p_te[pos]))

    print("\n=== ML BUZZER (calibrated P >= T) ===")
    print(f"{'T':>5} {'buzz%':>6} {'acc':>5} {'buzz@':>6} {'ev/q':>6}")
    for T in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = buzz_eval(test_by_q, T)
        print(f"{m['T']:>5.1f} {m['buzz%']:>6.0%} {m['acc']:>5.0%} {m['buzz@']:>6.0%} {m['ev/q']:>6.2f}")

    # naive baseline needs the stability feature (col index 2) per row
    naive_by_q = {}
    for i in te:
        naive_by_q.setdefault(groups[i], []).append((None, frac[i], X[i][2], y[i]))
    print("\n=== NAIVE STABILITY BASELINE (no learned calibration) ===")
    print(f"{'stab':>5} {'buzz%':>6} {'acc':>5} {'buzz@':>6} {'ev/q':>6}")
    for s in (1, 2, 3):
        m = naive_eval(naive_by_q, s)
        print(f"{m['stab']:>5} {m['buzz%']:>6.0%} {m['acc']:>5.0%} {m['buzz@']:>6.0%} {m['ev/q']:>6.2f}")

    print("\nfeature weights:", dict(zip(
        ["frac", "has_pred", "stability", "churn", "log_len", "is_calc"] + CATS,
        [round(w, 2) for w in clf.coef_[0]])))
