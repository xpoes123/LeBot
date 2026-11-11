"""A/B: Jev (TypeSafe decision model, ~0.1 s) as the buzz gate vs the LR buzzer.

For every (question, prefix) in sa_labels.json, ask Jev "will committing to the
current guess be judged correct?" -> P(yes). Score on the SAME held-out split as
buzzer.py: calibration (ECE) + buzz policy (earliest prefix with P>=T). Also tries
Jev's P as an extra LR feature. Jev answers cache to .cache_jev.json.

Run: python jev_ab.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

import answerer  # noqa: F401  (loads .env)
import buzzer
import sa_data

URL = "https://api.typesafe.ai/v1/systemone"
CACHE = os.path.join(os.path.dirname(__file__), ".cache_jev.json")
CRITERIA = {
    "yes": "the guess is the correct answer to the full question (a judge would accept it)",
    "no": "the guess is wrong, too vague, or the clues so far don't pin the answer down yet",
}


def state(text, cat, traj):
    hist = "; ".join(f"after {w + 1} words: {g}" for w, g in traj)
    return (f"National Science Bowl short-answer tossup, category {cat}. The question is being read aloud "
            f"and has been cut off mid-stream. Heard so far:\n\"{text}\"\n\n"
            f"A player's guesses so far ({hist}). Their current guess: {traj[-1][1]}")


def ask(st):
    r = requests.post(URL, timeout=15, headers={"Authorization": f"Bearer {os.environ['JEV_API_KEY']}"},
                      json={"model": "jev-latest", "state": st, "questions": {"ok": {
                          "type": "choice", "question": "If the player buzzes now with their current guess, "
                          "will it be judged correct?", "criteria": CRITERIA}}})
    r.raise_for_status()
    a = r.json()["answers"]["ok"]
    if "probabilities" in a:
        return float(a["probabilities"]["yes"])
    return a["confidence"] if a["choice"] == "yes" else 1 - a["confidence"]


def jev_probs():
    """-> {(qid, widx): P(yes)}; UNKNOWN guesses aren't asked (P=0)."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    texts = {sa_data._id(q): q["question_text"] for q in json.load(open(sa_data.CLEAN))}
    jobs = []
    for rec in json.load(open(buzzer.LABELS)).values():
        words, traj = texts[rec["id"]].split(), []
        for w, pred, *_ in rec["preds"]:
            if pred.upper() == "UNKNOWN" or not buzzer._norm(pred):
                continue
            traj.append((w, pred))
            k = f"{rec['id']}:{w}"
            if k not in cache:
                jobs.append((k, state(" ".join(words[: w + 1]), rec["category"], list(traj))))
    print(f"{len(jobs)} Jev calls to make ({len(cache)} cached)")

    def work(j):
        try:
            return j[0], ask(j[1])
        except Exception as e:
            print("  fail", j[0], type(e).__name__)
            return j[0], None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (k, p) in enumerate(ex.map(work, jobs)):
            if p is not None:
                cache[k] = p
            if i % 200 == 199:
                json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))
    return cache


def table(name, test_by_q):
    print(f"\n=== {name} ===\n{'T':>5} {'buzz%':>6} {'acc':>5} {'buzz@':>6} {'ev/q':>6}")
    for T in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = buzzer.buzz_eval(test_by_q, T)
        print(f"{T:>5.1f} {m['buzz%']:>6.0%} {m['acc']:>5.0%} {m['buzz@']:>6.0%} {m['ev/q']:>6.2f}")


if __name__ == "__main__":
    cache = jev_probs()
    rows = []
    for rec in json.load(open(buzzer.LABELS)).values():
        for r, p in zip(buzzer.featurize(rec), rec["preds"]):
            r[0].append(cache.get(f"{rec['id']}:{p[0]}", 0.0))  # Jev P(yes) as the last column
            rows.append(r)
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    frac = np.array([r[2] for r in rows])
    groups = np.array([r[3] for r in rows])
    jev = X[:, -1]

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(X, y, groups))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr][:, :-1], y[tr])
    both = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
    arms = {"LR buzzer (current)": lr.predict_proba(X[te][:, :-1])[:, 1],
            "Jev alone (zero training)": jev[te],
            "LR + Jev feature": both.predict_proba(X[te])[:, 1]}
    for name, p in arms.items():
        print(f"\n##### {name}")
        buzzer.reliability(p, y[te])
        by_q = {}
        for pos, i in enumerate(te):
            by_q.setdefault(groups[i], []).append((frac[i], y[i], p[pos]))
        table(name, by_q)
