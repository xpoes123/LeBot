"""Jev (TypeSafe decision model, ~0.1 s): P(buzzing now with the current guess is judged
correct). Used as an extra buzzer feature — see jev_ab.py for the A/B that earned it."""
import os

import requests

import answerer  # noqa: F401  (loads .env)

URL = "https://api.typesafe.ai/v1/systemone"
PROBS = os.path.join(os.path.dirname(__file__), "jev_probs.json")  # {"qid:widx": P} for sa_labels
CRITERIA = {
    "yes": "the guess is the correct answer to the full question (a judge would accept it)",
    "no": "the guess is wrong, too vague, or the clues so far don't pin the answer down yet",
}


def state(text, cat, traj):
    """traj = [(word_index or None, guess), ...] of the non-UNKNOWN guesses so far, current last."""
    hist = "; ".join(g if w is None else f"after {w + 1} words: {g}" for w, g in traj)
    return (f"National Science Bowl short-answer tossup, category {cat}. The question is being read aloud "
            f"and has been cut off mid-stream. Heard so far:\n\"{text}\"\n\n"
            f"A player's guesses so far ({hist}). Their current guess: {traj[-1][1]}")


def ask(st, timeout=15):
    r = requests.post(URL, timeout=timeout, headers={"Authorization": f"Bearer {os.environ['JEV_API_KEY']}"},
                      json={"model": "jev-latest", "state": st, "questions": {"ok": {
                          "type": "choice", "question": "If the player buzzes now with their current guess, "
                          "will it be judged correct?", "criteria": CRITERIA}}})
    r.raise_for_status()
    a = r.json()["answers"]["ok"]
    if "probabilities" in a:
        return float(a["probabilities"]["yes"])
    return a["confidence"] if a["choice"] == "yes" else 1 - a["confidence"]


def p_correct(text, cat, traj, timeout=2.0):
    """Live path: P(yes), 0.0 with no guess yet, None when Jev is unavailable (caller falls back)."""
    if not traj:
        return 0.0
    if not os.environ.get("JEV_API_KEY"):
        return None
    try:
        return ask(state(text, cat, traj), timeout)
    except Exception:
        return None


CALC_CRITERIA = {
    "yes": "everything needed to compute the answer has been given, and what the question asks for is either "
           "stated or obvious from the setup (e.g. given masses and forces -> the acceleration)",
    "no": "a needed quantity hasn't been given yet, or it's unclear what will be asked",
}


def calc_ready(text, cat, timeout=2.0):
    """Computational tossup: could a top player buzz now and compute it in the answer window? P(yes) or None.
    Question text only — no guess needed (the running guess is useless mid-calc). Springs example: ~0 until
    "minimum potential energy" (0.83), 0.93 at the end."""
    if not os.environ.get("JEV_API_KEY"):
        return None
    try:
        r = requests.post(URL, timeout=timeout, headers={"Authorization": f"Bearer {os.environ['JEV_API_KEY']}"},
                          json={"model": "jev-latest", "questions": {"ready": {
                              "type": "choice", "criteria": CALC_CRITERIA,
                              "question": "Could a top player buzz now and compute the answer in 2 seconds?"}},
                              "state": f"National Science Bowl computational short-answer tossup, category {cat}, "
                                       f"being read aloud (live transcript). Heard so far:\n\"{text}\""})
        r.raise_for_status()
        a = r.json()["answers"]["ready"]
    except Exception:
        return None
    if "probabilities" in a:
        return float(a["probabilities"]["yes"])
    return a["confidence"] if a["choice"] == "yes" else 1 - a["confidence"]
