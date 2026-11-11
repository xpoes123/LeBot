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


SETUP_CRITERIA = {
    "yes": "the problem's setup is complete: every quantity it gives has been stated, so the most natural question "
           "about it could be computed now (even if the question hasn't said exactly what it asks yet)",
    "no": "the setup is still being described: more quantities or conditions are clearly coming",
}


TYPE_CRITERIA = {
    "compute": "a computational problem: the answer will be worked out from given numbers or a setup (a value, "
               "count, probability, energy...)",
    "recall": "a knowledge question: the answer is a name, term, law, organism, element, person or concept to recall",
}


def qtype_ready(text, cat, timeout=2.0):
    """One call, three reads of the question text alone: P(it's a calculation), P(a top player could buzz now and
    compute it in 2 s) and P(the setup has given all its numbers, ask maybe not yet stated — full send bets there).
    -> (p_compute, p_ready, p_setup), or (None, None, None) when Jev is unavailable. Separates early: springs 0.61
    by "spring constant", "ball dropped from 20 m" 0.98, wolves / "this element with atomic number 26" 0.00."""
    if not os.environ.get("JEV_API_KEY"):
        return None, None, None
    try:
        r = requests.post(URL, timeout=timeout, headers={"Authorization": f"Bearer {os.environ['JEV_API_KEY']}"},
                          json={"model": "jev-latest", "questions": {
                              "type": {"type": "choice", "criteria": TYPE_CRITERIA,
                                       "question": "What kind of question is this turning out to be?"},
                              "ready": {"type": "choice", "criteria": CALC_CRITERIA,
                                        "question": "Could a top player buzz now and compute the answer in 2 seconds?"},
                              "setup": {"type": "choice", "criteria": SETUP_CRITERIA,
                                        "question": "Has the setup given all of its numbers yet?"}},
                              "state": f"National Science Bowl short-answer tossup, category {cat}, being read aloud "
                                       f"(live transcript). Heard so far:\n\"{text}\""})
        r.raise_for_status()
        a = r.json()["answers"]
    except Exception:
        return None, None, None

    def p(x, label):
        if "probabilities" in x:
            return float(x["probabilities"][label])
        return x["confidence"] if x["choice"] == label else 1 - x["confidence"]
    return p(a["type"], "compute"), p(a["ready"], "yes"), p(a["setup"], "yes")
