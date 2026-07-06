"""LeBot interactive demo server.
POST /analyze  {prefix, category, total_words, history:[{guess,mode}]}
            -> {guess, mode, stability_run, churn, frac, p_buzz, buzzes, reasoning}
GET  /       -> lebot.html
"""
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

import answerer
from buzzer import load_rows, CATS

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Train on all data at startup (demo — no held-out split)
_X, _y, *_ = load_rows()
_clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(_X, _y)

BUZZ_T = 0.7  # positive-EV threshold from eval


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


class _Step(BaseModel):
    guess: str
    mode: str


class AnalyzeReq(BaseModel):
    prefix: str
    category: str = "BIOLOGY"
    total_words: int = 50
    history: list[_Step] = []


@app.get("/")
def index():
    return HTMLResponse(Path("lebot.html").read_text())


@app.post("/analyze")
def analyze(req: AnalyzeReq):
    # run main guess + verbose reasoning in parallel (no added latency)
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_guess = ex.submit(answerer.anticipate_best, req.prefix, req.category, 3)
        f_verbose = ex.submit(answerer.anticipate_sa_verbose, req.prefix, req.category)
        guess, mode = f_guess.result()
        reasoning, _ = f_verbose.result()

    words_heard = len(req.prefix.split())
    total = max(req.total_words, words_heard)
    frac = words_heard / total

    is_unk = not guess or guess.upper() == "UNKNOWN"
    is_calc = 1.0 if mode == "calc" else 0.0

    all_guesses = [h.guess for h in req.history] + [guess]

    run = 0
    if not is_unk:
        for g in reversed(all_guesses):
            if g and g.upper() != "UNKNOWN" and _norm(g) == _norm(guess):
                run += 1
            else:
                break

    seen = {_norm(g) for g in all_guesses if g and g.upper() != "UNKNOWN"}
    churn = len(seen)

    cat_vec = [1.0 if req.category == c else 0.0 for c in CATS]
    feats = np.array([[
        frac, 0.0 if is_unk else 1.0, float(run), float(churn),
        np.log1p(total), is_calc,
    ] + cat_vec])

    p = float(_clf.predict_proba(feats)[0, 1])

    return {
        "guess": guess,
        "mode": mode,
        "stability_run": run,
        "churn": churn,
        "frac": round(frac, 3),
        "p_buzz": round(p, 3),
        "buzzes": p >= BUZZ_T,
        "reasoning": reasoning if mode == "recall" else "Computed via Python sandbox.",
    }
