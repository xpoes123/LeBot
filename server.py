"""LeBot interactive demo server.
POST /analyze  {prefix, category, total_words, history:[{guess,mode}]}
            -> {guess, mode, stability_run, churn, frac, p_buzz, buzzes, reasoning}
GET  /       -> lebot.html
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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
    # Run both in parallel — Sonnet verbose is already the latency bottleneck,
    # so using its answer costs nothing. Haiku votes give stability features + agreement.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_haiku = ex.submit(answerer.anticipate_best, req.prefix, req.category, 3)
        f_verbose = ex.submit(answerer.anticipate_sa_verbose, req.prefix, req.category)
        haiku_guess, mode = f_haiku.result()
        reasoning, sonnet_guess = f_verbose.result()

    if mode == "calc":
        guess = haiku_guess
    elif sonnet_guess and sonnet_guess.upper() != "UNKNOWN":
        guess = sonnet_guess
    else:
        guess = haiku_guess

    buzz = _buzz_features(req, guess, mode, haiku_guess)
    reasoning = reasoning if mode == "recall" else "Computed via Python sandbox."
    return {"guess": guess, "reasoning": reasoning, **buzz}


def _buzz_features(req: AnalyzeReq, guess: str, mode: str, haiku_guess: str):
    """Shared buzzer feature computation used by both endpoints."""
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
    feats = np.array([[frac, 0.0 if is_unk else 1.0, float(run), float(churn),
                       np.log1p(total), is_calc] + cat_vec])
    p = float(_clf.predict_proba(feats)[0, 1])
    agrees = _norm(haiku_guess) == _norm(guess) if not is_unk and haiku_guess else None
    # Hard gate: if Haiku has no guess at all, there isn't enough info to buzz.
    # Sonnet can still return a terse answer while its own reasoning says "I'll wait" —
    # Haiku UNKNOWN is the reliable signal that the stem is incomplete.
    haiku_no_guess = not haiku_guess or haiku_guess.upper() == "UNKNOWN"
    return dict(mode=mode, stability_run=run, churn=churn, frac=round(frac, 3),
                p_buzz=round(p, 3), buzzes=p >= BUZZ_T and not haiku_no_guess,
                haiku_vote=haiku_guess, agrees=agrees)


class FullReq(BaseModel):
    stem: str
    category: str = "BIOLOGY"
    stride: int = 1  # per-word by default; parallel so wall-clock time is the same


@app.post("/analyze-full")
def analyze_full(req: FullReq):
    """Process a complete stem in parallel. Returns the full trajectory as JSON."""
    words = req.stem.split()
    if not words:
        return {"steps": [], "total_words": 0}
    total = len(words)
    indices = list(range(req.stride, total + 1, req.stride))
    if not indices or indices[-1] < total:
        indices.append(total)

    def one(widx):
        prefix = " ".join(words[:widx])
        with ThreadPoolExecutor(max_workers=2) as ex:
            fh = ex.submit(answerer.anticipate_best, prefix, req.category, 3)
            fv = ex.submit(answerer.anticipate_sa_verbose, prefix, req.category)
            haiku_guess, mode = fh.result()
            reasoning, sonnet_guess = fv.result()
        if mode == "calc":
            guess = haiku_guess
        elif sonnet_guess and sonnet_guess.upper() != "UNKNOWN":
            guess = sonnet_guess
        else:
            guess = haiku_guess
        agrees = _norm(haiku_guess) == _norm(guess) if guess and guess.upper() != "UNKNOWN" else None
        return {"widx": widx, "guess": guess, "mode": mode, "haiku_vote": haiku_guess,
                "agrees": agrees, "reasoning": reasoning if mode == "recall" else "Computed via Python sandbox."}

    with ThreadPoolExecutor(max_workers=min(len(indices), 20)) as ex:
        raw = sorted(ex.map(one, indices), key=lambda r: r["widx"])

    # Stability/churn/P computed post-hoc over the ordered sequence
    prev_n, run, seen, steps = None, 0, set(), []
    for r in raw:
        guess = r["guess"]
        is_unk = not guess or guess.upper() == "UNKNOWN"
        ng = _norm(guess)
        if is_unk:
            run, prev_n = 0, None
        elif ng == prev_n:
            run += 1
        else:
            run, prev_n = 1, ng
        if not is_unk:
            seen.add(ng)
        churn = len(seen)
        frac = r["widx"] / total
        cat_vec = [1.0 if req.category == c else 0.0 for c in CATS]
        feats = np.array([[frac, 0.0 if is_unk else 1.0, float(run), float(churn),
                           np.log1p(total), 1.0 if r["mode"] == "calc" else 0.0] + cat_vec])
        p = float(_clf.predict_proba(feats)[0, 1])
        steps.append({**r, "stability_run": run, "churn": churn,
                      "frac": round(frac, 3), "p_buzz": round(p, 3), "buzzes": p >= BUZZ_T})

    return {"steps": steps, "total_words": total}


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/analyze-stream")
def analyze_stream(req: AnalyzeReq):
    """Streaming version: yields 'guess' at ~0.4s, 'buzz' shortly after, 'reasoning' last."""
    def generate():
        with ThreadPoolExecutor(max_workers=1) as ex:
            f_haiku = ex.submit(answerer.anticipate_best, req.prefix, req.category, 3)

            sonnet_guess = None
            for event, value in answerer.stream_answer(req.prefix, req.category):
                if event == "answer":
                    sonnet_guess = value
                    yield _sse({"type": "guess", "guess": value})

                    # Haiku is usually done by now — brief wait at most
                    haiku_guess, mode = f_haiku.result()
                    # Calc answer is exact: override Sonnet for those
                    final_guess = haiku_guess if mode == "calc" else (sonnet_guess or haiku_guess)
                    if final_guess != sonnet_guess:
                        yield _sse({"type": "guess", "guess": final_guess})

                    buzz = _buzz_features(req, final_guess, mode, haiku_guess)
                    yield _sse({"type": "buzz", **buzz})

                elif event == "reasoning":
                    reasoning = value if (sonnet_guess or "").upper() != "UNKNOWN" else ""
                    yield _sse({"type": "reasoning", "reasoning": reasoning})

        yield _sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
