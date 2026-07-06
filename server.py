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

BUZZ_T = 0.75  # raised from 0.70 — old threshold was tuned on Haiku-primary data


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


_nsba4 = json.loads(Path("nsba4_questions.json").read_text())


@app.get("/")
def index():
    return HTMLResponse(Path("lebot.html").read_text())


@app.get("/questions/nsba4")
def nsba4_questions():
    return {"questions": _nsba4}


@app.post("/analyze")
def analyze(req: AnalyzeReq):
    # Run both in parallel — Sonnet verbose is already the latency bottleneck,
    # so using its answer costs nothing. Haiku votes give stability features + agreement.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_haiku = ex.submit(answerer.anticipate_best, req.prefix, req.category, 3)
        f_verbose = ex.submit(answerer.anticipate_sa_verbose, req.prefix, req.category)
        haiku_guess, mode = f_haiku.result()
        reasoning, sonnet_guess = f_verbose.result()

    if mode in ("calc", "seq"):
        guess = haiku_guess
    elif sonnet_guess and sonnet_guess.upper() != "UNKNOWN":
        guess = sonnet_guess
    else:
        guess = haiku_guess

    buzz = _buzz_features(req, guess, mode, haiku_guess)
    reasoning = _reason_text(mode, reasoning)
    return {"guess": guess, "reasoning": reasoning, **buzz}


def _reason_text(mode, reasoning):
    if mode == "recall":
        return reasoning
    if mode == "seq":
        return "Solved deterministically from a known canonical ordering (no LLM)."
    return "Computed via Python sandbox."


def _buzz_features(req: AnalyzeReq, guess: str, mode: str, haiku_guess: str):
    """Shared buzzer feature computation used by both endpoints."""
    words_heard = len(req.prefix.split())
    total = max(req.total_words, words_heard)
    frac = words_heard / total
    is_unk = not guess or guess.upper() == "UNKNOWN"
    is_calc = 1.0 if mode in ("calc", "seq") else 0.0
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
    haiku_no_guess = not haiku_guess or haiku_guess.upper() == "UNKNOWN"
    agrees = _norm(haiku_guess) == _norm(guess) if not is_unk and not haiku_no_guess else None
    # Both models have guesses but disagree → penalize confidence
    if agrees is False:
        p *= 0.6
    # Hard gate: Haiku UNKNOWN = stem incomplete, don't buzz regardless of P
    return dict(mode=mode, stability_run=run, churn=churn, frac=round(frac, 3),
                p_buzz=round(p, 3), buzzes=p >= BUZZ_T and not haiku_no_guess,
                haiku_vote=haiku_guess, agrees=agrees)


class FullReq(BaseModel):
    stem: str
    category: str = "BIOLOGY"
    options: list[str] | None = None  # 4 options W/X/Y/Z -> multiple-choice letter mode
    stride: int = 1  # per-word by default; parallel so wall-clock time is the same


MC_LETTERS = ["W", "X", "Y", "Z"]


def _analyze_mc(req: FullReq, words, total, indices):
    """MC letter mode: predict W/X/Y/Z per prefix via self-consistency, blended with the
    empirical letter prior. Shows the blind-prior baseline (no stem) so you can SEE
    anticipation — the letter diverging from the pure option-prior is the real signal."""
    opts = req.options
    # blind prior once (no stem): the pure option/category prior to beat. Same model
    # (Sonnet) as the stem-guess so the comparison isolates what the STEM adds.
    blind_letters = answerer.blind(opts, req.category, n=5, model=answerer.MODEL)
    blind_letter, blind_conf = answerer.consensus_debiased(blind_letters, "", opts)

    def one(widx):
        prefix = " ".join(words[:widx])
        letters = answerer.guess(prefix, opts, req.category, n=5)
        letter, conf = answerer.consensus_debiased(letters, prefix, opts)
        val = opts[MC_LETTERS.index(letter)] if letter in MC_LETTERS else None
        return {"widx": widx, "letter": letter, "conf": conf, "option": val,
                "votes": letters}

    with ThreadPoolExecutor(max_workers=min(len(indices), 20)) as ex:
        raw = sorted(ex.map(one, indices), key=lambda r: r["widx"])

    prev, run, seen, steps = None, 0, set(), []
    for r in raw:
        letter = r["letter"]
        if letter is None:
            run, prev = 0, None
        elif letter == prev:
            run += 1
        else:
            run, prev = 1, letter
        if letter:
            seen.add(letter)
        diverges = letter is not None and letter != blind_letter
        frac = r["widx"] / total
        # PRIOR-AWARE buzz. Self-consistency confidence is uncalibrated (Sonnet saturates
        # at 1.0 even when it's just echoing the option-prior), so confidence alone can't
        # gate. A DIVERGENT answer means the stem moved the model off its blind prior ->
        # trust it and buzz early. An answer that merely ECHOES the blind prior is
        # indistinguishable from a blind guess until late -> require most of the stem.
        if letter is None:
            buzzes = False
        elif diverges:
            buzzes = r["conf"] >= BUZZ_T and run >= 2
        else:
            buzzes = r["conf"] >= BUZZ_T and run >= 2 and frac >= 0.66
        guess = f"{letter} — {r['option']}" if letter else "UNKNOWN"
        steps.append({
            "widx": r["widx"], "guess": guess, "letter": letter, "option": r["option"],
            "mode": "mc", "p_buzz": round(r["conf"], 3), "stability_run": run,
            "churn": len(seen), "frac": round(r["widx"] / total, 3), "buzzes": buzzes,
            "diverges": diverges, "reasoning": "",
            "haiku_vote": "".join(v[0] if v and v[0] in MC_LETTERS else "." for v in r["votes"]),
            "agrees": None,
        })
    return {"steps": steps, "total_words": total, "mode": "mc",
            "blind_prior": blind_letter, "blind_conf": round(blind_conf, 3),
            "options": opts}


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

    if req.options and len(req.options) == 4:
        return _analyze_mc(req, words, total, indices)

    def one(widx):
        prefix = " ".join(words[:widx])
        with ThreadPoolExecutor(max_workers=2) as ex:
            fh = ex.submit(answerer.anticipate_best, prefix, req.category, 3)
            fv = ex.submit(answerer.anticipate_sa_verbose, prefix, req.category)
            haiku_guess, mode = fh.result()
            reasoning, sonnet_guess = fv.result()
        if mode in ("calc", "seq"):
            guess = haiku_guess
        elif sonnet_guess and sonnet_guess.upper() != "UNKNOWN":
            guess = sonnet_guess
        else:
            guess = haiku_guess
        agrees = _norm(haiku_guess) == _norm(guess) if guess and guess.upper() != "UNKNOWN" else None
        return {"widx": widx, "guess": guess, "mode": mode, "haiku_vote": haiku_guess,
                "agrees": agrees, "reasoning": _reason_text(mode, reasoning)}

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
                           np.log1p(total), 1.0 if r["mode"] in ("calc", "seq") else 0.0] + cat_vec])
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
                    final_guess = haiku_guess if mode in ("calc", "seq") else (sonnet_guess or haiku_guess)
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
