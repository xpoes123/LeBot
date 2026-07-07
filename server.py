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


def _prefer_qualified(calc_ans, verbose_ans):
    """The calculator does exact arithmetic but strips physical qualifiers — a current's
    direction, a vector's sign, "into the junction". A bare "2" then matches the wrong MC
    option ("2 out" vs "2 in"). If the reasoning answer contains the SAME number and adds
    a qualifier, prefer it (keeps calc's magnitude, restores the direction)."""
    if not verbose_ans or verbose_ans.upper() == "UNKNOWN":
        return calc_ans
    if calc_ans and re.fullmatch(r"[\d/.,\s\-]+", calc_ans):  # calc answer is a bare number
        cnum = re.sub(r"\D", "", calc_ans)
        if cnum and cnum in re.sub(r"\D", "", verbose_ans) and \
           len(verbose_ans.split()) > len(calc_ans.split()):
            return verbose_ans
    return calc_ans


def _value_steps(category, words, total, indices):
    """Anticipate the ANSWER VALUE over stem prefixes (blind to any options — the way a
    player hears the stem before the choices). Returns the per-prefix trajectory with the
    calibrated LR-buzzer P. Shared by short-answer and MC (the stem phase is identical)."""
    def one(widx):
        prefix = " ".join(words[:widx])
        with ThreadPoolExecutor(max_workers=2) as ex:
            fh = ex.submit(answerer.anticipate_best, prefix, category, 3)
            fv = ex.submit(answerer.anticipate_sa_verbose, prefix, category)
            haiku_guess, mode = fh.result()
            reasoning, sonnet_guess = fv.result()
        if mode == "seq":
            guess = haiku_guess
        elif mode == "calc":
            guess = _prefer_qualified(haiku_guess, sonnet_guess)
        elif sonnet_guess and sonnet_guess.upper() != "UNKNOWN":
            guess = sonnet_guess
        else:
            guess = haiku_guess
        agrees = _norm(haiku_guess) == _norm(guess) if guess and guess.upper() != "UNKNOWN" else None
        return {"widx": widx, "guess": guess, "mode": mode, "haiku_vote": haiku_guess,
                "agrees": agrees, "reasoning": _reason_text(mode, reasoning)}

    with ThreadPoolExecutor(max_workers=min(len(indices), 20)) as ex:
        raw = sorted(ex.map(one, indices), key=lambda r: r["widx"])

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
        cat_vec = [1.0 if category == c else 0.0 for c in CATS]
        feats = np.array([[frac, 0.0 if is_unk else 1.0, float(run), float(churn),
                           np.log1p(total), 1.0 if r["mode"] in ("calc", "seq") else 0.0] + cat_vec])
        p = float(_clf.predict_proba(feats)[0, 1])
        steps.append({**r, "phase": "stem", "stability_run": run, "churn": churn,
                      "frac": round(frac, 3), "p_buzz": round(p, 3), "buzzes": p >= BUZZ_T})
    return steps


def _mc_epilogue(req: FullReq, stem_steps, total):
    """After the stem, the moderator reads the four choices W→X→Y→Z in order. The bot has
    already anticipated a VALUE from the stem; now it maps that value onto a letter and
    buzzes the moment the matching choice is read. If it already committed (buzzed) during
    the stem, that stem buzz stands and the options just confirm the letter."""
    opts = req.options
    # For W/X/Y/Z MC, do NOT commit mid-stem: the choices are read AFTER the stem. Some
    # questions ("which of these is NOT X", "which best describes...") can only be answered
    # by REASONING OVER THE CHOICES, not by anticipating a value — so the authoritative
    # answer is guess() over the full stem + all four choices. The stem value-anticipation
    # is kept only to show what the bot was thinking as it listened.
    for s in stem_steps:
        s["buzzes"] = False
    stem_value = next((s["guess"] for s in reversed(stem_steps)
                       if s["guess"] and s["guess"].upper() != "UNKNOWN"), None)

    # 1. If the bot anticipated a concrete value (recall/compute), match it to a choice.
    matched, conf, votes = None, 0.0, []
    if stem_value:
        for k, text in enumerate(opts):
            if answerer.judge(stem_value, text):
                matched, conf = MC_LETTERS[k], 1.0
                break
    # 2. Otherwise the answer must be SELECTED by reasoning over the choices themselves
    #    ("which of these is NOT X", "which best describes...") -> guess over all options.
    if not matched:
        votes = answerer.guess(req.stem, opts, req.category, n=5)
        matched, conf = answerer.consensus_debiased(votes, req.stem, opts)
        if matched and conf < 0.5:
            matched = None

    option_steps = []
    for k, (L, text) in enumerate(zip(MC_LETTERS, opts)):
        is_match = (L == matched)
        option_steps.append({
            "widx": total + k + 1, "phase": "option", "letter": L, "option": text,
            "guess": f"{L} — {text}" if is_match else (stem_value or text),
            "matches": is_match, "buzzes": is_match, "mode": "option",
            "p_buzz": round(conf, 3) if is_match else 0.0,
            "stability_run": 0, "churn": 0, "frac": 1.0, "reasoning": "",
            "haiku_vote": "".join(v[0] if v and v[0] in MC_LETTERS else "." for v in votes),
            "agrees": None,
        })

    return {
        "mode": "mc", "options": opts, "answer_value": stem_value, "matched_letter": matched,
        "mc_conf": round(conf, 3), "committed_widx": None,
        "steps": stem_steps + option_steps, "total_words": total, "stem_words": total,
    }


@app.post("/analyze-full")
def analyze_full(req: FullReq):
    """Process a complete stem in parallel. Returns the full trajectory as JSON.

    For MC, the stem is anticipated as a VALUE (blind to the choices, as in a real match),
    then an option-reveal epilogue maps that value to a letter."""
    words = req.stem.split()
    if not words:
        return {"steps": [], "total_words": 0}
    total = len(words)
    indices = list(range(req.stride, total + 1, req.stride))
    if not indices or indices[-1] < total:
        indices.append(total)

    steps = _value_steps(req.category, words, total, indices)
    if req.options and len(req.options) == 4:
        return _mc_epilogue(req, steps, total)
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
