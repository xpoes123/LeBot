"""Walkthrough page: for a few diverse questions, show the FULL per-prefix process —
the model's reasoning, the terse answer it would commit, whether it's right, and the
trained buzzer's calibrated P(correct) with the buzz point. So you can see how it works.

Run: python explain.py  -> explain.html
"""
import hashlib
import html
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import answerer
import buzzer

CLEAN = os.path.join(os.path.dirname(__file__), "packets", "questions_clean.json")
STRIDE = 3
T = 0.8


def stem_map():
    m = {}
    for q in json.load(open(CLEAN)):
        i = hashlib.md5((q["tournament"] + q["question_text"]).encode()).hexdigest()[:10]
        m[i] = q
    return m


def first_correct_frac(rec):
    for p in rec["preds"]:
        if p[2]:
            return p[0] / max(1, rec["nwords"] - 1)
    return None


def pick(data):
    """Choose ~5 diverse questions: early-correct, late-correct, never-correct,
    a computational MATH one, and a high-churn (changes its mind) one."""
    recs = list(data.values())
    chosen, used = [], set()

    def take(pred):
        for r in sorted(recs, key=lambda r: -len(r["preds"])):
            if r["id"] in used:
                continue
            if pred(r):
                used.add(r["id"])
                chosen.append(r)
                return

    fc = first_correct_frac
    take(lambda r: r["category"] != "MATH" and fc(r) is not None and fc(r) < 0.4)
    take(lambda r: fc(r) is not None and fc(r) > 0.7)
    take(lambda r: fc(r) is None and 12 <= r["nwords"] <= 45)
    take(lambda r: r["category"] == "MATH")
    take(lambda r: len({p[1] for p in r["preds"] if p[1] != "UNKNOWN"}) >= 3 and fc(r) is not None)
    return chosen


def build():
    data = json.load(open(buzzer.LABELS))
    stems = stem_map()
    chosen = pick(data)
    sel_ids = {r["id"] for r in chosen}

    # train the buzzer on everything EXCEPT the chosen questions (honest P)
    X, y, frac, groups = buzzer.load_rows()
    mask = np.array([g not in sel_ids for g in groups])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[mask], y[mask])

    cards = []
    for rec in chosen:
        q = stems.get(rec["id"])
        if not q:
            continue
        words = q["question_text"].split()
        idx = sorted(set(list(range(0, len(words), STRIDE)) + [len(words) - 1]))
        gold = rec["gold"]
        trace = []          # (widx, reasoning, answer, correct, mode)
        for j in idx:
            prefix = " ".join(words[: j + 1])
            v = answerer.solve(prefix, q["category"]) if any(ch.isdigit() for ch in prefix) else None
            if v is not None:
                trace.append((j, "🧮 computed with the calculator", v, answerer.judge(v, gold), "calc"))
            else:
                reasoning, ans = answerer.anticipate_sa_verbose(prefix, q["category"])
                trace.append((j, reasoning, ans, answerer.judge(ans, gold), "recall"))
        # features from this fresh trace, then buzzer P
        frec = {"id": rec["id"], "category": q["category"], "nwords": len(words),
                "gold": gold, "preds": [[j, a, c, mode] for j, _, a, c, mode in trace]}
        feats = [r[0] for r in buzzer.featurize(frec)]
        probs = clf.predict_proba(np.array(feats))[:, 1]
        buzz_i = next((i for i, p in enumerate(probs) if p >= T), None)
        cards.append({"q": q, "words": words, "gold": gold, "trace": trace,
                      "probs": probs, "buzz_i": buzz_i,
                      "buzz_word": idx[buzz_i] if buzz_i is not None else None})
    return cards


def render(cards):
    esc = html.escape
    out = ["""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>LeBot — How it thinks (walkthrough)</title><style>
body{background:#1a1b26;color:#c0caf5;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px}
h1{color:#7aa2f7}.intro{color:#9aa3b2;max-width:760px;margin-bottom:8px}
.card{background:#24283b;border:1px solid #2f334d;border-radius:10px;padding:18px;margin:18px 0;max-width:820px}
.qhead{font-size:13px;color:#565f89;margin-bottom:4px}.gold{color:#9ece6a;font-weight:600}
.full{margin:8px 0 14px;color:#a9b1d6}
.step{border-left:2px solid #2f334d;padding:8px 0 8px 14px;margin:0 0 4px}
.step.buzz{border-left-color:#bb9af7;background:#2a2e45;border-radius:0 8px 8px 0}
.heard{color:#7aa2f7;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.frag{color:#c0caf5}.reason{color:#9aa3b2;font-style:italic;margin:3px 0}
.commit{margin-top:3px}.ok{color:#9ece6a}.no{color:#f7768e}.unk{color:#565f89}
.p{font-variant-numeric:tabular-nums;color:#e0af68}
.bar{display:inline-block;height:8px;background:#7aa2f7;border-radius:2px;vertical-align:middle;margin-right:6px}
.buzztag{color:#bb9af7;font-weight:700}
</style></head><body>
<h1>LeBot — how it thinks, step by step</h1>
<div class=intro>For each question, every line is one moment as the question is read aloud:
what the model has <b>heard so far</b>, its <b>reasoning</b>, the <b>terse answer</b> it would
commit (✓/✗ vs the gold), and the trained buzzer's <b>P(correct)</b>. It buzzes 🔔 at the
first moment P crosses 80%. Probabilities are from a buzzer that did <i>not</i> train on these questions.</div>"""]

    for c in cards:
        q = c["q"]
        bw = c["buzz_word"]
        out.append(f"""<div class=card>
<div class=qhead>{esc(q['category'])} · {esc(q['tournament'])}</div>
<div class=full>{esc(q['question_text'])}</div>
<div>gold answer: <span class=gold>{esc(c['gold'])}</span></div><hr style="border-color:#2f334d">""")
        for i, (widx, reasoning, ans, correct, mode) in enumerate(c["trace"]):
            p = c["probs"][i]
            frac = widx / max(1, len(c["words"]) - 1)
            frag = " ".join(c["words"][: widx + 1])
            is_unk = ans.upper() == "UNKNOWN"
            acls = "unk" if is_unk else ("ok" if correct else "no")
            mark = "" if is_unk else (" ✓" if correct else " ✗")
            ansdisp = "—" if is_unk else esc(ans)
            buzz = (i == c["buzz_i"])
            out.append(f"""<div class="step{' buzz' if buzz else ''}">
<div class=heard>heard {frac:.0%}{' · 🔔 BUZZ' if buzz else ''}</div>
<div class=frag>{esc(frag)}</div>
<div class=reason>🧠 {esc(reasoning)}</div>
<div class=commit>commits: <span class={acls}>{ansdisp}{mark}</span>
&nbsp; <span class=bar style="width:{int(p*70)}px"></span><span class=p>P={p:.0%}</span>
{' <span class=buzztag>← buzzes here</span>' if buzz else ''}</div></div>""")
        if c["buzz_i"] is None:
            out.append('<div class=unk style="padding-left:14px">never reached P≥80% — abstains</div>')
        out.append("</div>")
    out.append("</body></html>")
    return "\n".join(out)


if __name__ == "__main__":
    cards = build()
    open("explain.html", "w").write(render(cards))
    print(f"wrote explain.html ({len(cards)} questions)")
