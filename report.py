"""Build a human-validatable HTML report of short-answer anticipation.

For each question: the stem revealed word by word, the bot's evolving guess at each
prefix, its calibrated buzz probability (out-of-fold, honest), where it would buzz,
and whether that was right. Grouped by outcome so you can sanity-check each type.

Run: python report.py  -> report.html
"""
import hashlib
import html
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

import buzzer

CLEAN = os.path.join(os.path.dirname(__file__), "packets", "questions_clean.json")
T = 0.8  # buzz threshold (the strong operating point)


def stem_map():
    m = {}
    for q in json.load(open(CLEAN)):
        i = hashlib.md5((q["tournament"] + q["question_text"]).encode()).hexdigest()[:10]
        m[i] = q["question_text"]
    return m


def build():
    data = json.load(open(buzzer.LABELS))
    stems = stem_map()
    X, y, frac, groups = buzzer.load_rows()
    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    oof = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]

    # index oof P by (qid, row-order)
    by_q = {}
    pos = 0
    for rec in data.values():
        k = len(rec["preds"])
        by_q[rec["id"]] = oof[pos:pos + k]
        pos += k

    cards = []
    for rec in data.values():
        qid = rec["id"]
        probs = by_q[qid]
        stem = stems.get(qid, "")
        words = stem.split()
        # buzz at first prefix with P>=T
        buzz_i = next((i for i, p in enumerate(rec["preds"]) if probs[i] >= T), None)
        rows = []
        for i, p in enumerate(rec["preds"]):
            rows.append({
                "word": p[0], "frac": p[0] / max(1, rec["nwords"] - 1),
                "pred": p[1], "correct": bool(p[2]), "p": float(probs[i]),
                "calc": (len(p) > 3 and p[3] == "calc"), "buzz": (i == buzz_i),
            })
        if buzz_i is None:
            outcome = "abstain"
        else:
            outcome = "correct" if rec["preds"][buzz_i][2] else "wrong"
        buzz_word = rec["preds"][buzz_i][0] if buzz_i is not None else None
        cards.append({
            "cat": rec["category"], "gold": rec["gold"], "tour": rec["tournament"],
            "words": words, "nwords": rec["nwords"], "rows": rows,
            "outcome": outcome, "buzz_word": buzz_word,
            "buzz_frac": (buzz_word / max(1, rec["nwords"] - 1)) if buzz_word is not None else None,
        })
    order = {"correct": 0, "wrong": 1, "abstain": 2}
    cards.sort(key=lambda c: (order[c["outcome"]], c["buzz_frac"] if c["buzz_frac"] is not None else 9))
    return cards


def render(cards):
    n = len(cards)
    nc = sum(c["outcome"] == "correct" for c in cards)
    nw = sum(c["outcome"] == "wrong" for c in cards)
    na = sum(c["outcome"] == "abstain" for c in cards)
    esc = html.escape
    out = [f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>LeBot — Short-Answer Anticipation</title><style>
body{{background:#1a1b26;color:#c0caf5;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px}}
h1{{color:#7aa2f7;font-size:22px}} .sub{{color:#565f89;margin-bottom:20px}}
.card{{background:#24283b;border:1px solid #2f334d;border-radius:10px;padding:16px;margin:14px 0;max-width:900px}}
.badge{{font-size:12px;padding:2px 8px;border-radius:6px;background:#414868;color:#c0caf5;margin-right:6px}}
.gold{{color:#9ece6a;font-weight:600}} .stem{{margin:10px 0;font-size:16px}}
.buzzw{{background:#bb9af7;color:#1a1b26;padding:0 4px;border-radius:4px;font-weight:700}}
.heard{{color:#c0caf5}} .unheard{{color:#565f89}}
table{{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}}
td,th{{text-align:left;padding:3px 8px;border-bottom:1px solid #2f334d}}
.ok{{color:#9ece6a}} .no{{color:#f7768e}} .unk{{color:#565f89}}
.bar{{display:inline-block;height:9px;background:#7aa2f7;border-radius:2px;vertical-align:middle}}
.buzzrow{{background:#2a2e45}}
.tag-correct{{color:#9ece6a}} .tag-wrong{{color:#f7768e}} .tag-abstain{{color:#e0af68}}
</style></head><body>
<h1>LeBot — Short-Answer Anticipation</h1>
<div class=sub>{n} questions · <span class=tag-correct>{nc} buzzed correct</span> ·
<span class=tag-wrong>{nw} buzzed wrong</span> ·
<span class=tag-abstain>{na} abstained</span> · buzz threshold P≥{T:.0%} · honest (out-of-fold) probabilities.
Each row = the answer the bot would give having heard only that many words. 🔔 = where it commits.</div>"""]

    for c in cards:
        tag = {"correct": "✓ buzzed CORRECT", "wrong": "✗ buzzed WRONG",
               "abstain": "— abstained (never confident)"}[c["outcome"]]
        # stem with buzz word highlighted
        bw = c["buzz_word"]
        parts = []
        for i, w in enumerate(c["words"]):
            cls = "heard" if (bw is not None and i <= bw) else ("unheard" if bw is not None else "heard")
            if i == bw:
                parts.append(f'<span class=buzzw>{esc(w)} 🔔</span>')
            else:
                parts.append(f'<span class={cls}>{esc(w)}</span>')
        stem_html = " ".join(parts)
        loc = (f"buzz @ word {bw}/{c['nwords']} ({c['buzz_frac']:.0%} in)"
               if bw is not None else "no buzz")
        trows = []
        for r in c["rows"]:
            pcls = "ok" if r["correct"] else ("unk" if r["pred"].upper() == "UNKNOWN" else "no")
            pred = esc(r["pred"]) if r["pred"].upper() != "UNKNOWN" else "<i>—</i>"
            if r.get("calc"):
                pred = "🧮 " + pred
            mark = "✓" if r["correct"] else ("" if r["pred"].upper() == "UNKNOWN" else "✗")
            trows.append(
                f'<tr class="{"buzzrow" if r["buzz"] else ""}">'
                f'<td>{r["word"]} <span class=unk>({r["frac"]:.0%})</span></td>'
                f'<td class={pcls}>{pred} {mark}</td>'
                f'<td><span class=bar style="width:{int(r["p"]*60)}px"></span> {r["p"]:.0%}'
                f'{" 🔔" if r["buzz"] else ""}</td></tr>')
        out.append(f"""<div class=card>
<div><span class=badge>{esc(c['cat'])}</span><span class=badge>{esc(c['tour'])}</span>
<span class="tag-{c['outcome']}">{tag}</span> · <span class=unk>{loc}</span></div>
<div class=stem>{stem_html}</div>
<div>gold answer: <span class=gold>{esc(c['gold'])}</span></div>
<table><tr><th>heard</th><th>bot's guess</th><th>P(correct)</th></tr>{''.join(trows)}</table>
</div>""")
    out.append("</body></html>")
    return "\n".join(out)


if __name__ == "__main__":
    cards = build()
    open("report.html", "w").write(render(cards))
    print(f"wrote report.html ({len(cards)} questions)")
