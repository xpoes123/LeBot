"""Knowledge base = the 25k past questions, retrievable by (partial) stem.

v0 retrieval: TF-IDF + cosine (sklearn, no torch). Good at surfacing recycled /
near-duplicate questions from a partial stem, which is the main signal a
past-questions KB carries. Upgrade to embeddings only if this falls short.

CRITICAL: retrieve(..., exclude_tournament=X) drops X's own questions, so when we
test on a tournament the KB can't just return the question itself. That keeps the
"studied other material, now facing a new question" setup honest.
"""
import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

QFILE = os.path.join(os.path.dirname(__file__), "packets", "questions_clean.json")


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


class KB:
    def __init__(self, questions):
        self.q = questions
        self.tour = np.array([x["tournament"] for x in questions])
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.mat = self.vec.fit_transform(_norm(x["question_text"]) for x in questions)

    def retrieve(self, text, k=5, exclude_tournament=None):
        qv = self.vec.transform([_norm(text)])
        sims = linear_kernel(qv, self.mat).ravel()
        if exclude_tournament is not None:
            sims = np.where(self.tour == exclude_tournament, -1.0, sims)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.q[i]) for i in idx if sims[i] > 0]


def load(path=QFILE):
    return KB(json.load(open(path)))


if __name__ == "__main__":
    kb = load()
    print(f"KB: {len(kb.q)} questions, vocab {len(kb.vec.vocabulary_)}")

    # demo: retrieve for a PARTIAL stem, excluding the source's own tournament
    sample = next(x for x in kb.q if x["question_style"] == "SHORT_ANSWER"
                  and len(x["question_text"].split()) > 12)
    words = sample["question_text"].split()
    print(f"\nSOURCE [{sample['tournament']}/{sample['category']}] ans={sample['correct_answer']!r}")
    print(f"  full: {sample['question_text'][:110]}")
    for frac in (0.4, 0.7, 1.0):
        partial = " ".join(words[: max(1, int(len(words) * frac))])
        hits = kb.retrieve(partial, k=2, exclude_tournament=sample["tournament"])
        print(f"\n  at {frac:.0%}: \"{partial[:80]}...\"")
        for sim, h in hits:
            print(f"    {sim:.2f} [{h['tournament']}] ans={h['correct_answer']!r}  {h['question_text'][:70]}")

    # recyclability: for short-answer Qs, how often does the FULL stem find a
    # cross-tournament near-duplicate with the SAME answer? (how much is recycled)
    sa = [x for x in kb.q if x["question_style"] == "SHORT_ANSWER"][:400]
    dup = same = 0
    for x in sa:
        hits = kb.retrieve(x["question_text"], k=1, exclude_tournament=x["tournament"])
        if hits and hits[0][0] > 0.6:
            dup += 1
            if _norm(hits[0][1]["correct_answer"]) == _norm(x["correct_answer"]):
                same += 1
    print(f"\nrecyclability (400 short-answer Qs): {dup} have a cross-tournament "
          f"near-dup (sim>0.6), {same} of those share the answer")
