"""Smoke-test: the integrated router produces 4-element preds with calc/recall modes."""
import json
import sa_data

clean = json.load(open("packets/questions_clean.json"))
math_q = next(q for q in clean if q["question_style"] == "SHORT_ANSWER"
              and q["question_type"] == "TOSSUP" and q["category"] == "MATH"
              and any(c.isdigit() for c in q["question_text"]) and len(q["question_text"].split()) < 40)
bio_q = next(q for q in clean if q["question_style"] == "SHORT_ANSWER"
             and q["question_type"] == "TOSSUP" and q["category"] == "BIOLOGY"
             and len(q["question_text"].split()) < 40)

for q in (math_q, bio_q):
    qid, rec = sa_data.process(q)
    print(f"\n[{rec['category']}] gold={rec['gold']!r}")
    print(f"  {q['question_text'][:90]}")
    for widx, pred, correct, mode in rec["preds"]:
        if pred != "UNKNOWN":
            print(f"    w{widx:>2} [{mode:6}] {pred!r} {'OK' if correct else 'x'}")
