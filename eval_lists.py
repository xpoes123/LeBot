"""A/B eval: answer accuracy on full-stem numbered-list questions.
Run twice (before/after a META_SA edit) and compare. Golds are index-lists."""
import json
import re
import answerer

qs = json.load(open("nsba4_questions.json"))
lists = [q for q in qs if "1)" in q["stem"]
         and any(w in q["stem"].lower() for w in ["identify all", "order the", "rank the"])]


def norm_idx(s):
    """Normalize an index-list answer to a sorted-or-ordered tuple of ints, or None."""
    nums = re.findall(r"\d+", s or "")
    return tuple(nums) if nums else ()


def correct(pred, gold, stem):
    """Match pred to gold. gold is an index-list; pred may be indices OR item names."""
    if not pred or pred.upper() == "UNKNOWN":
        return False
    # ordering questions care about order; identify-all care about set
    is_order = any(w in stem.lower() for w in ["order the", "rank the"])
    gi = re.findall(r"\d+", gold)
    pi = re.findall(r"\d+", pred)
    if pi:
        return (pi == gi) if is_order else (set(pi) == set(gi))
    # pred gave names -> map via the numbered list in the stem
    items = {m[0]: m[1].strip().lower() for m in re.findall(r"(\d+)\)\s*([^;:\n]{2,45})", stem)}
    if not items:
        return False
    want = [items.get(n, "\0") for n in gi]
    pl = pred.lower()
    return all(w in pl for w in want)


hits = 0
none_hits = 0
none_total = sum(1 for q in lists if q["answer"].strip() in ("0", "none"))
for q in lists:
    ans, mode = answerer.anticipate_best(q["stem"], q["category"], n=3)
    ok = correct(ans, q["answer"], q["stem"])
    hits += ok
    is_none = q["answer"].strip() in ("0", "none")
    if is_none:
        none_hits += ok
    flag = "NONE" if is_none else "    "
    print(f"{flag} R{q['round']}Q{q['number']:<2} gold={q['answer']:<11} got={ans[:28]:<28} {'OK' if ok else 'XX'}")

print(f"\nOverall: {hits}/{len(lists)}  ({hits/len(lists):.0%})")
print(f"None-cases: {none_hits}/{none_total}")
