"""Offline test of the LIVE bot's decision loop, mocked without Discord or a microphone.

Simulates a question being transcribed word-by-word (as Whisper would emit it live) and
calls the SAME VPS /analyze endpoint the bot uses after each transcript update, printing
the guess / P(buzz) / buzz trajectory. This validates everything the bot does *after* it
hears audio — the part that matters — so the only remaining unknown is the audio transport.

Run: ./botvenv/bin/python test_pipeline.py
"""
import os
import sys

import httpx

LEBOT_URL = os.environ.get("LEBOT_URL", "https://lebot.djiang.xyz")

# (question, category, expected answer) — a reader would speak these left to right.
CASES = [
    ("This enzyme in the inner mitochondrial membrane synthesizes ATP as protons flow "
     "down their gradient", "BIOLOGY", "ATP synthase"),
    ("What is the powerhouse of the cell", "BIOLOGY", "mitochondria"),
    ("Identify all of the following three senses whose signals relay through the "
     "thalamus 1) Vision 2) Olfaction 3) Audition", "BIOLOGY", "1, 3"),
]


def run_case(client, stem, category, expected):
    words = stem.split()
    print(f"\n=== {category}  (expect: {expected}) ===")
    buzzed_at = None
    history = []  # the bot must feed prior guesses back so stability/churn accumulate
    for i in range(2, len(words) + 1):
        prefix = " ".join(words[:i])
        try:
            d = client.post(f"{LEBOT_URL}/analyze", json={
                "prefix": prefix, "category": category,
                "total_words": max(40, i * 2), "history": history}).json()
        except Exception as e:
            print(f"  w{i}: request failed: {e}")
            continue
        guess, p, buzz = d.get("guess", "?"), d.get("p_buzz", 0), d.get("buzzes")
        history.append({"guess": guess, "mode": d.get("mode", "recall")})
        mark = ""
        if buzz and buzzed_at is None:
            buzzed_at = i
            mark = "  <== BUZZ"
        # only print when something interesting changes, to keep it readable
        if buzz or guess.upper() != "UNKNOWN":
            print(f"  {i:2}w  …{prefix[-34:]:34}  {guess[:22]:22} P={p}{mark}")
    print(f"  -> buzzed at word {buzzed_at} of {len(words)}" if buzzed_at
          else "  -> never buzzed")
    return buzzed_at is not None


if __name__ == "__main__":
    with httpx.Client(timeout=30) as c:
        # smoke: endpoint reachable
        try:
            c.get(f"{LEBOT_URL}/").raise_for_status()
        except Exception as e:
            sys.exit(f"cannot reach {LEBOT_URL}: {e}")
        results = [run_case(c, *case) for case in CASES]
    print(f"\n{sum(results)}/{len(results)} cases buzzed. Pipeline (transcript->analyze->buzz) OK.")
