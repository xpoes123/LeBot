# LeBot — autonomous improvement session, 2026-11-11 (evening)

Goal: beat top Science Bowl players in live buzzer races on tossups at the weekend demo. All numbers are
held-out (no training was done; Jev decides, the LLM answers). Free/cached analysis first, then two
rehearsal harnesses, then three small PRs deployed to lebot.djiang.xyz and verified live.

## What I found (cached trajectories, 120 held-out SA tossups with human buzz logs; 78 unique — JHU shares packets with Stanford)

| | buzz% | acc | median buzz point | beats fastest human in room | beats median correct human | wrong interrupts |
|---|---|---|---|---|---|---|
| balanced (live this morning) | 63% | 83% | 76% | 12% | 18% | 8% |
| full send (Jev ≥ 0.6, no guard) | 72% | 51% | 49% | 8% | 13% | 33% |
| humans (fastest correct, median over Qs) | — | — | 57% | | | |

- **The gap is anticipation, not gating:** of the 76 questions a human won, 69 are ones where the bot never had the
  right answer early enough; only 7 were "had it, didn't fire". With an *oracle* gate (fire on the first correct
  guess, 0.3 s latency) the bot beats the median human on 47% of questions; the live Jev gate gets ~20%.
- **Latency is the next biggest term:** 26 of those 76 losses had a correct guess in hand at or before the human's
  buzz word — lost to the ~1.2 s guess-then-judge chain (+ ASR lag). Shipped fix #1 below.
- **The end-of-question race:** on ~30% of tossups nobody buzzes correctly in-stem; /read was waiting a 1.5 s silence
  timer before answering. Shipped fix #2.
- **The gate frontier is flat.** Requiring guess stability (same answer N ticks) on top of Jev, or lowering/raising T,
  moves points along the same curve (recall-only: ~31% beat-median at ~81% acc, any variant ±2 questions). Jev's
  calibration: after the question has "asked", P ≥ 0.95 is 96% right, 0.8–0.95 85%, 0.6–0.8 55%; before the ask
  everything under 0.95 is ≤ 48% right — the asked-yet guard is doing the right thing.

## Shipped (PRs #58–#62, all merged + deployed + verified live)

1. **PR #58 — stale-guess / fresh-judge.** At every new word the server first asks Jev about the *latest completed*
   Sonnet guess against the words heard since (~0.15 s) while Sonnet re-guesses in the background (~1 s). A right
   guess has usually settled ticks earlier, so the buzz lands ~3 words sooner. Held out, balanced: median buzz
   76% → 71% of the stem, accuracy unchanged (identical Jev inputs, only sooner); Jev volume unchanged (the fresh
   guess reuses Jev's answer when it matches). Live: a stale tick returns in ~0.2 s vs ~1.2 s.
2. **PR #59 + #60 — end-of-question speed.** /read ends a short-answer tossup on Deepgram's endpoint (relay
   `endpointing=500`) right after the closing "?" (incl. the empty endpoint message the handler used to drop) —
   ~1 s sooner than the silence timer, and there's no interrupt penalty once the question is done. "?" alone was
   *not* safe: Deepgram interims arrive ~1/s and put a "?" on mid-sentence pauses ("43 squared? Minus thirty three").
3. **PR #60 — no mid-read calc buzz in balanced/confident.** Rehearsal: Jev rated "What is the value of 43 squared"
   0.98 ready on a mid-sentence pause and Opus answered 1849 (−4). Balanced/confident now buzz calc tossups at the
   sentence's end and Opus computes there (760 in 1.0 s); held out, readiness fired late on calc tossups anyway
   (springs: word 44/46). Full send keeps mid-read readiness and the MLE setup bets.

## Rehearsal harness (new, in the session scratchpad — see memory for the path)

`rehearse.py`: Deepgram TTS reads a tossup (+3 s padded silence so the endpoint fires, like a live mic) → local
`/listen` relay → the same tick/buzz/end logic as `read.html`, in Python → timeline of where the buzz lands vs the
audio and what it answers. 7 synthesized tossups, balanced: 7/7 in match terms (recall buzzes 1–5 s before the
end; calc computed at the "?" within 1.1 s; MC answered with the choice's text before the choices were read, which
NSB accepts).

**12 held-out Stanford tossups with human buzz logs (balanced, after all four PRs' logic):** 9/12 right. Bot
buzzed before the *fastest* human in the room on 2 (seafloor spreading word 9 = tie; inductor word 11 vs 15),
before the *median* correct human on 5, lost the race but answered right on 4 (incl. 2 at the end: "charge" and
the Azurill probability 1/3), 1 wrong interrupt ("tin pest" at Jev 0.96 before the ask — the known pre-ask
risk), 2 wrong at the end (an "identify all" list whose items the ASR heard as "I get all of the following…" and
a data-truncated list question). The rehearsal also caught, and PRs #61/#62 fixed, three live bugs: a phrase-
completion guess ("…plotted on the eightfold" → "Eightfold Way", −4), a hyphen defeating the parrot guard
("The Mid-Atlantic Ridge"), and the 1.5 s silence timer ending a question mid-read on a transcript gap.

4. **PR #61 — guesser guards.** Reject a guess that completes the phrase being read (held out: 11 such guesses,
   0 right); `when|where|why` no longer count as "the question has asked" (they open clauses in tossups).
5. **PR #62 — silence timer needs Deepgram's endpoint** (a transcript gap is the transcriber lagging, not the
   reader stopping); parrot guard compares without punctuation.

## Tried, didn't help (not shipped)

- **Bank hints** (TF-IDF over the 25k past questions, answers of the 5 most similar past questions in the guess
  prompt; `.cache_hint.json` from an earlier session): aggregate accuracy 68% → 78% looked great, but on
  *recall* tossups the oracle ceiling is identical (71% = 71%); the gain was format priming on calc questions
  (bare numbers instead of "Using L ∝ R²T⁴") plus recycled "identify all" lists. Calc is routed to Opus anyway.
- **Guess-stability gating** (same answer ≥2/≥3 ticks, alone or OR'd with Jev): no frontier gain, leave-one-out.
- **Threshold sweeps** (T 0.6–0.95, pre-ask 0.9/0.95/none): balanced (0.8 / 0.95 pre-ask) is on the frontier.

## Spend

- Anthropic: ≈ $3–4 (three 7-question rehearsals, two 12-question rehearsal runs, Haiku judge calls for new
  (gold, guess) pairs). Jev: ≈ 1,100 calls (rehearsals + live verification; all cached analysis reused
  `.cache_force.json` / `.cache_live.json`). Both well under the $20 / 3,000-call budget.

## Top 3 recommendations for the weekend

1. **Run `/read` in balanced mode** (default). Full send loses: 33% wrong interrupts held out. Confident mode is
   safe but ~10 points later. Open the page once with `?key=…`, pick the category by reading the announcement
   ("Toss-up, biology, short answer, …") — autoCategory listens for it — and read at a natural pace; pause briefly
   after the "?" so Deepgram's endpoint fires (that is the end buzz).
2. **Expect to win calc and MC, and the end-of-question race; expect to lose in-stem recall races to a player who
   knows the clue.** The bot's edge is breadth (~84% right at the end) and speed after the question ends, not
   early recall buzzes (it beats the median correct human on ~1 in 5 tossups). Pick packets with a healthy calc/MC
   share if the point is to win.
3. **Rehearse the real room once with `rehearse.py` or a few spoken questions before the audience arrives**:
   the failure modes left are ASR drops (a lost "squared" → wrong value; /relisten catches some) and a moderator
   who runs sentences together without a pause after "?" (then the 1.5 s silence timer ends it). Keep
   `live.py`/`/read` on the VPS build (`git -C /opt/lebot log -1` should be ≥ #62). Known remaining failure modes:
   confident pre-ask topic guesses (Jev ≥ 0.95 before the ask is right ~76%: "tin pest" for allotropes), and
   "identify all" lists where the ASR garbles the lead-in.
