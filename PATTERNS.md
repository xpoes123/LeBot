# Science Bowl Mechanical Patterns — mined 2026-07-06

Findings from four parallel subagents: one read the official NSB 2026 Rules PDF,
three mined `packets/questions_clean.json` (24,359 questions). Corpus split:
66% short-answer / 34% MC; 52% tossup / 48% bonus.

Each pattern is tagged: **[RULE]** = grounded in official rules (ground truth),
**[STAT]** = empirical regularity from the corpus, **[BUILD]** = proposed
mechanical component.

---

## A. Answer-format rules (from NSB 2026 Rules PDF — ground truth)

- **[RULE] MC → answer the LETTER alone** (Rule 3-3). W/X/Y/Z is one token, immune to
  the "verbal answer must be exact" trap. Never give both letter + value (Rule 3-7:
  both must then be correct). → *only add a failure mode.*
- **[RULE] List/selection questions ("identify all that…") — distinguish items ANY
  sufficient way** (Appendix A-2f). Accepts item numbers, ordinal position ("the last
  one", "next to last"), any sufficient verbal descriptor, or "all"/"none". CONFIRMS
  David's thesis — you never need to name/spell an entity, just distinguish it.
  Sufficiency is an *un-appealable judgment call* (Rule 7-2) → over-specify.
- **[RULE] Ordering questions must give the FULL order** — but each element may be a
  number or any sufficient descriptor, not the exact name. (No rule authorizes a
  *partial* order or omitting items — correction to the original belief: the
  "name only the distinguishing item" trick is for SELECTION, not ORDERING.)
- **[RULE] Multiple-answer SA — label "which is which" or answer in asked order**
  (A-2e). "largest is 7, smallest is 2" is safe in any sequence; "7 and 2" in wrong
  order without labels is WRONG.
- **[RULE] Omit units** (A-2h) — optional, but stating a wrong/non-equivalent unit is a
  failure mode. Unspecified → SI base units.
- **[RULE] Surname only** (A-2i) — "Einstein", not "Albert Einstein" (first name, if
  given, must be correct → don't give it).
- **[RULE] Name the entity, don't describe it** (A-2g) — "Newton's Second Law" not
  "F=ma"; but short canonical symbols count ("c" for speed of light).
- **[RULE] Formula only when no isomers** (A-2c) — "CO2" ok for carbon dioxide;
  "C2H6O" NOT ok for ethanol (isomers exist). Disambiguate ion charge ("copper (II)",
  never "copper ion").
- **[RULE] Numeric form** (A-2a) — exact + simplest; integers as integers; no scientific
  notation unless requested; no repeating decimals (use fractions); radicals
  rationalized; angles in [0,2π); coefficients GCD=1.
- **[RULE] Answer token ONLY, first utterance commits** (Rule 3-7) — any preface
  ("the answer is…"), restating, or spoken reasoning counts as WRONG. No self-correction.
- **[RULE] Interrupt risk is asymmetric** (Rule 6-2): wrong interrupt → opponent +4;
  wrong answer on a *fully-read* tossup → 0 to your team. So on a full read, guessing is
  nearly free; interrupting requires real confidence. (In 3+ team tiebreakers, −1 for
  wrong → abstain below P(correct)=0.5.)
- **[RULE] MC: may interrupt before choices are read** (Rule 5-2 timing starts after
  full read; the 4 options are known up front). Buzz from the stem if confident.
- **[RULE] Hardcodable defaults** (Appendix A-1): g=9.8, c=3e8, sound=340 m/s, abs
  zero −273°C, base 10, radians, fair coins/dice, deck of 52, functions over reals.

## B. Multiple-choice empirical priors (from corpus, n=8,241 4-option MC)

- **[STAT] Correct-letter prior is NOT uniform**: W 21.5% / X 27.2% / Y 27.4% / Z 23.9%
  (χ²≈40, p<1e-4). Measure the bot's real edge against **27% (best-guess X)**, not 25%.
  Break ties toward X/Y, away from W.
- **[STAT] Numeric options are ~88% pre-sorted ascending** (92.7% sorted either way,
  n=914). → Option *position carries no information*; kill any positional prior on
  numeric questions.
- **[STAT] Sorted-numeric answers skew MIDDLE**: middle two ranks 63.8% vs extremes
  36.2%; the single largest value is the rarest correct answer. Mild "avoid extremes,
  especially the largest" prior.
- **[STAT] Negated MC stems (CAPS NOT/EXCEPT/LEAST) skew Y/Z ≈ 59%** (χ²=21, p<0.001).
- **[STAT] "closest to" (estimation) → X/Y ≈ 67%** (the sorted-numeric middle bias).
- **[STAT] Do NOT use length heuristics** — longest/shortest option is *below* chance.
  "all/none of the above" and lone mismatched-unit options are *distractor* tells
  (~15-18%, below chance) → down-weight.

## C. List-question structure (from corpus)

- **[STAT] Ordering answers: ~96% are index permutations, 72% exactly 3 items**
  ("3, 1, 2"). Normalize whitespace around commas.
- **[STAT] Multi-select answer distribution**: PAIR 41% > SINGLE 26% > TRIPLE/ALL 23% >
  NONE 6%. **The modal answer is a PAIR** → "pick the one true statement" is wrong ~74%
  of the time. *Evaluate each numbered item independently as true/false*, emit the set.
- **[STAT] "none"/0 base rate ~6%, "all" ~8%** — the necessity trap is real but rare;
  never rule them out, but don't over-predict them either.

## D. Linguistic cues → early format/category commitment

- **[STAT] Openers predict answer FORMAT** (commit from first ~4 tokens):
  - "which of the following" → MC letter (97% MC)
  - "identify all" → subset {ALL, NONE, n ONLY, n AND m} (100% SA)
  - "order/rank/arrange the following" → index permutation (100% SA)
  - "how many" / "calculate/compute/find" / unit-words → bare number
  - "what is the name/term for", "refers to", "is defined as" → short recall term (1-3 words)
- **[STAT] Category tell**: "researchers/scientists at…" → 98% ENERGY category.
- **[STAT] Negation ~9% of all questions**, usually UPPERCASE (41% of "NOT", nearly all
  "EXCEPT"). Detect case-sensitive; uppercase = high-confidence flip. Silently missing it
  inverts the answer — the single most common catastrophic-error source.
- **[STAT] Definitional frames** ("is the term for") front a 2-3 word recall noun,
  90%+ SA — constrain the decoder to a short noun phrase, disable numeric/list formats.

## E. Mechanically-solvable ordering (proposed [BUILD])

~12-13% of ordering questions sit on a **canonical monotonic sequence** and can be solved
with zero LLM reasoning by looking up each item's position and reading the direction word:
periodic trends (atomic radius / ionization energy / electronegativity / electron
affinity), EM spectrum (wavelength/frequency/energy), Mohs hardness, taxonomic rank,
planetary order, geologic time scale, metamorphic grade. Another ~20% are table-lookup
axes (boiling/melting point, pKa, reduction potential, density) solvable if the bot
carries small reference tables. → A `sequences.py` with these canonical orders + an
axis-detector in the answerer would deterministically nail this slice.

---

## What was applied to the answerer (see git log)

See `answerer.py` META_SA / META and the backtest in `eval_lists.py`. Priority order:
1. Multi-select "evaluate each item independently" (biggest accuracy lever).
2. Answer-format rules (omit units, surname, name-not-description, letter for MC).
3. Format router by opener (commit format early).
4. (Deferred / documented) MC letter priors, canonical-sequence ordering solver.
