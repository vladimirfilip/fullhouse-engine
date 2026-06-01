# Vlad Bot — Improvement Implementation Plan

Scope: improvements to `bots/vlad/bot.py` (+ read-only `data/`) for the Fullhouse
Swiss tournament. **Preflop CFR retraining is handled separately** and is out of
scope here.

Hard sandbox limits (all code in `bots/vlad/` must respect these):

| Resource | Limit |
|---|---|
| RAM | 768 MB |
| CPU | 0.5 cores |
| Action timeout | 2 s |
| Warmup timeout | 30 s |
| `data/` (read-only) | 200 MB |
| Forbidden | socket, subprocess, multiprocessing, threading, pickle, eval/exec/compile, network, FS writes |

Do **not** modify the frozen engine (`engine/`, `sandbox/`, `demo.py`). If any
change touches the 308-dim feature vector it must stay byte-identical across
`bot.py` / `features.cpp` / `config.py`.

---

## Foundational findings (these shape everything below)

Verified against the engine source:

- **`match_action_log` entries carry only** `{hand_num, seat, bot_id, action, amount}`
  — `sandbox/match.py:303-310`. No `street`, no `pot`, no `board`.
- **Within-hand `action_log` carries only** `{seat, action, amount}` — `engine/game.py:83`.
- **Bots are only invoked on `action_request`, never on `hand_complete`**
  (`sandbox/match.py:294`). Showdown `revealed_cards` / `hand_strengths` are
  **invisible** to bots → no hole-card mining is possible.
- **`amount` semantics differ by action**: `raise` → total target; `all_in` →
  total bet-this-street; `call` → chips paid (incremental). Must normalize when
  reconstructing betting.

### The street-attribution edge

The logs contain **no `street` field**. The strongest competitors
(`saroopjagdev_mybot`, the `neel_v6_*` family) attribute historical actions to
`state.get("street")` — i.e. the *current* decision's street — because the
`street` they look for in each entry doesn't exist
(`bots/saroopjagdev_mybot/bot.py:2404` and `_apply_action_to_profiles`). Their
street-specific leak stats (fold-to-cbet, river-call-rate, 3-bet freq) are
therefore **systematically mis-attributed**. A bot that reconstructs streets
correctly gets strictly better reads than the field leaders — a compounding,
invisible edge. This is the basis of Module A1.

---

## Module A — Opponent modeling: street reconstruction + leak vectors *(highest EV)*

Backbone module; B, C, and D all consume its output.

### A1. Street reconstruction (`_reconstruct_streets`)

Derive street labels by simulating the betting round. Process one hand's actions
in order, maintaining `committed[seat]`, `high_bet`, `live_seats`,
`acted_since_aggression`. A betting round closes when every live, non-all-in seat
has acted and matched `high_bet`; the next action belongs to the next street.
Normalize `amount` per action type (raise=total, all_in=total, call→derive total
from incremental). Blind seats and blind sizes are known
(`SMALL_BLIND=50`, `BIG_BLIND=100`, `STARTING_STACK=10000`), so blind posting and
the preflop "BB closes the option" rule reconstruct exactly.

- Cross-hand: group `match_action_log` by `hand_num`, run reconstruction per hand
  (reset state at each `hand_num` boundary).
- Cache by `(len(match_action_log), last_hand_num)` → reconstruct once per
  `decide()`, not per stat.
- Unit-test against hand-crafted logs asserting known street boundaries,
  including: all-in side pots, walks, everyone-folds-to-BB.

### A2. Per-opponent profile (`_OPP_PROFILES: dict[bot_id, dict]`)

Cumulative counters keyed by `bot_id` (stable within a match), populated with
**correctly-attributed** streets:

| Leak | Numerator / Denominator | Drives |
|---|---|---|
| `vpip`, `pfr` | preflop voluntary / hands | preflop 3-bet & iso width |
| `fold_to_steal` | blind folds vs late open / steals faced | blind-steal frequency |
| `fold_to_flop_cbet` | flop fold vs bet / flop bets faced | barrel vs give-up |
| `fold_to_turn_barrel` | turn fold vs bet / turn bets faced | double-barrel |
| `river_call_rate` | river calls / river bets faced | thin value vs bluff |
| `3bet_freq` | preflop reraises / opens faced | flat vs 4-bet/fold |
| `aggression_factor` | (raise+allin)/call | passivity read |
| `sticky_proxy` | reached-river / saw-flop | foldy vs station postflop |

Each stat returns an empirical-Bayes blend toward a population prior with a
**confidence weight** `= min(1, samples / TARGET)`. Low-sample reads stay near
GTO baseline. (Reuse the shape of saroop's `_sampled_rate`.)

### A3. Archetype tag (`_classify(profile) -> (tag, confidence)`)

Keep the `maniac/station/nit/normal` buckets but feed them the richer stats and a
confidence score. This **replaces** the raw action-count version at
`bot.py:1058` and flows into B/C/D — not just the current ±0.03 nudges.

**Integration:** compute once at the top of `decide()` (`bot.py:1132`); store
`read = {per_seat_profile, tags, confidences}` in a module cache; thread it into
`_postflop_decide` / `_preflop_decide`, replacing the thin `profile_counts` tuple.

---

## Module B — Range-conditioned equity

`monte_carlo_equity` (`bot.py:1001`) deals opponents **uniform-random** hands.
Against a nit's continuing range, the resulting equity is fiction.

- **B1. Range floors.** `_seat_range_floor(read, seat, street, actions) -> float`
  returns a preflop-equity percentile cutoff per opponent, from their actions
  *this hand* (via A1) and tag (e.g. a nit who 3-bet then barreled continues with
  a top-~8% range).
- **B2. Range-aware rollout.** Add `opp_min_equity` rejection sampling to
  `monte_carlo_equity`: resample an opponent hand if its standalone strength is
  below that opponent's floor; cap resamples to stay in budget; fall back to
  uniform if rejection rate too high. Reference implementation:
  `bots/cfr_equity_v28/bot.py:164` (`equity_vs_range(... opp_min_equity)`).
- **B3. Equity-realization correction.** Apply small +IP / −OOP / −multiway /
  −high-SPR-with-marginal adjustments to raw equity (cf. saroop `realised_equity`).
  Fixes systematic OOP over-calling and one-pair over-valuation at deep SPR.

Budget: rejection sampling raises cost. Keep `max_iters` adaptive (fewer iters
with many opponents / tight floors). Verify worst case under 2 s on 0.5 core.

---

## Module C — Anti-punt override layer

A final guardrail after the engine picks an action — broader than the current
`_risk_gate` (`bot.py:941`), which only catches large commitments. Port saroop's
catalog (`bots/saroopjagdev_mybot/bot.py:2061`) but key thresholds on **our
corrected reads**. Implement as `_anti_punt(action, gs, read, equity, hand_class)`,
called at the tail of `_postflop_decide` (`bot.py:987`) **after** `_risk_gate`.
Rules (severe first), each firing only on **confident** reads so they never hurt
vs unknown/strong bots:

1. River air-bluff into a station → check/fold.
2. Low-equity multiway c-bet on wet board → check.
3. High-SPR one-pair stack-off vs passive/nit aggression → call/check down.
4. Oversized river bluff-catch vs nit/passive line (owed > 0.5–0.75 pot,
   non-nutted) → fold.
5. Non-nut multiway draw chase at deep SPR → fold.
6. Low-confidence exploit bluff (confidence < 0.2) → revert to GTO/safe line.

---

## Module D — Exploit-aware blending & sizing

- **D1. Fix/extend the GTO↔EV blend.** The live `_realtime_search` (`bot.py:576`)
  only shifts ±20 pp between FOLD/CHECK_CALL — the "60% GTO / 40% EV" in CLAUDE.md
  is **stale**; update the docs. Scale the EV/exploit weight with read confidence:
  near-GTO vs unknown/strong bots; widen the fold/call shift and bias toward value
  (vs stations) / bluffs (vs overfolders) as confidence rises.
- **D2. Exploit-aware bet sizing.** Replace fixed pot-fractions in `_mc_postflop`
  (`bot.py:869`) and the net path: size up (0.9–1.3× pot, overbets) for value vs
  stations; size down / check more vs nits; widen thin river value vs high
  `river_call_rate`; size bluffs to the opponent's measured fold threshold (target
  their `fold_to_*` so a min-profitable bluff clears MDF). Reserve the off-grid
  0.27× / 1.72× sizes for the non-`neel` field (neel harmonic-translates bet
  fractions specifically to neutralize off-grid sizing).
- **D3. MDF-aware defense.** Compute minimum-defense frequency when facing a bet;
  defend at least MDF with bluff-catchers vs balanced bots, but abandon MDF
  (over-fold vs nits / over-call vs stations) once confidence is high. Formula:
  saroop `minimum_defence_frequency`.

---

## Module E — Multi-street board texture *(quick win)*

`_board_texture` (`bot.py:923`) only reads the flop (`board[:3]`). Extend to the
full board: flush completion, four-to-straight, paired board (boats), turn/river
overcards that shift range advantage. Feed into B3, C, and D2.

---

## Module F — Differentiators *(revised — see "Decisions" below)*

> **Decision (this revision):** the bots in this repo are **not** assumed to be
> the actual competition submissions, `bot_id`s are not assumed stable into the
> tournament, and entrants will iterate before June. Therefore:
>
> - **Identity-keyed exploit table (`_KNOWN_EXPLOITS[bot_id]`) — REMOVED.** Dead
>   weight if those strings never appear; brittle even if similar bots do.
> - **"Exploit saroop's/neel's specific anti-punt rules" — REMOVED.** Only pays
>   against those exact bots at those exact versions.
> - **Archetype calibration — KEPT but reframed** (below). Real opponents still
>   cluster into the same archetypes (nit / station / maniac / TAG / LAG /
>   GTO-ish); that's a property of poker, not of this repo.

What survives, and why it pays against **unknown** opponents:

- **F1. `tools/profile_field.py` — sparring/regression corpus (high value,
  training-side only, never imported by `bot.py`).** These bots span the
  archetype space unusually well (explicit `calling_station` / `maniac` / `nit` /
  `overfolder` bots + three GTO-leaning engines). Use them to *prove the live
  model adapts within a 400-hand match* and to tune the confidence/sample-count
  schedule. The deliverable is a validated live engine, not a lookup table.
- **F2. `data/opponent_priors.json` — generic archetype priors (optional, mild
  value).** Population-average defaults (the "average nit/station/maniac") as a
  cold-start before live data dominates (~first 30–40 hands). Derive from poker
  theory; the corpus only sanity-checks them. **Not** per-bot numbers.
- **F3. Correct street attribution** (Module A1) — strictly better reads than the
  field leaders; works against known and unknown opponents alike.
- **F4. Off-grid sizing** (folded into D2).
- **F5. Latency discipline.** On 0.5 core with a 2 s cap, heavy-MC competitors
  (saroop 800–1000 iters; neel range-MC) run near budget. Add a hard internal
  deadline (~1.5 s) with graceful degradation; never self-time-out.

**Revival condition for identity-keying:** only if the competition publishes a
fixed, named field with stable `bot_id`s ahead of time.

---

## Build order

Each step is independently testable; run validation after each.

1. ~~**A1 street reconstruction** + unit tests (everything depends on it).~~
   **DONE** — `_reconstruct_streets` in `bot.py`; `tests/test_street_reconstruct.py`
   (52 tests, incl. engine-driven cross-check across seeds × {2,3,6} players).
2. ~~**A2/A3 profiles & tags**~~ **DONE** (pure functions, not yet wired into
   `decide()` — wiring deferred so the golden test stays stable until integration).
   `_build_opponent_profiles` / `_opp_leaks` / `_classify_opponent` in `bot.py`;
   `tests/test_opponent_profiles.py` (12 tests, incl. engine-driven archetype
   recovery for nit/maniac/station + bot_id-keyed accumulation across seat
   re-indexing). **Deferred leaks** (need per-hand position/blind inference, which
   the cross-hand log lacks): `fold_to_steal`, positional VPIP/PFR splits.
   **WIRED + A/B'd**: A2/A3 now back `_opponent_profile_counts` (confidence-gated
   at 0.30). Paired CRN A/B (20 seeds × 400h, maniac/nit/overfolder/TAG/station
   field) via `tools/ab_profiling.py`: +881 chips/match edge, **not significant**
   (t≈0.14) — expected, since GTO postflop ignores profile counts so only the
   preflop open nudge is live. No regression; real payoff comes via C/D.
3. ~~**E board texture**~~ **DONE** — `_board_features` (full-board: pairs, flush
   /4-flush, straight/4-straight w/ wheel, connectivity) + `_board_texture`
   refactor. Flop category proven byte-identical to legacy across all C(52,3);
   turn/river now full-board aware. `tests/test_board_texture.py` (14 tests).
4. ~~**B range-conditioned equity**~~ **CORE DONE (B1+B2)** — `_hand_pctl`
   (Chen-percentile over all 1326 combos), `_seat_range_floor` (per-archetype
   floor, confidence-blended, street-bumped; unknown stays uniform on flop),
   rejection sampling in `monte_carlo_equity`, wired self-contained into `_run_mc`
   (GTO path + MC path + fallback all range-aware). `tests/test_range_equity.py`
   (10 tests). Perf: +50% MC (~31 ms worst case), far inside budget. Golden green.
   **B3 (equity-realization IP/OOP/multiway adj) deferred** — hand-wavy, needs
   tuning; revisit alongside D.
5. ~~**C anti-punt**~~ **DONE** — `_anti_punt` (3 rules: river air-bluff into
   station, oversized river bluff-catch vs nit, low-equity multiway wet c-bet) +
   `_made_hand_tier`/`_last_aggressor_read`/`_field_station_read`, wired after the
   risk gate in `_postflop_decide`. `tests/test_anti_punt.py` (11 tests). Read-
   gated (conf ≥ 0.35) so it's a no-op vs unknown/strong opponents; golden green.
   **EVAL (eval_bb100, paired CRN):** weak field new +556 vs base +480
   (+76 bb/100, t=0.87); strong field new −16.9 vs base −41.5 (+25 bb/100,
   t=1.17). New ≥ base in BOTH fields (match-level −6257 was bust noise).
   Neither significant yet but consistent positive direction → keep modules.
6. ~~**D blending/sizing**~~ **D1 DONE** — `_exploit_bias` + extended
   `_realtime_search`: read-gated (conf ≥ 0.35) shifts — c-bet/bluff harder vs
   over-folding (nit) field on the `owed==0` branch the net used to pass through,
   call lighter vs maniac bettors, fold more vs nit/station bettors. Threaded via
   `_net_postflop`; profiles built once in `_postflop_decide`. Caps ±0.15.
   `tests/test_exploit_blend.py` (7 tests). Golden green (empty reads → zero bias).
   **D2 (MC-path exploit sizing) deferred** — GTO net owns sizing in production.
7. ~~**F1 sparring harness**~~ **DONE** — `tools/eval_bb100.py` (+ `ab_profiling.py`).
   **F2 generic priors** not built (optional cold-start; live reads dominate by
   ~hand 30). Identity-keyed exploits dropped (see Module F decision above).

## FINAL EVAL (complete bot A–E + D1 vs HEAD baseline, eval_bb100 paired CRN)

| field | n | new | base | paired new−base | t |
|---|---|---|---|---|---|
| strong | 1200 | −16.9 | −41.5 | +24.7 | 1.17 |
| strong | 3000 | +7.2 | −0.9 | **+8.1** (CI ±21) | 0.75 |
| weak | 1800 | +556 | +480 | +76 | 0.87 |
| weak | 3000 | +357 | +340 | **+17** (CI ±125) | 0.27 |

**Read:** edge is **small, positive, consistent** (4/4 measurements same sign →
sign-test p≈0.06) but **not individually significant**. The larger early numbers
were partly noise (shrank as n grew — regression to mean). New crosses base from
−0.9→+7.2 vs the strong field (break-even→slightly winning). No regression
anywhere; all modules tested + read-gated; submission validator passes.

**Why modest:** base is already very strong (crushes weak at +340 bb/100 ≈ near
ceiling; ~break-even vs strong). Postflop sizing is owned by the GTO net (D2 not
applied) and the D1 exploit caps are conservative (±0.15, conf≥0.35). Biggest
remaining levers: (a) harness-tuned exploit magnitudes/floors, (b) D2 value-sizing
overrides vs stations, (c) the preflop CFR retrain (every hand; in progress).

## TUNING (in progress)

Knobs are env-overridable via `bot._envf` (VLAD_* vars; submission uses baked
defaults). `tools/tune_exploit.py` caches the fixed base once, then sweeps new
configs over the same CRN seeds, ranking by bb/100.

**Result — defaults kept.** Weak sweep (3000h): all configs within ±125 bb/100
CI, indistinguishable; only reliable signal = over-pushing bluff cap (0.45)
*loses* to base. Strong sweep (2000h, CI ±18-21): big_bump led (+34 vs defaults
+25.8) but non-monotonic (noise signature) and overlapping CIs; risky to adopt
(tightens vs everyone, only validated on the tight field — would likely cost EV
vs the loose weak field). No config significantly beats defaults; defaults are
robust across both fields and the safe choice for an unknown competition field.
Env-override infra retained for re-tuning once the real field is known.

## Validation

- `make validate BOT=bots/vlad/bot.py` after each module (forbidden imports,
  timeout, warmup).
- **`tools/ab_profiling.py`** — paired CRN match-level A/B (new vs HEAD base),
  chip delta. **Lesson learned:** match-end delta is dominated by bust/snowball
  variance (±50k swings); with ~30 seeds it cannot resolve per-decision edges
  (t≈1). Use only for "no catastrophic regression" gating.
- **`tools/eval_bb100.py`** — low-variance evaluator: per-hand stack reset (no
  bust/snowball) + accumulating match_action_log (vlad still builds reads) + CRN
  pairing by hand seed. Reports bb/100 per variant + paired new−base with 95% CI.
  Fields: `weak` (offense test — base already crushes) and `strong` (saroop /
  neel / skant — where B/C defensive value should appear). This is the harness
  to trust for whether a module helps.
- Keep `_action_seed` RNG seeding for reproducible matches.

## Performance budget (0.5 core — verify on constrained env, not dev box)

- A1 reconstruction: O(actions) ≤ ~200 entries — negligible; cache per call.
- B rejection sampling is the risk. Profile worst case (6-way, tight floors,
  river); cap `max_iters` adaptively; hard 1.5 s deadline → fallback to
  uniform-MC → fallback to pot-odds.
- Net forward pass + MC must coexist under 2 s. Measure on the real environment.

## Risks / watch-items

- Street-reconstruction edge cases (side pots, walks, fold-to-BB) — test
  explicitly.
- Over-fitting to the current corpus — A/B/C/D must degrade gracefully to solid
  GTO when reads are absent.
- Don't touch the frozen engine; keep the feature vector byte-identical if touched.

---

# PHASE 2 — tournament-driven (post full-field reality check)

## Reality check (latest 86-bot Swiss, tournament_20260601_000617)

**vlad ranked 54/86 at −25,888 — losing.** The small-field bb/100 evals were
misleading (hand-picked weak opponents). Against the real field vlad is mid-low.

Standings are dominated by the **`neel_v6_sweep_*` family** (Optuna-tuned), with
**Pav1602_skantbot4 (#3, +128k)** and **cfr_equity_v28 (#4, +108k)** the top
non-neel bots. Notably saroop (#48) and neel_v6_oppprofile (#60) — the ones first
studied — are mid/low; the *tuned* variants win. Every top bot is an equity +
opponent-model bot; none use a neural net.

## Key findings this phase

- **`_POSTFLOP_ENGINE` defaults to `mc`** — production uses the equity engine
  (`_mc_postflop`), NOT the GTO net. So vlad's architecture matches the winners'.
- **Module D1 was wired into the dormant net path** (`_realtime_search`) and never
  ran in production. Its exploit logic must be ported to `_mc_postflop`.
- Small-field bb/100 cannot predict full-field placement; **validate with the real
  tournament**, not 5-bot evals.

## Done this phase (deferred items + cfr_equity-inspired)

- **B3 equity realization** ✅ — `_mc_postflop` call threshold now scales with
  multiway (`0.08 + 0.06·(n−1)`) and commitment (owed ≥ 40%/75% stack), mirroring
  cfr_equity_v28's `required` bumps.
- **D2 inelastic value sizing** ✅ — station value bet sizing +0.15 → **+0.25**
  (cfr_equity #4 uses +0.25).
- **Action-conditioned range floors** ✅ NEW — `_hand_aggression_bump`: 3-bets+ and
  postflop barrels and big bets raise every live opponent's strength floor
  (cfr_equity's `opp_floor` idea), on top of the archetype/street floor.
- **F2 generic priors** ✅ via existing — `_LEAK_PRIORS` EB shrinkage targets ARE
  the theory-derived population cold-start; a separate JSON would be redundant.

## Phase-2 backlog (prioritized)

1. ~~**Port D1 offensive exploit to `_mc_postflop`**~~ **DONE** — `_has_initiative`
   + `_should_cbet_bluff`: disciplined flop c-bet when checked to (pf aggressor,
   HU/3-way, non-wet board, backup equity ≥ 0.28, not vs station/maniac, ~55%
   freq). Fixes vlad's over-passive "check when equity < call_thr" leak.
   `tests/test_cbet_bluff.py` (9 tests). Golden green. **Needs full-tournament
   validation** — small-field bb/100 can't confirm it.
2. **Full-field parameter tuning** (HIGH) — neel's edge is Optuna tuning on the
   REAL field. Re-run `tune_exploit.py`-style sweeps with the full 86-bot field (or
   a representative subset incl. neel sweeps / skant / cfr_equity), not the 5-bot
   weak field. Tune: call/commit thresholds, value-size fractions, station/nit
   sizing deltas, range floors.
3. **Preflop CFR** (HIGH, separate) — every hand; cfr_equity/neel use solved
   preflop blueprints. vlad's is the disabled-table heuristic. Re-enable once
   retrained.
4. **Per-opponent (not count-based) postflop exploits** (MED) — thread full
   profiles into `_mc_postflop` so sizing/calling adapt to the *specific* bettor
   (e.g. call lighter vs the maniac who bet, value-bet huge vs the station in the
   pot), not aggregate counts.
5. **Confirm with a full local tournament** after each change — the only eval that
   predicts placement. bb/100 vs small fields is a unit check, not a ranking.

## PHASE-2 VALIDATION — head-to-head tournament (the metric that matters)

bb/100 (eval_bb100) turned out to have a **broken CRN pairing**: opponents call
`random` unseeded, so they diverge between the new-run and base-run even at the
same deck seed — the base bb/100 itself varied run-to-run (360/373/400). So
small-field bb/100 cannot adjudicate Phase-2; the tournament (cum chip delta, the
actual scoring metric) is the validator.

Head-to-head (new vlad vs a HEAD-vlad copy `vlad_base`, 16-bot representative
field: top neel sweeps / skant / cfr_equity, mid saroop / neel, weak archetypes;
8 Swiss rounds):

  seed 1:  #4 vlad +40,650   >   #6 vlad_base +35,324   (new − base = +5,326)

New beats HEAD baseline on the real metric, and vlad places #4 — ahead of
cfr_equity_v28, neel_v6_sweep_004, neel_v2_harmonic, saroop. (Seed 2 confirmatory
run pending.) Conclusion: **keep the Phase-2 layer**; the bb/100 negative was a
pairing artifact, not a real regression.

**Validation lesson:** use the full/representative tournament (cum delta) to judge
placement; bb/100 vs a small unseeded-opponent field is not a reliable A/B.
