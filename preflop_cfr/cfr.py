"""
External-Sampling MCCFR for the preflop game — CFR+ variant.

Tables (shared across all traversals):
    regret_sum:   int64 key -> float64[N_ACTIONS]   (kept floored at 0 — RM+)
    strategy_sum: int64 key -> float64[N_ACTIONS]   (iteration-weighted average)
    visit_sum:    int64 key -> float                (true visit count, for prune)

One call to run_iteration(traverser, regret_sum, strategy_sum, visit_sum, t, ...)
plays out one complete preflop tree from a freshly-dealt deck:
  - Chance node: concrete cards dealt (external sampling).
  - Opponent nodes: sample one action from the current regret-matched strategy;
    accumulate to strategy_sum weighted by the iteration index t (linear CFR).
  - Traverser nodes: enumerate all legal actions, recurse; accumulate regrets
    with the regret-matching-plus floor (negative cumulative regret reset to 0).

Returns the traverser's expected chip-EV for the root of that traversal.

CFR+ vs vanilla CFR
-------------------
Two changes accelerate convergence by roughly an order of magnitude on a tree
this small:
  1. RM+ regret floor: cumulative regret is clamped to ≥0 after every update, so
     an action that was briefly bad recovers in one good iteration instead of
     waiting for many iterations to climb back from a deep negative.
  2. Linear averaging: the average strategy weights iteration t by t, so the
     better late-iteration strategies dominate the final average.

Performance notes (hot path runs millions of times):
  - Regret matching returns a plain Python list over the *legal* subset, avoiding
    a length-9 np.zeros allocation per node.
  - Opponent actions are sampled with random.random() + a manual cumulative walk
    instead of np.random.choice (which allocates and validates a probability
    vector on every call — ~10–40× slower for a 4-element legal set here).
  - The 64-bit info-set hash is memoized on the canonical (pos, history, bucket)
    tuple, so after warm-up every node is a C-level dict lookup rather than a
    string build + pure-Python FNV byte loop.
  - Hand buckets are computed once per deal (run_iteration), not per node.
"""

from __future__ import annotations

import random

import numpy as np

from preflop_cfr import config
from preflop_cfr.cards import hand_to_bucket, ALL_CARDS
from preflop_cfr.abstraction import infoset_key
from preflop_cfr.game import (
    PreflopState, make_initial_state, is_terminal,
    terminal_utilities, legal_actions, apply_action,
)

N_PLAYERS = config.N_PLAYERS

# Memoize (hero_pos, history_tuple, bucket) -> int64 FNV key.  Bounded by the
# number of distinct info sets (~1e5), so it warms up once and then turns every
# per-node key computation into a single tuple-keyed dict lookup.  Process-local
# (each parallel worker rebuilds it); purely a speed cache, never serialized.
_KEY_CACHE: dict[tuple, int] = {}


# ── Regret matching (RM+) ───────────────────────────────────────────────────────

def _strategy_over_legal(regrets: np.ndarray, legal: list[int]) -> list[float]:
    """
    Current strategy from cumulative regrets, returned as a list aligned to
    `legal`.  Regrets are kept ≥0 in storage (RM+), so this is just a
    normalisation; the max() guards the all-zero (uniform) case.
    """
    pos = [r if r > 0.0 else 0.0 for r in (regrets[a] for a in legal)]
    total = 0.0
    for p in pos:
        total += p
    if total > 0.0:
        inv = 1.0 / total
        return [p * inv for p in pos]
    u = 1.0 / len(legal)
    return [u] * len(legal)


def _get_or_init(table: dict[int, np.ndarray], key: int,
                 base: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """
    Fetch the row for `key`, creating it on first touch.

    When `base` is given (parallel worker warm-start), a missing row is *copied
    from base on first touch* rather than the whole base table being copied up
    front.  A worker therefore only materialises the info sets its chunk actually
    visits — peak per-worker memory drops from "all info sets" to "touched this
    chunk", which lets more workers fit under the same RAM budget.  With base=None
    (single-process master, or the strategy/visit tables) a missing row is zeros.
    """
    v = table.get(key)
    if v is None:
        if base is not None:
            b = base.get(key)
            v = b.copy() if b is not None else np.zeros(config.N_ACTIONS,
                                                         dtype=np.float64)
        else:
            v = np.zeros(config.N_ACTIONS, dtype=np.float64)
        table[key] = v
    return v


def _infoset_key(seat: int, dealer_seat: int, history: list, bucket: int) -> int:
    """Memoized 64-bit info-set key for (position-rel-dealer, history, bucket)."""
    hero_pos = (seat - dealer_seat) % N_PLAYERS
    hist     = tuple(a for _, a in history)
    tk       = (hero_pos, hist, bucket)
    key = _KEY_CACHE.get(tk)
    if key is None:
        key = infoset_key(hero_pos, hist, bucket)
        _KEY_CACHE[tk] = key
    return key


# ── CFR traversal ──────────────────────────────────────────────────────────────

def _traverse(
    state: PreflopState,
    traverser: int,
    regret_sum:   dict[int, np.ndarray],
    strategy_sum: dict[int, np.ndarray],
    visit_sum:    dict[int, float],
    buckets:      list[int],
    weight:       float,
    regret_base:  dict[int, np.ndarray] | None = None,
) -> float:
    """
    Recursive ES-MCCFR (CFR+) traversal.  Returns traverser's EV from this node.

    External sampling: opponent and chance actions are sampled by their own
    probabilities, so the visit frequency already supplies the counterfactual
    reach π₋ᵢ(I).  Regret and average-strategy updates therefore carry NO
    explicit reach weight (the average-strategy iteration weight is a separate,
    deliberate linear-CFR term — not a reach term).

    `regret_base` (parallel worker warm-start, see train._worker_delta): the
    shared regret snapshot this chunk warm-starts from.  Regret rows are lazily
    copied out of it on first touch; single-process passes None (regret_sum is
    the live master).
    """
    if is_terminal(state) or state.to_act == -1:
        # to_act == -1: betting round closed with ≥2 live → equity leaf.
        return terminal_utilities(state)[traverser]

    seat  = state.to_act
    legal = legal_actions(state)
    key   = _infoset_key(seat, state.dealer_seat, state.history, buckets[seat])

    regrets = _get_or_init(regret_sum, key, regret_base)
    probs   = _strategy_over_legal(regrets, legal)   # aligned to `legal`

    if seat != traverser:
        # Opponent node: accumulate the iteration-weighted average strategy and
        # one true visit, then sample a single action to continue down.
        s_entry = _get_or_init(strategy_sum, key)
        for i, a in enumerate(legal):
            s_entry[a] += weight * probs[i]
        visit_sum[key] = visit_sum.get(key, 0.0) + 1.0

        # Manual inverse-CDF sample (no np.random.choice allocation).
        r = random.random()
        cum = 0.0
        chosen = legal[-1]
        for i, a in enumerate(legal):
            cum += probs[i]
            if r <= cum:
                chosen = a
                break
        return _traverse(apply_action(state, chosen), traverser,
                         regret_sum, strategy_sum, visit_sum, buckets, weight,
                         regret_base)

    # Traverser node: enumerate all legal actions, accumulate RM+ regrets.
    action_evs = [_traverse(apply_action(state, a), traverser,
                            regret_sum, strategy_sum, visit_sum, buckets, weight,
                            regret_base)
                  for a in legal]
    node_ev = 0.0
    for i in range(len(legal)):
        node_ev += probs[i] * action_evs[i]
    for i, a in enumerate(legal):
        # RM+: clamp cumulative regret at 0 in storage so it responds in one
        # good iteration instead of climbing back from deep negative.
        v = regrets[a] + (action_evs[i] - node_ev)
        regrets[a] = v if v > 0.0 else 0.0
    return node_ev


def run_iteration(
    traverser:    int,
    regret_sum:   dict[int, np.ndarray],
    strategy_sum: dict[int, np.ndarray],
    visit_sum:    dict[int, float],
    weight:       float,
    dealer_seat:  int = 0,
    regret_base:  dict[int, np.ndarray] | None = None,
) -> float:
    """
    Run one ES-MCCFR (CFR+) traversal for `traverser` from a freshly-dealt game.
    Updates regret_sum, strategy_sum and visit_sum in place.  `weight` is the
    linear-CFR iteration weight applied to the average-strategy accumulation.
    `regret_base` is the optional warm-start snapshot for lazy copy-on-touch in
    parallel workers (None in single-process).
    Returns the traverser's chip-EV estimate for this traversal.
    """
    # Only the hole cards are dealt from this deck (board cards for equity leaves
    # are drawn separately in equity._rollout_equity), so sampling the 2·N needed
    # cards is cheaper than shuffling the full 52-card deck.
    deck  = random.sample(ALL_CARDS, 2 * config.N_PLAYERS)
    state = make_initial_state(dealer_seat=dealer_seat, deck=deck)
    # A4: hands are fixed for the whole traversal — bucket each seat once here
    # rather than re-deriving it (string conversion + dict lookups) per node.
    buckets = [hand_to_bucket(h[0], h[1]) for h in state.hands]
    return _traverse(state, traverser, regret_sum, strategy_sum, visit_sum,
                     buckets, weight, regret_base)
