"""
External-Sampling MCCFR for the preflop game.

Tables (shared across all traversals):
    regret_sum:   int64 key -> float64[N_ACTIONS]
    strategy_sum: int64 key -> float64[N_ACTIONS]

One call to run_iteration(traverser, regret_sum, strategy_sum, ...) plays
out one complete preflop tree from a freshly-dealt deck:
  - Chance node: concrete cards dealt (external sampling).
  - Opponent nodes: sample one action from the current regret-matched strategy;
    accumulate to strategy_sum weighted by reach probability.
  - Traverser nodes: enumerate all legal actions, recurse; accumulate regrets.

Returns the traverser's expected chip-EV for the root of that traversal.
"""

from __future__ import annotations

import random

import numpy as np

from preflop_cfr import config
from preflop_cfr.cards import hand_to_bucket, ALL_CARDS
from preflop_cfr.abstraction import infoset_key_from_log
from preflop_cfr.game import (
    PreflopState, make_initial_state, is_terminal,
    terminal_utilities, legal_actions, apply_action,
)


# ── Regret matching ────────────────────────────────────────────────────────────

def _regret_match(regrets: np.ndarray, legal: list[int]) -> np.ndarray:
    """
    Compute current strategy from regret sums over legal actions.
    Returns a probability array of length N_ACTIONS (zeros on illegal actions).
    """
    strat = np.zeros(config.N_ACTIONS, dtype=np.float64)
    pos = np.maximum(regrets[legal], 0.0)
    total = pos.sum()
    if total > 0:
        strat[legal] = pos / total
    else:
        strat[legal] = 1.0 / len(legal)
    return strat


def _get_or_init(table: dict[int, np.ndarray], key: int) -> np.ndarray:
    v = table.get(key)
    if v is None:
        v = np.zeros(config.N_ACTIONS, dtype=np.float64)
        table[key] = v
    return v


# ── CFR traversal ──────────────────────────────────────────────────────────────

def _traverse(
    state: PreflopState,
    traverser: int,
    regret_sum:   dict[int, np.ndarray],
    strategy_sum: dict[int, np.ndarray],
) -> float:
    """
    Recursive ES-MCCFR traversal.  Returns traverser's EV from this node.

    External sampling: opponent and chance actions are sampled by their own
    probabilities, so the visit frequency already supplies the counterfactual
    reach π₋ᵢ(I).  Regret and average-strategy updates therefore carry NO
    explicit reach weight — adding one (as a prior version did) double-counts
    the reach and biases the solution.  This matches canonical ES-MCCFR.
    """
    if is_terminal(state) or state.to_act == -1:
        # to_act == -1: betting round closed with ≥2 live → equity leaf.
        return terminal_utilities(state)[traverser]

    seat    = state.to_act
    legal   = legal_actions(state)
    bucket  = hand_to_bucket(state.hands[seat][0], state.hands[seat][1])
    key     = infoset_key_from_log(
        hero_seat    = seat,
        dealer_seat  = state.dealer_seat,
        n_players    = state.n_players,
        action_history = state.history,
        bucket       = bucket,
    )

    regrets = _get_or_init(regret_sum, key)
    strat   = _regret_match(regrets, legal)

    if seat != traverser:
        # Opponent node: accumulate average strategy (unweighted — see above),
        # then sample a single action to continue down.
        s_entry = _get_or_init(strategy_sum, key)
        for a in legal:
            s_entry[a] += strat[a]

        probs  = np.array([strat[a] for a in legal])
        chosen = legal[int(np.random.choice(len(legal), p=probs / probs.sum()))]
        return _traverse(apply_action(state, chosen), traverser,
                         regret_sum, strategy_sum)

    # Traverser node: enumerate all legal actions, accumulate regrets.
    action_evs = {a: _traverse(apply_action(state, a), traverser,
                               regret_sum, strategy_sum)
                  for a in legal}
    node_ev = sum(strat[a] * action_evs[a] for a in legal)
    for a in legal:
        regrets[a] += action_evs[a] - node_ev
    return node_ev


def run_iteration(
    traverser:    int,
    regret_sum:   dict[int, np.ndarray],
    strategy_sum: dict[int, np.ndarray],
    dealer_seat:  int = 0,
) -> float:
    """
    Run one ES-MCCFR traversal for `traverser` from a freshly-dealt game.
    Updates regret_sum and strategy_sum in place.
    Returns the traverser's chip-EV estimate for this traversal.
    """
    # Only the hole cards are dealt from this deck (board cards for equity leaves
    # are drawn separately in equity._rollout_equity), so sampling the 2·N needed
    # cards is cheaper than shuffling the full 52-card deck.
    deck  = random.sample(ALL_CARDS, 2 * config.N_PLAYERS)
    state = make_initial_state(dealer_seat=dealer_seat, deck=deck)
    return _traverse(state, traverser, regret_sum, strategy_sum)


def visit_counts(regret_sum: dict[int, np.ndarray]) -> dict[int, int]:
    """Return approximate visit count per info-set (sum of abs regrets as proxy)."""
    return {k: int(np.abs(v).sum()) for k, v in regret_sum.items()}
