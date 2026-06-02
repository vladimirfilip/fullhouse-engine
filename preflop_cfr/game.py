"""
Lightweight preflop-only game state for tabular CFR.

Mirrors engine/game.py blind/seat semantics and bot.py _abstract_to_raw sizing,
but without I/O, logging, or postflop streets.

Seat assignment: 0=dealer(BTN), 1=SB, 2=BB, 3=UTG, 4=HJ, 5=CO  (6-max)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import eval7

from preflop_cfr import config
from preflop_cfr.cards import fresh_deck, deal_hands
from preflop_cfr.equity import multiway_equity


# ── Equity-realization factors (postflop-aware leaf) ──────────────────────────
# The preflop-only model has no postflop play, so a checked-to-flop leaf would
# otherwise pay every hand its raw all-in equity for free — rewarding cheap
# multiway flops and giving aggression no value (the limp-heavy / loose-blind
# pathology).  We approximate postflop realization at such leaves: equity is
# reweighted toward in-position and the seat with initiative, then renormalised
# so the split stays zero-sum (a valid pot division).  All-in showdowns are NOT
# adjusted — there cards run out and equity is fully realised.
#
# Starting values are a calibration knob (see plan): verify opens raise (not
# limp) and blinds tighten, then tune.
REAL_OOP_FACTOR  = 0.85   # most out-of-position live seat
REAL_IP_FACTOR   = 1.05   # most in-position live seat
REAL_INIT_BONUS  = 0.07   # ×(1+β) for the preflop aggressor, ×(1-β) for callers


# ── Abstract-to-chips sizing (mirrors bot.py _abstract_to_raw, no jitter) ─────

_FRAC_MAP: dict[int, float] = {
    config.BET_0_27X_POT: 0.27,
    config.BET_THIRD_POT: 1.0 / 3.0,
    config.BET_HALF_POT:  0.50,
    config.BET_FULL_POT:  1.00,
    config.BET_1_72X_POT: 1.72,
    config.BET_2X_POT:    2.00,
}


def _abstract_to_chips(action_idx: int, bets_seat: int, stack_seat: int,
                        pot: int, current_bet: int, min_raise: int) -> int:
    """
    Convert an abstract action index to the total-bet-to chip amount for a seat.
    Returns 0 for FOLD/CHECK_CALL.
    """
    if action_idx in (config.FOLD, config.CHECK_CALL):
        return 0

    owed    = max(0, current_bet - bets_seat)
    all_tot = bets_seat + stack_seat
    eff_pot = pot + owed

    if action_idx == config.ALL_IN:
        return all_tot

    frac     = _FRAC_MAP[action_idx]
    raise_by = max(round(eff_pot * frac), min_raise)
    target   = current_bet + raise_by
    if target >= all_tot:
        return all_tot
    return target


# ── Game state ─────────────────────────────────────────────────────────────────

@dataclass
class PreflopState:
    """
    Mutable preflop game state.  Clone before apply_action to preserve the parent.
    """
    n_players:    int
    dealer_seat:  int
    stacks:       list[int]
    bets:         list[int]            # chips committed this street
    total_inv:    list[int]            # total invested this hand (for equity)
    folded:       list[bool]
    all_in:       list[bool]
    pot:          int
    current_bet:  int
    min_raise:    int                  # minimum raise increment
    n_raises:     int
    to_act:       int                  # -1 if round is over
    needs_to_act: set[int]            # seats still needing to act this round
    hands:        list[list[eval7.Card]]
    history:      list[tuple[int, int]]  # (seat, abstract_action_idx), blinds excluded

    def owed(self, seat: int) -> int:
        return max(0, self.current_bet - self.bets[seat])

    def is_active(self, seat: int) -> bool:
        return not self.folded[seat] and not self.all_in[seat] and self.stacks[seat] > 0

    def clone(self) -> "PreflopState":
        return PreflopState(
            n_players    = self.n_players,
            dealer_seat  = self.dealer_seat,
            stacks       = list(self.stacks),
            bets         = list(self.bets),
            total_inv    = list(self.total_inv),
            folded       = list(self.folded),
            all_in       = list(self.all_in),
            pot          = self.pot,
            current_bet  = self.current_bet,
            min_raise    = self.min_raise,
            n_raises     = self.n_raises,
            to_act       = self.to_act,
            needs_to_act = set(self.needs_to_act),
            hands        = self.hands,   # immutable after the deal — share, don't copy
            history      = list(self.history),
        )


# ── Terminal detection ─────────────────────────────────────────────────────────

def is_terminal(state: PreflopState) -> bool:
    live = [i for i in range(state.n_players) if not state.folded[i]]
    if len(live) <= 1:
        return True
    # All live players are all-in (or committed equal amounts, none can act)
    return all(state.all_in[i] or state.stacks[i] == 0 for i in live)


def _last_aggressor_seat(state: PreflopState) -> int:
    """Seat of the last voluntary raise/all-in (initiative), or -1 if none."""
    for seat, action in reversed(state.history):
        if action not in (config.FOLD, config.CHECK_CALL):
            return seat
    return -1


def _realization_weights(state: PreflopState, live: list[int]) -> list[float]:
    """
    Per-live-seat equity-realization weight for a checked-to-flop leaf.

    Combines postflop position (most-IP seat realises most) with initiative
    (the preflop aggressor realises more, pure callers less).  Returned in the
    same order as `live`; the caller renormalises so the pot split is zero-sum.
    """
    n = state.n_players
    # Postflop action order: SB acts first … BTN last.  rank = later = more IP.
    post_rank = {seat: (seat - state.dealer_seat - 1) % n for seat in live}
    ranks_sorted = sorted(post_rank.values())
    lo, hi = ranks_sorted[0], ranks_sorted[-1]
    span = (hi - lo) or 1
    aggressor = _last_aggressor_seat(state)

    weights = []
    for seat in live:
        frac = (post_rank[seat] - lo) / span        # 0 (most OOP) … 1 (most IP)
        w = REAL_OOP_FACTOR + frac * (REAL_IP_FACTOR - REAL_OOP_FACTOR)
        if aggressor != -1:
            w *= (1.0 + REAL_INIT_BONUS) if seat == aggressor \
                else (1.0 - REAL_INIT_BONUS)
        weights.append(w)
    return weights


def terminal_utilities(state: PreflopState) -> list[float]:
    """
    Chip-EV utilities at a terminal node.
    Entry i = expected chip gain/loss for seat i.
    """
    live = [i for i in range(state.n_players) if not state.folded[i]]

    utils = [-float(state.total_inv[i]) for i in range(state.n_players)]

    if len(live) == 1:
        utils[live[0]] += state.pot
        return utils

    live_hands = [state.hands[i] for i in live]
    equities   = multiway_equity(live_hands)

    # All-in showdown → cards run out, equity fully realised, no adjustment.
    # Otherwise ≥1 live seat has chips behind (postflop play the flat model
    # ignores) → reweight equity by realization factors, renormalised to keep
    # the pot split zero-sum.
    all_in_showdown = all(state.all_in[i] or state.stacks[i] == 0 for i in live)
    if not all_in_showdown:
        w = _realization_weights(state, live)
        weighted = [equities[j] * w[j] for j in range(len(live))]
        tot = sum(weighted)
        if tot > 0:
            equities = [x / tot for x in weighted]

    for j, seat in enumerate(live):
        utils[seat] += equities[j] * state.pot

    return utils


# ── Legal actions ──────────────────────────────────────────────────────────────

def legal_actions(state: PreflopState) -> list[int]:
    """
    Return list of legal abstract action indices for state.to_act.
    Mirrors bot.py legal-action enumeration.
    """
    seat  = state.to_act
    owed  = state.owed(seat)
    stack = state.stacks[seat]
    legal = [config.CHECK_CALL]

    if owed > 0:
        legal.append(config.FOLD)

    if stack > 0:
        my_bet  = state.bets[seat]
        all_tot = my_bet + stack
        cur     = state.current_bet
        eff_pot = state.pot + owed

        if state.n_raises < config.MAX_RAISES_PREFLOP:
            last_tgt = -1
            for a_idx, frac in (
                (config.BET_0_27X_POT, 0.27),
                (config.BET_THIRD_POT, 1.0 / 3.0),
                (config.BET_HALF_POT,  0.50),
                (config.BET_FULL_POT,  1.00),
                (config.BET_1_72X_POT, 1.72),
                (config.BET_2X_POT,    2.00),
            ):
                if a_idx not in config.PREFLOP_ACTIONS:
                    continue
                rb  = max(round(eff_pot * frac), state.min_raise)
                tgt = cur + rb
                if tgt < all_tot and tgt != last_tgt:
                    last_tgt = tgt
                    legal.append(a_idx)

        legal.append(config.ALL_IN)

    return legal


# ── Mutable-state apply / undo (used by the shared-mem CFR path) ───────────────

class UndoRecord(NamedTuple):
    """Minimal snapshot to reverse one apply_action_inplace call."""
    seat:            int
    old_stack:       int
    old_bet:         int
    old_total_inv:   int
    old_pot:         int
    old_current_bet: int
    old_folded:      bool
    old_all_in:      bool
    old_n_raises:    int
    old_min_raise:   int
    old_to_act:      int
    old_needs_to_act: frozenset   # restored as set() on undo


def apply_action_inplace(state: PreflopState, action_idx: int) -> UndoRecord:
    """
    Apply action_idx to state **in place** and return an UndoRecord.

    Cheaper than apply_action (no dataclass clone) for the shared-mem
    traversal path where _traverse_shared calls undo_action on the way
    back up.  The public apply_action is unchanged for the dict-based path.
    """
    seat = state.to_act
    undo = UndoRecord(
        seat            = seat,
        old_stack       = state.stacks[seat],
        old_bet         = state.bets[seat],
        old_total_inv   = state.total_inv[seat],
        old_pot         = state.pot,
        old_current_bet = state.current_bet,
        old_folded      = state.folded[seat],
        old_all_in      = state.all_in[seat],
        old_n_raises    = state.n_raises,
        old_min_raise   = state.min_raise,
        old_to_act      = state.to_act,
        old_needs_to_act = frozenset(state.needs_to_act),
    )
    owed = state.owed(seat)
    state.needs_to_act.discard(seat)

    if action_idx == config.FOLD:
        state.folded[seat] = True

    elif action_idx == config.CHECK_CALL:
        if owed > 0:
            paid = min(owed, state.stacks[seat])
            _put_in(state, seat, paid)
            if state.stacks[seat] == 0:
                state.all_in[seat] = True

    else:
        target   = _abstract_to_chips(action_idx, state.bets[seat],
                                      state.stacks[seat], state.pot,
                                      state.current_bet, state.min_raise)
        old_cb   = state.current_bet
        chips_in = target - state.bets[seat]
        _put_in(state, seat, chips_in)
        if state.stacks[seat] == 0:
            state.all_in[seat] = True
        raise_size = state.current_bet - old_cb
        if raise_size >= state.min_raise:
            state.n_raises += 1
            state.min_raise = max(raise_size, config.BIG_BLIND)
            state.needs_to_act = {
                i for i in range(state.n_players)
                if state.is_active(i) and i != seat
            }

    state.history.append((seat, action_idx))
    state.to_act = _next_to_act(state)
    return undo


def undo_action(state: PreflopState, undo: UndoRecord) -> None:
    """Reverse the most recent apply_action_inplace call."""
    seat = undo.seat
    state.stacks[seat]    = undo.old_stack
    state.bets[seat]      = undo.old_bet
    state.total_inv[seat] = undo.old_total_inv
    state.pot             = undo.old_pot
    state.current_bet     = undo.old_current_bet
    state.folded[seat]    = undo.old_folded
    state.all_in[seat]    = undo.old_all_in
    state.n_raises        = undo.old_n_raises
    state.min_raise       = undo.old_min_raise
    state.to_act          = undo.old_to_act
    state.needs_to_act    = set(undo.old_needs_to_act)
    state.history.pop()


# ── Apply action ───────────────────────────────────────────────────────────────

def _put_in(s: PreflopState, seat: int, amount: int):
    amount = max(0, min(amount, s.stacks[seat]))
    s.stacks[seat]    -= amount
    s.bets[seat]      += amount
    s.total_inv[seat] += amount
    s.pot             += amount
    if s.bets[seat] > s.current_bet:
        s.current_bet = s.bets[seat]


def apply_action(state: PreflopState, action_idx: int) -> PreflopState:
    """Return new PreflopState after applying action_idx. Does not mutate state."""
    s    = state.clone()
    seat = s.to_act
    owed = s.owed(seat)

    s.needs_to_act.discard(seat)

    if action_idx == config.FOLD:
        s.folded[seat] = True

    elif action_idx == config.CHECK_CALL:
        if owed > 0:
            paid = min(owed, s.stacks[seat])
            _put_in(s, seat, paid)
            if s.stacks[seat] == 0:
                s.all_in[seat] = True

    else:
        # raise or all-in
        target     = _abstract_to_chips(action_idx, s.bets[seat], s.stacks[seat],
                                         s.pot, s.current_bet, s.min_raise)
        old_cb     = s.current_bet
        chips_in   = target - s.bets[seat]
        _put_in(s, seat, chips_in)
        if s.stacks[seat] == 0:
            s.all_in[seat] = True

        raise_size = s.current_bet - old_cb
        if raise_size >= s.min_raise:
            s.n_raises  += 1
            s.min_raise  = max(raise_size, config.BIG_BLIND)
            # Reopen action for everyone who can still act, except the raiser
            s.needs_to_act = {
                i for i in range(s.n_players)
                if s.is_active(i) and i != seat
            }

    s.history.append((seat, action_idx))
    s.to_act = _next_to_act(s)
    return s


def _next_to_act(s: PreflopState) -> int:
    """Return the next seat that needs to act, or -1 if the round is over."""
    if not s.needs_to_act:
        return -1
    # Find the first seat in needs_to_act going clockwise from current to_act
    n = s.n_players
    for offset in range(1, n + 1):
        seat = (s.to_act + offset) % n
        if seat in s.needs_to_act and s.is_active(seat):
            return seat
    return -1


# ── Factory: initial state ─────────────────────────────────────────────────────

def make_initial_state(
    dealer_seat: int = 0,
    n_players:   int = config.N_PLAYERS,
    stack:       int = config.INITIAL_STACK,
    deck:        Optional[list[eval7.Card]] = None,
) -> PreflopState:
    """Set up initial preflop state: post blinds, deal hands, UTG to act."""
    if deck is None:
        deck = fresh_deck()
    hands = deal_hands(deck, n_players)

    stacks    = [stack] * n_players
    bets      = [0] * n_players
    total_inv = [0] * n_players
    folded    = [False] * n_players
    all_in_f  = [False] * n_players
    pot       = 0

    sb_seat = (dealer_seat + 1) % n_players
    bb_seat = (dealer_seat + 2) % n_players

    sb_amt = min(config.SMALL_BLIND, stacks[sb_seat])
    bb_amt = min(config.BIG_BLIND,   stacks[bb_seat])

    stacks[sb_seat]    -= sb_amt
    bets[sb_seat]       = sb_amt
    total_inv[sb_seat]  = sb_amt
    pot                += sb_amt

    stacks[bb_seat]    -= bb_amt
    bets[bb_seat]       = bb_amt
    total_inv[bb_seat]  = bb_amt
    pot                += bb_amt

    current_bet = bb_amt
    min_raise   = config.BIG_BLIND

    if stacks[sb_seat] == 0:
        all_in_f[sb_seat] = True
    if stacks[bb_seat] == 0:
        all_in_f[bb_seat] = True

    # Everyone must act preflop (including BB who gets the option)
    needs_to_act: set[int] = {
        i for i in range(n_players)
        if not all_in_f[i] and stacks[i] > 0
    }

    # UTG is first to act: first active seat after BB
    utg = (bb_seat + 1) % n_players
    for offset in range(n_players):
        s = (utg + offset) % n_players
        if s in needs_to_act:
            utg = s
            break

    return PreflopState(
        n_players    = n_players,
        dealer_seat  = dealer_seat,
        stacks       = stacks,
        bets         = bets,
        total_inv    = total_inv,
        folded       = folded,
        all_in       = all_in_f,
        pot          = pot,
        current_bet  = current_bet,
        min_raise    = min_raise,
        n_raises     = 0,
        to_act       = utg,
        needs_to_act = needs_to_act,
        hands        = hands,
        history      = [],
    )
