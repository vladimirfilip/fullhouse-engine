"""
Lightweight preflop-only game state for tabular CFR.

Mirrors engine/game.py blind/seat semantics and bot.py _abstract_to_raw sizing,
but without I/O, logging, or postflop streets.

Seat assignment: 0=dealer(BTN), 1=SB, 2=BB, 3=UTG, 4=HJ, 5=CO  (6-max)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import eval7

from preflop_cfr import config
from preflop_cfr.cards import fresh_deck, deal_hands
from preflop_cfr.equity import multiway_equity


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
