"""
Fullhouse Hackathon — No-Limit Texas Hold'em Game Engine
6-max, using eval7 (same library as MIT Pokerbots production).

Design principles:
  - Pure logic. No I/O, no bot loading, no subprocess calls.
  - The match orchestrator (sandbox/match.py) calls this.
  - All state is serialisable to JSON for logging and replay.
  - eval7: higher score = stronger hand (opposite of treys).
"""

import eval7
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMALL_BLIND    = 50
BIG_BLIND      = 100
STARTING_STACK = 10_000
MAX_PLAYERS    = 9


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Player:
    seat: int
    bot_id: str
    stack: int
    hole_cards: list = field(default_factory=list)
    is_folded: bool = False
    is_all_in: bool = False
    bet_this_street: int = 0

    @property
    def is_active(self):
        return not self.is_folded and not self.is_all_in

    def to_public_dict(self):
        return {
            "seat": self.seat,
            "bot_id": self.bot_id,
            "stack": self.stack,
            "is_folded": self.is_folded,
            "is_all_in": self.is_all_in,
            "bet_this_street": self.bet_this_street,
            "hole_cards": None,
        }


@dataclass
class Action:
    seat: int
    action: str
    amount: int = 0

    def to_dict(self):
        return {"seat": self.seat, "action": self.action, "amount": self.amount}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PokerEngine:
    def __init__(self, hand_id: str, bot_ids: list,
                 dealer_seat: int = 0,
                 starting_stacks: Optional[dict] = None):
        assert 2 <= len(bot_ids) <= MAX_PLAYERS

        stacks = starting_stacks or {}
        self.players = [
            Player(seat=i, bot_id=bid, stack=stacks.get(bid, STARTING_STACK))
            for i, bid in enumerate(bot_ids)
        ]
        self.n = len(self.players)
        self.dealer_seat = dealer_seat % self.n
        self.hand_id = hand_id

        self.pot = 0
        self.community_cards = []
        self.street = "preflop"
        self.action_log = []
        self.current_bet = 0
        self.min_raise = BIG_BLIND
        self._needs_to_act = set()
        self._deck = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_hand(self) -> dict:
        self._post_blinds()
        self._deal_hole_cards()
        first = self._utg_seat()
        self._set_needs_to_act_except(self._bb_seat())
        return self._build_state(first)

    def apply_action(self, seat: int, raw: dict) -> dict:
        action = self._validate(seat, raw)
        self.action_log.append(action.to_dict())
        self._needs_to_act.discard(seat)
        p = self.players[seat]

        if action.action == "fold":
            p.is_folded = True

        elif action.action == "check":
            pass

        elif action.action == "call":
            owed = self.current_bet - p.bet_this_street
            self._put_in(seat, min(owed, p.stack))
            if p.stack == 0:
                p.is_all_in = True

        elif action.action == "raise":
            chips_needed = action.amount - p.bet_this_street
            prev_bet = self.current_bet
            self._put_in(seat, min(chips_needed, p.stack))
            self.min_raise = self.current_bet - prev_bet
            if p.stack == 0:
                p.is_all_in = True
            # reopen action to everyone except raiser
            self._set_needs_to_act_except(seat)

        elif action.action == "all_in":
            prev_bet = self.current_bet
            self._put_in(seat, p.stack)
            if self.current_bet > prev_bet:
                self.min_raise = max(self.min_raise, self.current_bet - prev_bet)
                self._set_needs_to_act_except(seat)
            p.is_all_in = True

        # check: only one player left (everyone folded)
        remaining = [pl for pl in self.players if not pl.is_folded]
        if len(remaining) == 1:
            return self._award_uncontested(remaining[0])

        return self._advance_if_street_over(seat)

    # ------------------------------------------------------------------
    # Internal: flow
    # ------------------------------------------------------------------

    def _advance_if_street_over(self, last_seat: int) -> dict:
        next_seat = self._next_actor(last_seat)
        if next_seat is not None:
            return self._build_state(next_seat)
        return self._advance_street()

    def _advance_street(self) -> dict:
        for p in self.players:
            p.bet_this_street = 0
        self.current_bet = 0
        self.min_raise = BIG_BLIND

        if self.street == "preflop":
            self.community_cards += self._deck.deal(3)
            self.street = "flop"
        elif self.street == "flop":
            self.community_cards += self._deck.deal(1)
            self.street = "turn"
        elif self.street == "turn":
            self.community_cards += self._deck.deal(1)
            self.street = "river"
        elif self.street == "river":
            return self._showdown()

        first = self._first_postflop_actor()
        if first is None:
            return self._run_it_out()

        self._set_needs_to_act_except(None)
        return self._build_state(first)

    def _run_it_out(self) -> dict:
        while self.street != "river":
            if self.street == "flop":
                self.community_cards += self._deck.deal(1)
                self.street = "turn"
            elif self.street == "turn":
                self.community_cards += self._deck.deal(1)
                self.street = "river"
        return self._showdown()

    # ------------------------------------------------------------------
    # Internal: seat selection
    # ------------------------------------------------------------------

    def _sb_seat(self):
        return (self.dealer_seat + 1) % self.n

    def _bb_seat(self):
        return (self.dealer_seat + 2) % self.n

    def _utg_seat(self):
        """First actor preflop: seat after BB."""
        bb = self._bb_seat()
        for offset in range(1, self.n + 1):
            s = (bb + offset) % self.n
            if self.players[s].is_active:
                return s
        return bb  # fallback heads-up

    def _next_actor(self, from_seat: int) -> Optional[int]:
        if not self._needs_to_act:
            return None
        for offset in range(1, self.n + 1):
            seat = (from_seat + offset) % self.n
            if seat in self._needs_to_act and self.players[seat].is_active:
                return seat
        return None

    def _first_postflop_actor(self) -> Optional[int]:
        for offset in range(1, self.n + 1):
            s = (self.dealer_seat + offset) % self.n
            if self.players[s].is_active:
                return s
        return None

    def _set_needs_to_act_except(self, exclude_seat: Optional[int]):
        self._needs_to_act = {
            p.seat for p in self.players
            if p.is_active and p.seat != exclude_seat
        }

    # ------------------------------------------------------------------
    # Internal: chips
    # ------------------------------------------------------------------

    def _post_blinds(self):
        sb, bb = self._sb_seat(), self._bb_seat()
        self._put_in(sb, min(SMALL_BLIND, self.players[sb].stack))
        self._put_in(bb, min(BIG_BLIND,   self.players[bb].stack))
        self.current_bet = BIG_BLIND
        self.min_raise   = BIG_BLIND
        self.action_log.append({"seat": sb, "action": "small_blind", "amount": SMALL_BLIND})
        self.action_log.append({"seat": bb, "action": "big_blind",   "amount": BIG_BLIND})

    def _put_in(self, seat: int, amount: int):
        amount = max(0, min(amount, self.players[seat].stack))
        self.players[seat].stack -= amount
        self.players[seat].bet_this_street += amount
        self.pot += amount
        if self.players[seat].bet_this_street > self.current_bet:
            self.current_bet = self.players[seat].bet_this_street

    def _deal_hole_cards(self):
        self._deck = eval7.Deck()
        self._deck.shuffle()
        for p in self.players:
            p.hole_cards = self._deck.deal(2)

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    def _validate(self, seat: int, raw: dict) -> Action:
        p = self.players[seat]
        act = str(raw.get("action", "fold")).lower().strip()
        amount = int(raw.get("amount", 0))

        if act not in ("fold", "check", "call", "raise", "all_in"):
            return Action(seat, "fold")

        owed = self.current_bet - p.bet_this_street

        if act == "check" and owed > 0:
            act = "call"

        if act == "raise":
            min_total = self.current_bet + self.min_raise
            amount = max(amount, min_total)
            if (amount - p.bet_this_street) >= p.stack:
                return Action(seat, "all_in", p.stack + p.bet_this_street)
            return Action(seat, "raise", amount)

        if act == "all_in":
            amount = p.stack + p.bet_this_street

        return Action(seat, act, amount)

    # ------------------------------------------------------------------
    # Internal: resolution
    # ------------------------------------------------------------------

    def _showdown(self) -> dict:
        contenders = [p for p in self.players if not p.is_folded]
        if len(contenders) == 1:
            return self._award_uncontested(contenders[0])

        scored = [(eval7.evaluate(p.hole_cards + self.community_cards), p)
                  for p in contenders]
        best = max(s for s, _ in scored)
        winners = [p for s, p in scored if s == best]

        split = self.pot // len(winners)
        remainder = self.pot % len(winners)
        results = []
        for i, w in enumerate(winners):
            award = split + (remainder if i == 0 else 0)
            w.stack += award
            results.append({"bot_id": w.bot_id, "seat": w.seat, "amount": award})

        revealed = {p.bot_id: [str(c) for c in p.hole_cards] for _, p in scored}
        return self._build_result(results, showdown=True, revealed=revealed)

    def _award_uncontested(self, winner: Player) -> dict:
        winner.stack += self.pot
        return self._build_result(
            [{"bot_id": winner.bot_id, "seat": winner.seat, "amount": self.pot}],
            showdown=False
        )

    # ------------------------------------------------------------------
    # Internal: serialisation
    # ------------------------------------------------------------------

    def _build_state(self, seat: int) -> dict:
        p = self.players[seat]
        owed = max(0, self.current_bet - p.bet_this_street)
        return {
            "type": "action_request",
            "hand_id": self.hand_id,
            "street": self.street,
            "seat_to_act": seat,
            "pot": self.pot,
            "community_cards": [str(c) for c in self.community_cards],
            "current_bet": self.current_bet,
            "min_raise_to": self.current_bet + self.min_raise,
            "amount_owed": owed,
            "can_check": owed == 0,
            "your_cards": [str(c) for c in p.hole_cards],
            "your_stack": p.stack,
            "your_bet_this_street": p.bet_this_street,
            "players": [pl.to_public_dict() for pl in self.players],
            "action_log": list(self.action_log),
        }

    def _build_result(self, winners: list, showdown: bool,
                      revealed: Optional[dict] = None) -> dict:
        return {
            "type": "hand_complete",
            "hand_id": self.hand_id,
            "street": self.street,
            "pot": self.pot,
            "community_cards": [str(c) for c in self.community_cards],
            "winners": winners,
            "showdown": showdown,
            "revealed_cards": revealed or {},
            "action_log": list(self.action_log),
            "final_stacks": {p.bot_id: p.stack for p in self.players},
        }
