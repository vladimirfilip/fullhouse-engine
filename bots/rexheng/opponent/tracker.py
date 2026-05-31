"""Update OppStats from a hand's action_log. Idempotent on full-log replays."""
from __future__ import annotations

from .stats import OppStats


def replay_action_log(stats_by_seat: dict[int, OppStats], state: dict) -> None:
    """Replay action_log into per-seat stats. Called every decide() but only
    advances state for actions newer than what we've seen.

    State here is the current game_state dict. We use action_log to derive
    what each opponent did this hand, accumulating across hands implicitly
    because state['hand_id'] changes between hands.
    """
    log = state.get("action_log", [])
    n_players = len(state.get("players", []))

    # Reset per-hand counters via marker — track via 'hand_id'
    # Simpler approach: we accumulate every observed action, hand boundaries
    # are detected by hand_id changes in the caller.
    pass  # handled in OpponentModel.observe_state


class OpponentModel:
    """Holds per-seat stats; observes states across a match."""

    def __init__(self) -> None:
        self.stats: dict[int, OppStats] = {}
        self._seen_hand_ids: set = set()
        self._last_action_count_per_hand: dict = {}
        # Per-hand transient state
        self._cur_hand_id = None
        self._cur_n_players = 0
        self._cur_first_aggressor = None
        self._cur_cbet_opportunity_seats = set()
        self._cur_check_raise_opp_seat = None  # who checked, eligible for cr next
        self._cur_voluntary_seats = set()
        self._cur_pfr_seat = None
        self._cur_facing_open_seat = None
        self._cur_three_bettor = None
        self._cur_seen_flop_seats = set()
        self._cur_processed_action_idx = 0

    def _ensure(self, seat: int) -> OppStats:
        if seat not in self.stats:
            self.stats[seat] = OppStats()
        return self.stats[seat]

    def _new_hand(self, hand_id: str, n_players: int) -> None:
        self._cur_hand_id = hand_id
        self._cur_n_players = n_players
        self._cur_first_aggressor = None
        self._cur_cbet_opportunity_seats = set()
        self._cur_check_raise_opp_seat = None
        self._cur_voluntary_seats = set()
        self._cur_pfr_seat = None
        self._cur_facing_open_seat = None
        self._cur_three_bettor = None
        self._cur_seen_flop_seats = set()
        self._cur_processed_action_idx = 0

    def observe_state(self, state: dict, my_seat: int) -> None:
        """Call before decide(). Advances stats based on new actions."""
        hand_id = state.get("hand_id")
        n_players = len(state.get("players", []))
        if hand_id != self._cur_hand_id:
            # Increment hands_seen for each opponent in the new hand
            for seat in range(n_players):
                if seat != my_seat:
                    self._ensure(seat).hands_seen += 1
            self._new_hand(hand_id, n_players)

        log = state.get("action_log", [])
        new_actions = log[self._cur_processed_action_idx:]
        for action in new_actions:
            self._process_action(action, my_seat, state)
        self._cur_processed_action_idx = len(log)

    def _process_action(self, action: dict, my_seat: int, state: dict) -> None:
        seat = action.get("seat")
        atype = action.get("action")
        amount = action.get("amount", 0)
        street = action.get("street", state.get("street"))
        if seat is None or seat == my_seat:
            return
        s = self._ensure(seat)
        s.bot_id = state.get("players", [{}])[seat].get("bot_id", "") if seat < len(state.get("players", [])) else ""
        s.total_actions += 1

        if street == "preflop":
            if atype in ("call", "raise", "all_in"):
                if seat not in self._cur_voluntary_seats:
                    self._cur_voluntary_seats.add(seat)
                    s.vpip_count += 1
            if atype in ("raise", "all_in"):
                if self._cur_pfr_seat is None:
                    self._cur_pfr_seat = seat
                    s.pfr_count += 1
                else:
                    # 3-bet
                    if self._cur_three_bettor is None:
                        self._cur_three_bettor = seat
                        if seat != self._cur_pfr_seat:
                            s.three_bet_count += 1
                            s.facing_open_count += 1
            elif atype == "fold" and self._cur_pfr_seat is not None and self._cur_three_bettor is None:
                # Faced an open and folded — counts as facing-open opportunity
                s.facing_open_count += 1

        # Aggression / call counting (any street)
        if atype in ("raise", "bet", "all_in"):
            s.aggression_actions += 1
            pot_at_bet = max(state.get("pot", 1), 1)
            if amount > 0:
                s.bet_sizes.append(min(amount / pot_at_bet, 4.0))
        elif atype == "call":
            s.call_actions += 1

        # Cbet tracking: PFR aggressor's first action on the flop
        if street == "flop" and seat == self._cur_pfr_seat:
            if seat not in self._cur_cbet_opportunity_seats:
                self._cur_cbet_opportunity_seats.add(seat)
                s.cbet_opp_count += 1
                if atype in ("raise", "bet", "all_in"):
                    s.cbet_count += 1

        # Saw flop / showdown
        if street == "flop":
            self._cur_seen_flop_seats.add(seat)
            if seat not in self._cur_seen_flop_seats:
                s.saw_flop += 1


def archetype(stats: OppStats) -> str:
    """Map stats to one of: nit, TAG, LAG, calling_station, maniac, default."""
    n = stats.hands_seen
    if n < 12:
        return "default"
    vpip = stats.vpip
    pfr = stats.pfr
    af = stats.af
    if vpip > 0.45 and af < 1.0:
        return "calling_station"
    if vpip > 0.40 and af > 2.5:
        return "maniac"
    if vpip < 0.18:
        return "nit"
    if vpip < 0.28 and pfr > 0.18:
        return "TAG"
    if pfr > 0.28:
        return "LAG"
    return "default"


def fingerprint(stats: OppStats) -> str:
    """Bot-origin fingerprint: ai_naive / human_naive / hybrid / serious / unknown."""
    if stats.hands_seen < 20:
        return "unknown"
    if stats.has_bet_size_concentration and stats.cbet_freq > 0.85 and stats.check_raise_freq < 0.05:
        return "ai_naive"
    if stats.cbet_freq > 0.70 and stats.check_raise_freq < 0.05:
        return "hybrid"
    if stats.check_raise_freq > 0.10 or stats.fold_to_cbet_pct < 0.3:
        return "serious"
    return "human_naive"
