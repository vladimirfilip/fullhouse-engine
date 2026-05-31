"""Per-seat opponent statistics. Stateless update from action_log replay."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OppStats:
    bot_id: str = ""
    hands_seen: int = 0
    vpip_count: int = 0
    pfr_count: int = 0
    three_bet_count: int = 0
    facing_open_count: int = 0
    cbet_count: int = 0
    cbet_opp_count: int = 0
    fold_to_cbet: int = 0
    fold_to_cbet_opp: int = 0
    aggression_actions: int = 0  # raises + bets
    call_actions: int = 0
    saw_flop: int = 0
    went_to_showdown: int = 0
    total_actions: int = 0
    bet_sizes: list[float] = field(default_factory=list)  # as fraction of pot
    check_raise_count: int = 0
    check_raise_opp_count: int = 0

    @property
    def vpip(self) -> float:
        return self.vpip_count / max(self.hands_seen, 1)

    @property
    def pfr(self) -> float:
        return self.pfr_count / max(self.hands_seen, 1)

    @property
    def three_bet_pct(self) -> float:
        return self.three_bet_count / max(self.facing_open_count, 1)

    @property
    def af(self) -> float:
        return self.aggression_actions / max(self.call_actions, 1)

    @property
    def wtsd(self) -> float:
        return self.went_to_showdown / max(self.saw_flop, 1)

    @property
    def cbet_freq(self) -> float:
        return self.cbet_count / max(self.cbet_opp_count, 1)

    @property
    def fold_to_cbet_pct(self) -> float:
        return self.fold_to_cbet / max(self.fold_to_cbet_opp, 1)

    @property
    def check_raise_freq(self) -> float:
        return self.check_raise_count / max(self.check_raise_opp_count, 1)

    @property
    def has_bet_size_concentration(self) -> bool:
        """Are bet sizes clustered around π·pot, 0.5, 0.66, 0.75? Tells AI-naive."""
        if len(self.bet_sizes) < 4:
            return False
        ai_targets = [0.5, 0.66, 0.75, 1.0, 3.14159 / 4]
        hits = sum(
            1 for s in self.bet_sizes
            if any(abs(s - t) < 0.04 for t in ai_targets)
        )
        return hits / len(self.bet_sizes) > 0.7
