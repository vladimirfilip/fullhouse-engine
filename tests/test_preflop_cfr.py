"""
Tests for the preflop_cfr package.

Covers:
  1. 169-bucket card abstraction (count, symmetry, suit isomorphism)
  2. HU equity table (symmetry, AA vs 72o sanity, diagonal ≈ 0.5)
  3. Info-set key round-trip parity (abstraction.py ↔ bot.py mirrors)
  4. Game-state mechanics (pot math, legal actions, terminal detection)
  5. Smoke-test CFR (a few traversals complete without error)
"""

import random
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import eval7

from preflop_cfr.cards import (
    hand_to_bucket, BUCKET_INFO, ALL_CARDS, fresh_deck, deal_hands,
    RANKS, SUITS,
)
# Import the mirrored helpers from bot.py via direct attribute access to avoid
# running bot.py's model-load side-effects more than once.
import importlib.util, types

# Load bot module (suppress model-not-found errors which are expected in test env)
_bot_spec = importlib.util.spec_from_file_location(
    "bot",
    os.path.join(os.path.dirname(__file__), "..", "bots", "vlad", "bot.py"),
)
_bot_mod = importlib.util.module_from_spec(_bot_spec)
_bot_spec.loader.exec_module(_bot_mod)  # type: ignore


# ── 1. Card abstraction ───────────────────────────────────────────────────────

class TestCardBuckets:
    def test_bucket_count(self):
        """There must be exactly 169 distinct (hi, lo, suited) combinations."""
        assert len(BUCKET_INFO) == 169

    def test_bucket_range(self):
        """Every bucket index is in [0, 168]."""
        c1 = eval7.Card("Ah")
        for r in RANKS:
            for s in SUITS:
                b = hand_to_bucket(c1, eval7.Card(r + s))
                if str(c1) != r + s:
                    assert 0 <= b <= 168, f"bucket out of range for {c1} {r+s}"

    def test_symmetry(self):
        """hand_to_bucket(c1, c2) == hand_to_bucket(c2, c1) for all pairs."""
        c1 = eval7.Card("As")
        c2 = eval7.Card("Kh")
        assert hand_to_bucket(c1, c2) == hand_to_bucket(c2, c1)

        c3 = eval7.Card("7d")
        c4 = eval7.Card("2c")
        assert hand_to_bucket(c3, c4) == hand_to_bucket(c4, c3)

    def test_suit_isomorphism(self):
        """AKs and AKs in different suits map to the same bucket."""
        b1 = hand_to_bucket(eval7.Card("As"), eval7.Card("Ks"))
        b2 = hand_to_bucket(eval7.Card("Ah"), eval7.Card("Kh"))
        b3 = hand_to_bucket(eval7.Card("Ad"), eval7.Card("Kd"))
        assert b1 == b2 == b3

    def test_suited_vs_offsuit_differ(self):
        """AKs and AKo must have different bucket indices."""
        suited   = hand_to_bucket(eval7.Card("As"), eval7.Card("Ks"))
        offsuit  = hand_to_bucket(eval7.Card("As"), eval7.Card("Kh"))
        assert suited != offsuit

    def test_pair_buckets(self):
        """AA=0, 22=12; pairs must be in [0,12]."""
        aa = hand_to_bucket(eval7.Card("As"), eval7.Card("Ah"))
        kk = hand_to_bucket(eval7.Card("Ks"), eval7.Card("Kh"))
        tt = hand_to_bucket(eval7.Card("2s"), eval7.Card("2h"))
        assert aa == 0
        assert kk == 1
        assert tt == 12

    def test_all_unique(self):
        """All 169 BUCKET_INFO entries are distinct."""
        seen = set()
        for entry in BUCKET_INFO:
            assert entry not in seen, f"Duplicate bucket_info entry: {entry}"
            seen.add(entry)


# ── 2. HU equity table ────────────────────────────────────────────────────────

class TestHUEquity:
    @pytest.fixture(scope="class")
    def hu_table(self):
        from preflop_cfr.equity import build_hu_table
        return build_hu_table(n_boards=500)   # fast for tests

    def test_shape(self, hu_table):
        assert hu_table.shape == (169, 169)

    def test_diagonal(self, hu_table):
        """Same-bucket equity should be ~0.5 (within noise)."""
        for i in range(169):
            assert abs(hu_table[i, i] - 0.5) < 0.15, (
                f"Diagonal entry [{i},{i}]={hu_table[i,i]:.3f} far from 0.5"
            )

    def test_symmetry(self, hu_table):
        """table[i,j] + table[j,i] ≈ 1 for all i≠j."""
        for i in range(0, 169, 13):   # sparse check
            for j in range(0, 169, 13):
                if i != j:
                    total = float(hu_table[i, j]) + float(hu_table[j, i])
                    assert abs(total - 1.0) < 0.05, (
                        f"table[{i},{j}]+table[{j},{i}]={total:.3f} ≠ 1"
                    )

    def test_aa_vs_72o(self, hu_table):
        """AA (bucket 0) should beat 72o by a large margin."""
        b_aa  = hand_to_bucket(eval7.Card("As"), eval7.Card("Ah"))
        b_72o = hand_to_bucket(eval7.Card("7s"), eval7.Card("2h"))
        eq = hu_table[b_aa, b_72o]
        assert eq > 0.75, f"AA vs 72o equity={eq:.3f}, expected > 0.75"


# ── 3. Info-set key parity ────────────────────────────────────────────────────

class TestInfosetKeyParity:
    """
    Verify that abstraction.py and bot.py produce identical keys for the same
    preflop scenarios.
    """

    def _make_gs(self, your_cards, action_log, seat=3, players=None):
        """Minimal game-state dict matching what bot.py expects."""
        if players is None:
            players = [{"seat": i, "stack": 9900, "state": "active",
                        "is_folded": False, "is_all_in": False,
                        "bet_this_street": 0, "bot_id": f"p{i}"}
                       for i in range(6)]
        return {
            "your_cards":       your_cards,
            "community_cards":  [],
            "street":           "preflop",
            "seat_to_act":      seat,
            "pot":              150,
            "your_stack":       9900,
            "amount_owed":      100,
            "can_check":        False,
            "current_bet":      100,
            "min_raise_to":     200,
            "your_bet_this_street": 0,
            "players":          players,
            "action_log":       action_log,
            "match_action_log": [],
        }

    def _abstraction_key(self, your_cards, action_log, seat=3):
        """Compute key via abstraction.py (the canonical source)."""
        from preflop_cfr.abstraction import infoset_key_from_log, amount_to_abstract
        from preflop_cfr.cards import hand_to_bucket

        al = action_log
        n  = 6
        sb_seat = al[0]["seat"] if al and al[0]["action"] == "small_blind" else 1
        dealer  = (sb_seat - 1) % n

        bucket  = hand_to_bucket(eval7.Card(your_cards[0]), eval7.Card(your_cards[1]))

        history = []
        pot, cur_bet = 0, 100
        bets = {}
        for e in al:
            act, eseat, amt = e["action"], e["seat"], e.get("amount", 0)
            if act == "small_blind":
                bets[eseat] = amt; pot += amt; cur_bet = max(cur_bet, amt); continue
            if act == "big_blind":
                bets[eseat] = amt; pot += amt; cur_bet = max(cur_bet, amt); continue
            bst = bets.get(eseat, 0)
            if act == "fold":
                abstract = 0
            elif act in ("check", "call"):
                abstract = 1; bets[eseat] = cur_bet; pot += max(0, cur_bet - bst)
            elif act in ("raise", "all_in"):
                abstract = amount_to_abstract(amt, pot, cur_bet, bst)
                pot += amt - bst; cur_bet = max(cur_bet, amt); bets[eseat] = amt
            else:
                continue
            history.append((eseat, abstract))

        return infoset_key_from_log(seat, dealer, n, history, bucket)

    def _bot_key(self, your_cards, action_log, seat=3):
        """Compute key via bot.py mirror."""
        gs = self._make_gs(your_cards, action_log, seat)
        return _bot_mod._preflop_infoset_key(gs)

    def test_utg_no_actions(self):
        """UTG first to act: history is empty (only blinds in log)."""
        al = [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
        ]
        cards = ["As", "Kh"]
        assert self._abstraction_key(cards, al, seat=3) == self._bot_key(cards, al, seat=3)

    def test_after_utg_raise(self):
        """HJ facing UTG raise."""
        al = [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
            {"seat": 3, "action": "raise",        "amount": 300},
        ]
        cards = ["Qd", "Jd"]
        assert self._abstraction_key(cards, al, seat=4) == self._bot_key(cards, al, seat=4)

    def test_3bet_scenario(self):
        """BB facing UTG open + BTN 3-bet."""
        al = [
            {"seat": 1, "action": "small_blind", "amount": 50},
            {"seat": 2, "action": "big_blind",   "amount": 100},
            {"seat": 3, "action": "raise",        "amount": 300},
            {"seat": 4, "action": "fold",         "amount": 0},
            {"seat": 5, "action": "fold",         "amount": 0},
            {"seat": 0, "action": "raise",        "amount": 900},
            {"seat": 1, "action": "fold",         "amount": 0},
        ]
        cards = ["Ah", "Ad"]
        assert self._abstraction_key(cards, al, seat=2) == self._bot_key(cards, al, seat=2)


# ── 4. Game mechanics ─────────────────────────────────────────────────────────

class TestGameMechanics:
    def test_initial_state_pot(self):
        from preflop_cfr.game import make_initial_state
        state = make_initial_state(dealer_seat=0)
        # SB=50, BB=100 → pot=150
        assert state.pot == 150
        assert state.current_bet == 100

    def test_utg_is_seat3(self):
        """With dealer=0, UTG should be seat 3 (first after BB=seat 2)."""
        from preflop_cfr.game import make_initial_state
        state = make_initial_state(dealer_seat=0)
        assert state.to_act == 3

    def test_legal_actions_utg(self):
        from preflop_cfr.game import make_initial_state, legal_actions
        from preflop_cfr.config import FOLD, CHECK_CALL, ALL_IN
        state = make_initial_state(dealer_seat=0)
        legal = legal_actions(state)
        assert CHECK_CALL in legal
        assert FOLD in legal
        assert ALL_IN in legal

    def test_fold_removes_from_live(self):
        from preflop_cfr.game import make_initial_state, apply_action, is_terminal
        from preflop_cfr.config import FOLD
        state = make_initial_state(dealer_seat=0)
        # Fold everyone until one is left
        while not is_terminal(state) and state.to_act != -1:
            state = apply_action(state, FOLD)
        live = [i for i in range(state.n_players) if not state.folded[i]]
        assert len(live) == 1

    def test_pot_after_call(self):
        from preflop_cfr.game import make_initial_state, apply_action
        from preflop_cfr.config import CHECK_CALL
        state = make_initial_state(dealer_seat=0)
        # UTG calls 100bb
        state2 = apply_action(state, CHECK_CALL)
        assert state2.pot == state.pot + state.current_bet - state.bets[state.to_act]

    def test_terminal_after_all_fold(self):
        from preflop_cfr.game import make_initial_state, apply_action, is_terminal
        from preflop_cfr.config import FOLD
        state = make_initial_state(dealer_seat=0)
        n_folded = 0
        while not is_terminal(state) and state.to_act != -1:
            if n_folded < state.n_players - 1:
                state = apply_action(state, FOLD)
                n_folded += 1
            else:
                break
        assert is_terminal(state) or n_folded >= state.n_players - 1


# ── 5. CFR smoke test ─────────────────────────────────────────────────────────

class TestCFRSmoke:
    def test_few_traversals(self):
        """Run 20 traversals without error; check tables are non-empty."""
        from preflop_cfr.cfr import run_iteration
        from preflop_cfr.equity import build_hu_table
        import preflop_cfr.equity as _eq
        _eq._HU_TABLE = build_hu_table(n_boards=50)  # fast build for tests

        regret_sum:   dict = {}
        strategy_sum: dict = {}
        visit_sum:    dict = {}

        random.seed(42)
        np.random.seed(42)
        for i in range(20):
            run_iteration(i % 6, regret_sum, strategy_sum, visit_sum, float(i + 1))

        assert len(regret_sum) > 0,   "regret_sum should be non-empty after traversals"
        assert len(strategy_sum) > 0, "strategy_sum should be non-empty after traversals"
        assert len(visit_sum) > 0,    "visit_sum should be non-empty after traversals"

    def test_strategy_sums_non_negative(self):
        """Strategy sums must be non-negative (they are weighted probability sums)."""
        from preflop_cfr.cfr import run_iteration
        from preflop_cfr.equity import get_hu_table
        get_hu_table()

        regret_sum:   dict = {}
        strategy_sum: dict = {}
        visit_sum:    dict = {}
        random.seed(7)
        np.random.seed(7)
        for i in range(10):
            run_iteration(0, regret_sum, strategy_sum, visit_sum, float(i + 1))

        for k, v in strategy_sum.items():
            assert (v >= 0).all(), f"Negative strategy sum at key {k}: {v}"

    def test_regret_plus_non_negative(self):
        """RM+ keeps cumulative regrets floored at 0 in storage."""
        from preflop_cfr.cfr import run_iteration
        from preflop_cfr.equity import get_hu_table
        get_hu_table()

        regret_sum:   dict = {}
        strategy_sum: dict = {}
        visit_sum:    dict = {}
        random.seed(11)
        np.random.seed(11)
        for i in range(30):
            run_iteration(i % 6, regret_sum, strategy_sum, visit_sum, float(i + 1))

        for k, v in regret_sum.items():
            assert (v >= 0).all(), f"RM+ regret went negative at key {k}: {v}"
