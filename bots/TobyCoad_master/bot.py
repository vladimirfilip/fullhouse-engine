"""
# STRATEGY NOTES — FullHouseMaster

## The LLM-Convergence Meta
Most competitors at a hackathon will use LLMs (Claude, GPT-4, etc.) to help design
their bots. This creates a strong convergence towards:
  - TAG-adjacent play: tight preflop selection, position-aware
  - Pot-odds decision-making: call when equity > pot_odds
  - "AI-default" bet sizing: 50-66% pot on every street
  - C-bet heavy: fire flop, check-fold turn without improvement

These bots are PREDICTABLE. Predictable bots are exploitable bots.

## Exploitation Plan
1. vs TAG/GTO-bots: Their range is face-up. Steal relentlessly preflop.
   Float their c-bets and take pots away when they check turn. Their
   "give up on turn" pattern is extremely exploitable.

2. vs Passive bots (calling stations): Never bluff. Value-bet every
   street with top pair+. Use small bets (25-33%) to keep them calling.
   They can't fold, so charge them maximum streets.

3. vs Bluff-heavy bots: Widen calling range substantially. Check strong
   hands and let them bluff into us. Raise river with disguised monsters.

4. vs Tight bots: Attack any weakness. Steal blinds 3x normal frequency.
   3-bet their opens to take pots preflop. Fold to their resistance.

## Bet Sizing Philosophy
- Avoid the AI default of 50-60% pot. Mix:
  - 25-33% pot: thin value, probing, keeping fish in
  - 66-75% pot: standard value, charging draws
  - 100%+ pot: polarised river bets, overbets vs capped ranges
- Non-standard sizing is harder for opponents (human or AI) to exploit

## Calling Threshold
Never auto-fold to pressure. Adjust call range based on opponent type:
  - vs aggro: call with equity > pot_odds - 0.05
  - vs tight: call with equity > pot_odds + 0.05
  - vs balanced: call with equity > pot_odds
"""

import eval7
import random

BOT_NAME = "FullHouseMaster"
BOT_AVATAR = "robot_2"

RANKS = "23456789TJQKA"
ALL_CARDS = [r + s for r in RANKS for s in "shdc"]

# ── Preflop ranges ────────────────────────────────────────────────────────────

PREMIUM = {"AA", "KK", "QQ", "JJ"}
STRONG_BROADWAY = {"AKs", "AKo", "AQs", "AQo", "AJs", "ATs", "KQs", "KQo"}
STRONG_PAIRS = {"TT", "99", "88"}
PLAYABLE_IP = {
    "77", "66", "55", "44", "33", "22",
    "AJo", "ATo", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
    "KJs", "KJo", "KTs", "KTo", "K9s",
    "QJs", "QJo", "QTs", "Q9s",
    "JTs", "JTo", "J9s", "J8s",
    "T9s", "T8s", "98s", "97s", "87s", "86s", "76s", "75s", "65s", "54s",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _hand_key(cards):
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    if RANKS.index(r1) < RANKS.index(r2):
        r1, r2, s1, s2 = r2, r1, s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def _preflop_tier(key):
    """4=premium, 3=strong, 2=playable, 1=marginal, 0=trash."""
    if key in PREMIUM:
        return 4
    if key in STRONG_BROADWAY or key in STRONG_PAIRS:
        return 3
    if key in PLAYABLE_IP:
        return 2
    # Any pair
    if len(key) == 2 and key[0] == key[1]:
        return 2
    return 0


def _sb_seat(state):
    for e in state["action_log"]:
        if e["action"] == "small_blind":
            return e["seat"]
    return None


def _in_position(state):
    """True if we act last this street (postflop: SB acts last in HU)."""
    my = state["seat_to_act"]
    sb = _sb_seat(state)
    n  = len(state["players"])
    if sb is None:
        return True
    if n == 2:
        return (my != sb) if state["street"] == "preflop" else (my == sb)
    dealer = (sb - 1) % n
    active = [p["seat"] for p in state["players"] if not p["is_folded"]]
    ordered = [(dealer + 1 + i) % n for i in range(n) if (dealer + 1 + i) % n in active]
    return bool(ordered) and my == ordered[-1]


def _equity(hole, board, n_sims=300):
    """Monte Carlo equity vs random opponent hand."""
    known = set(hole + board)
    deck  = [c for c in ALL_CARDS if c not in known]
    wins  = 0
    need  = 5 - len(board)
    for _ in range(n_sims):
        random.shuffle(deck)
        run = board + deck[:need]
        opp = deck[need:need + 2]
        my  = eval7.evaluate([eval7.Card(c) for c in hole + run])
        op  = eval7.evaluate([eval7.Card(c) for c in opp  + run])
        wins += (1 if my > op else 0.5 if my == op else 0)
    return wins / n_sims


def _hand_strength_class(hole, board):
    """0=air, 1=weak pair, 2=top pair/overpair, 3=two pair/set, 4=monster."""
    if not board:
        return 0
    try:
        score = eval7.evaluate([eval7.Card(c) for c in hole + board])
        htype = eval7.handtype(score)
    except Exception:
        return 0
    if "Four" in htype or "Full" in htype or "Straight Flush" in htype:
        return 4
    if "Straight" in htype or "Flush" in htype:
        return 4
    if "Three" in htype or "Two" in htype:
        return 3
    if "Pair" in htype:
        board_top = max(RANKS.index(c[0]) for c in board)
        hole_high = max(RANKS.index(c[0]) for c in hole)
        # Overpair or top-pair
        if hole_high > board_top:
            return 2
        if any(RANKS.index(c[0]) == board_top for c in board
               if c[0] in {h[0] for h in hole}):
            return 2
        return 1
    return 0


# ── Opponent modelling ────────────────────────────────────────────────────────

def _build_opp_model(state):
    """
    Parse match_action_log to estimate opponent tendencies.
    Returns dict with aggression_rate, fold_rate, call_rate, vpip_estimate.
    """
    log = state.get("match_action_log", [])
    my_seat = state["seat_to_act"]
    opp_acts = [e for e in log if e.get("seat") != my_seat and
                e.get("action") not in ("small_blind", "big_blind")]

    n = len(opp_acts)
    if n < 8:
        return {"type": "unknown", "aggression": 0.30, "fold_rate": 0.30}

    agg   = sum(1 for e in opp_acts if e["action"] in ("raise", "all_in")) / n
    folds = sum(1 for e in opp_acts if e["action"] == "fold") / n
    calls = sum(1 for e in opp_acts if e["action"] == "call") / n

    if agg > 0.45:
        opp_type = "aggro"       # bluff_heavy, LAG
    elif folds > 0.50:
        opp_type = "tight"       # TAG, tight_passive, position_exploiter
    elif calls > 0.55 and agg < 0.10:
        opp_type = "station"     # loose_passive, calling station
    else:
        opp_type = "balanced"    # default TAG/GTO

    return {"type": opp_type, "aggression": agg, "fold_rate": folds, "call_rate": calls}


def _current_hand_reads(state):
    """Quick read of opponent behaviour this hand from action_log."""
    log = state["action_log"]
    my_seat = state["seat_to_act"]
    opp_acts = [e for e in log if e.get("seat") != my_seat and
                e.get("action") not in ("small_blind", "big_blind")]
    raised_pf = any(e["action"] in ("raise", "all_in") for e in opp_acts
                    if state["street"] != "preflop")  # rough proxy
    bet_count = sum(1 for e in opp_acts if e["action"] in ("raise", "all_in", "call"))
    return {"bet_count": bet_count, "raised_pf": raised_pf}


# ── Sizing helpers ────────────────────────────────────────────────────────────

def _raise_action(state, pot_mult, cur_bet_add=True):
    """
    Build a raise action.
    pot_mult: fraction of the pot to bet (e.g. 0.33, 0.75, 1.2).
    """
    pot    = state["pot"]
    cur    = state["current_bet"]
    min_r  = state["min_raise_to"]
    stack  = state["your_stack"]
    bet_in = state["your_bet_this_street"]

    target = int(pot * pot_mult) + (cur if cur_bet_add else 0)
    target = max(target, min_r)
    target = min(target, stack + bet_in)

    chips_needed = target - bet_in
    if chips_needed >= stack:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target}


# ── Preflop strategy ──────────────────────────────────────────────────────────

def _decide_preflop(state, ip, model):
    hole   = state["your_cards"]
    pot    = state["pot"]
    owed   = state["amount_owed"]
    can_chk = state["can_check"]
    key    = _hand_key(hole)
    tier   = _preflop_tier(key)
    otp    = model["type"]

    # ── No raise yet (or we're first in) ──
    if owed <= 100:   # no raise / just a blind
        if tier == 4:   # premium
            # Raise ~3x; against stations raise slightly less to keep them in
            mult = 2.5 if otp == "station" else 3.0
            return _raise_action(state, mult)

        if tier == 3:   # strong
            return _raise_action(state, 2.5)

        if tier == 2 and ip:   # playable IP
            return _raise_action(state, 2.5)

        if tier == 2 and otp == "tight":   # steal vs tight players
            return _raise_action(state, 2.5)

        if can_chk:
            return {"action": "check"}

        # Marginal — complete the blind cheaply or fold
        if owed <= 100:
            return {"action": "call"}
        return {"action": "fold"}

    # ── Facing a raise ──
    raise_size = owed + state["your_bet_this_street"]  # approx

    if tier == 4:
        # 3-bet premium hands always
        return _raise_action(state, 3.0)

    if tier == 3 and ip:
        # 3-bet strong hands IP vs tight opponents (their range is face-up)
        if otp in ("tight", "balanced"):
            return _raise_action(state, 3.0)
        return {"action": "call"}

    if tier == 2 and ip and otp == "tight":
        # Float / 3-bet bluff vs tight raiser IP to steal pot preflop
        if random.random() < 0.4:
            return _raise_action(state, 2.8)
        return {"action": "call"}

    if tier >= 2:
        return {"action": "call"}

    # Trash: fold (or check if free)
    if can_chk:
        return {"action": "check"}
    return {"action": "fold"}


# ── Postflop strategy ─────────────────────────────────────────────────────────

def _decide_postflop(state, ip, model, eq):
    street   = state["street"]
    pot      = state["pot"]
    owed     = state["amount_owed"]
    can_chk  = state["can_check"]
    stack    = state["your_stack"]
    otp      = model["type"]
    strength = _hand_strength_class(state["your_cards"], state["community_cards"])

    pot_odds = owed / (pot + owed) if owed > 0 else 0

    # Effective SPR
    opp_stacks = [p["stack"] for p in state["players"]
                  if p["seat"] != state["seat_to_act"] and not p["is_folded"]]
    eff_stack = min(stack, min(opp_stacks)) if opp_stacks else stack
    spr = eff_stack / max(pot, 1)

    # Call cushion: looser vs aggro (they bluff more), tighter vs tight
    call_cushion = {"aggro": -0.06, "station": 0.0, "tight": 0.06,
                    "balanced": 0.0, "unknown": 0.0}.get(otp, 0.0)

    # ── When checked to (we have initiative or opponent checked) ──
    if can_chk:
        # vs STATION: never bluff, always bet for value with any pair+
        if otp == "station":
            if eq > 0.52:
                # Small bet to keep them calling
                mult = 0.28 if eq < 0.65 else 0.50
                return _raise_action(state, mult)
            return {"action": "check"}

        # vs AGGRO: check strong hands to let them bluff into us
        if otp == "aggro":
            if strength >= 3:
                return {"action": "check"}   # induce bluff
            if eq > 0.55 and ip:
                return _raise_action(state, 0.60)
            return {"action": "check"}

        # vs TIGHT: bet every time in position (they'll fold)
        if otp == "tight" and ip:
            if eq > 0.30:
                return _raise_action(state, 0.65)
            return {"action": "check"}

        # vs BALANCED / default:
        if ip:
            if eq > 0.60:
                # Value bet — mix sizing: thin with medium equity, fat with strong
                mult = 0.33 if eq < 0.68 else (0.75 if eq < 0.80 else 1.1)
                return _raise_action(state, mult)
            if eq > 0.40 and random.random() < 0.35:
                # Semi-bluff / probe
                return _raise_action(state, 0.50)
            return {"action": "check"}
        else:
            # OOP: bet only with clear value
            if eq > 0.65:
                mult = 0.33 if otp == "station" else 0.60
                return _raise_action(state, mult)
            return {"action": "check"}

    # ── Facing a bet ──

    # Adjust call threshold for opponent type
    call_threshold = pot_odds + call_cushion

    # vs STATION: fold more bluffs (they have real hands when they bet)
    if otp == "station":
        if eq > 0.60:
            if strength >= 3 and random.random() < 0.30:
                return _raise_action(state, 0.85)  # raise for value
            return {"action": "call"}
        if eq > pot_odds:
            return {"action": "call"}
        return {"action": "fold"}

    # vs AGGRO: call wider, trap-raise with strong hands
    if otp == "aggro":
        if strength >= 3:
            # Induce: sometimes flat-call, spring on river
            if street == "river" or (strength == 4 and random.random() < 0.5):
                return _raise_action(state, 1.2)
            return {"action": "call"}
        if eq > call_threshold:
            return {"action": "call"}
        # Looser call with draws vs aggro (they're often bluffing)
        if eq > pot_odds - 0.08 and spr > 3:
            return {"action": "call"}
        return {"action": "fold"}

    # vs TIGHT: fold more; they have it when they bet
    if otp == "tight":
        if eq > 0.70:
            return _raise_action(state, 0.85)
        if eq > call_threshold:
            return {"action": "call"}
        return {"action": "fold"}

    # Default (balanced / unknown)
    if eq > 0.70 and strength >= 2:
        if random.random() < 0.35:
            return _raise_action(state, 0.80)
        return {"action": "call"}
    if eq > call_threshold:
        return {"action": "call"}

    # River: non-standard overbet bluff vs capped ranges at low frequency
    if street == "river" and ip and eq < 0.25 and random.random() < 0.18:
        return _raise_action(state, 1.25)

    return {"action": "fold"}


# ── Main entry point ──────────────────────────────────────────────────────────

def decide(game_state: dict) -> dict:
    street = game_state["street"]
    hole   = game_state["your_cards"]
    board  = game_state["community_cards"]

    ip    = _in_position(game_state)
    model = _build_opp_model(game_state)

    if street == "preflop":
        return _decide_preflop(game_state, ip, model)

    # Postflop: compute equity with fewer sims on later streets (faster)
    n_sims = {"flop": 300, "turn": 220, "river": 150}.get(street, 200)
    eq = _equity(hole, board, n_sims)

    return _decide_postflop(game_state, ip, model, eq)
