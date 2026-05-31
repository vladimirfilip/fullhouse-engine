"""neel_v2 — opponent-modeling upgrade of the neel baseline.

Improvements over v1:
  - Position detection: derives BTN/SB/BB/CO/MP/UTG from action_log blinds
  - Opponent model: classifies each bot as maniac/station/nit/reg using
    the cross-hand match_action_log (rolling 200-entry window)
  - Exploit layer: tighten vs maniacs, value-bet thin vs stations,
    steal aggressively vs nits
  - Board texture: dry/semi-wet/wet → dynamic c-bet frequency
"""

import hashlib
from collections import Counter

try:
    import eval7
except Exception:
    try:
        from treys import Card as _TC, Deck as _TD, Evaluator as _TE

        class _CC:
            def __init__(self, text):
                self.text = str(text)
                self._t = _TC.new(self.text[0] + self.text[1].lower())
            def __str__(self): return self.text
            def __eq__(self, o): return isinstance(o, _CC) and self.text == o.text
            def __hash__(self): return hash(self.text)

        class _CD:
            def __init__(self):
                raw = _TD().cards
                self.cards = [_CC(_TC.int_to_str(c)[0].upper() + _TC.int_to_str(c)[1]) for c in raw]
            def shuffle(self):
                import random; random.shuffle(self.cards)
            def peek(self, n): return self.cards[:n]

        class _CE:
            Card = _CC; Deck = _CD; _ev = _TE()
            _names = {1:"Straight Flush",2:"Four of a Kind",3:"Full House",
                      4:"Flush",5:"Straight",6:"Trips",7:"Two Pair",8:"Pair",9:"High Card"}
            @classmethod
            def evaluate(cls, cards):
                tc = [c._t if isinstance(c, _CC) else _CC(c)._t for c in cards]
                return 7463 - cls._ev.evaluate(tc[2:], tc[:2])
            @classmethod
            def handtype(cls, score):
                return cls._names.get(cls._ev.get_rank_class(7463 - score), "Unknown")

        eval7 = _CE
    except Exception:
        eval7 = None

BOT_NAME = "Neel v2"

RANK_VALUE = {r: i for i, r in enumerate("23456789TJQKA", start=2)}
PREMIUM_PAIRS  = {"A", "K", "Q", "J", "T"}
STRONG_BWAY    = {"AK", "AQ", "AJ", "KQ"}
PLAYABLE_BWAY  = {"AT", "KJ", "KT", "QJ", "QT", "JT"}
SMALL_PAIRS    = {"9", "8", "7", "6", "5", "4", "3", "2"}

BASE = {
    "value_bar_multi":        0.64,
    "value_bar_shorthanded":  0.55,
    "call_add_multi":         0.11,
    "call_add_shorthanded":   0.05,
    "bluff_equity":           0.45,
    "bluff_pos_thresh":       0.62,
    "bluff_freq":             0.06,
    "bluff_bet":              0.45,
    "value_bet":              0.90,
    "facing_raise":           1.05,
    "strong_pre_bet":         1.25,
    "bway_pre_bet":           0.95,
    "steal_pre_bet":          0.55,
    "playable_pos_thresh":    0.58,
    "playable_call_pot":      0.15,
}


# ---------------------------------------------------------------------------
# Opponent model (derived fresh each call from rolling match_action_log)
# ---------------------------------------------------------------------------

def _opp_stats(match_log):
    stats = {}
    for e in match_log:
        bid = e.get("bot_id"); act = e.get("action")
        if not bid or not act: continue
        s = stats.setdefault(bid, {"t": 0, "r": 0, "c": 0, "f": 0})
        s["t"] += 1
        if act in ("raise", "all_in"): s["r"] += 1
        elif act == "call":            s["c"] += 1
        elif act == "fold":            s["f"] += 1
    return stats

def _classify(stats, bot_id):
    s = stats.get(bot_id, {}); t = s.get("t", 0)
    if t < 15: return "unknown"
    aggr = s["r"] / t; fold = s["f"] / t; call = s["c"] / t
    if aggr > 0.38:                      return "maniac"
    if fold > 0.55 and aggr < 0.20:     return "nit"
    if call > 0.45 and aggr < 0.18:     return "station"
    return "reg"


# ---------------------------------------------------------------------------
# Position detection
# ---------------------------------------------------------------------------

def _get_pos(state):
    """Returns (label, score_0_to_1). score=1.0 is BTN (best position)."""
    log  = state.get("action_log", [])
    seat = state["seat_to_act"]
    n    = len(state["players"])
    bb_seat = sb_seat = None
    for e in log:
        a = e.get("action")
        if a == "big_blind":   bb_seat = e["seat"]
        elif a == "small_blind": sb_seat = e["seat"]

    if bb_seat is None:
        # fallback: use seat order
        active = [p for p in state["players"] if not p.get("is_folded")]
        seats  = [p["seat"] for p in active]
        idx    = seats.index(seat) if seat in seats else 0
        return "MP", idx / max(1, len(seats) - 1)

    if n == 2:
        return ("BTN", 1.0) if seat == sb_seat else ("BB", 0.0)

    btn = (bb_seat - 2) % n
    off = (seat - btn) % n

    if n >= 6:
        labels = {0:"BTN",1:"SB",2:"BB",3:"UTG",4:"MP",5:"CO"}
        scores = {0:1.0, 1:0.30, 2:0.15, 3:0.00, 4:0.35, 5:0.75}
    elif n == 5:
        labels = {0:"BTN",1:"SB",2:"BB",3:"MP",4:"CO"}
        scores = {0:1.0, 1:0.30, 2:0.15, 3:0.10, 4:0.75}
    elif n == 4:
        labels = {0:"BTN",1:"SB",2:"BB",3:"CO"}
        scores = {0:1.0, 1:0.30, 2:0.15, 3:0.65}
    else:
        labels = {0:"BTN",1:"SB",2:"BB"}
        scores = {0:1.0, 1:0.50, 2:0.00}

    return labels.get(off, "MP"), scores.get(off, 0.40)


# ---------------------------------------------------------------------------
# Board texture
# ---------------------------------------------------------------------------

def _texture(community_cards):
    """0=dry, 1=semi-wet, 2=wet."""
    if len(community_cards) < 3: return 1
    board = community_cards[:3]
    ranks = [c[0] for c in board]; suits = [c[1] for c in board]
    paired = len(set(ranks)) < 3
    if paired: return 0
    has_fd = max(Counter(suits).values()) >= 2
    rv     = sorted(RANK_VALUE[r] for r in ranks)
    has_sd = (rv[-1] - rv[0]) <= 4
    if has_fd and has_sd: return 2
    if has_fd or has_sd:  return 1
    return 0


# ---------------------------------------------------------------------------
# Equity calculation (unchanged from v1)
# ---------------------------------------------------------------------------

def _active_opp_count(state):
    me = state["seat_to_act"]; c = 0
    for p in state["players"]:
        if p["seat"] != me and not p.get("is_folded") and p.get("stack", 0) >= 0:
            c += 1
    return max(1, c)

def _postflop_equity(state):
    if eval7 is None: return _fallback(state)
    hole  = [eval7.Card(c) for c in state["your_cards"]]
    board = [eval7.Card(c) for c in state["community_cards"]]
    opps  = _active_opp_count(state)
    if len(board) == 5:
        return _river_equity(eval7.evaluate(hole + board))
    trials = 90 if len(board) == 3 else 120
    wins = 0.0; dead = set(hole + board)
    for _ in range(trials):
        deck = eval7.Deck()
        deck.cards = [c for c in deck.cards if c not in dead]
        deck.shuffle()
        draw = deck.peek(2 * opps + (5 - len(board)))
        opp_hands = [draw[i*2:i*2+2] for i in range(opps)]
        runout = board + draw[2*opps:]
        hs = eval7.evaluate(hole + runout)
        os = [eval7.evaluate(oh + runout) for oh in opp_hands]
        best = max(os) if os else -1
        if hs > best:   wins += 1.0
        elif hs == best:
            ties = 1 + sum(1 for s in os if s == hs)
            wins += 1.0 / ties
    return wins / trials

def _fallback(state):
    ranks  = [c[0] for c in state["your_cards"] + state["community_cards"]]
    counts = sorted((ranks.count(r) for r in set(ranks)), reverse=True)
    if counts and counts[0] >= 3: return 0.72
    if counts and counts[0] == 2: return 0.48
    high = max(RANK_VALUE[r] for r in ranks)
    return 0.38 + (high - 10) * 0.03

def _river_equity(score):
    name = str(eval7.handtype(score)).lower()
    if "straight flush" in name or "quads" in name: return 0.96
    if "full house" in name:  return 0.88
    if "flush" in name:       return 0.78
    if "straight" in name:    return 0.72
    if "trips" in name:       return 0.62
    if "two pair" in name:    return 0.52
    if "pair" in name:        return 0.34
    return 0.18

def _pot_odds(owed, pot):
    return 0.0 if owed <= 0 else owed / max(1, pot + owed)

def _raise_to(state, frac):
    stack = state["your_stack"]; cur = state["your_bet_this_street"]
    pot   = max(1, state["pot"])
    tgt   = int(state["current_bet"] + pot * frac)
    tgt   = max(tgt, state["min_raise_to"])
    tgt   = min(tgt, stack + cur)
    if tgt <= cur:
        return {"action": "call"} if state["amount_owed"] else {"action": "check"}
    return {"action": "raise", "amount": tgt}

def _roll(state, salt):
    key = "|".join([salt, str(state.get("hand_id","")), str(state.get("seat_to_act","")),
                    state.get("street",""), ",".join(state.get("your_cards",[])),
                    ",".join(state.get("community_cards",[])),
                    str(len(state.get("action_log",[])))])
    d = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") / float(1 << 64)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def decide(state):
    if state.get("type") == "warmup":
        return {"action": "check"}

    match_log = state.get("match_action_log", [])
    stats     = _opp_stats(match_log)
    pos_label, pos_score = _get_pos(state)

    my_seat = state["seat_to_act"]

    # Classify active opponents
    opp_types = {}
    for p in state["players"]:
        if p["seat"] == my_seat: continue
        if not p.get("is_folded") and p.get("stack", 0) >= 0:
            bid = p.get("bot_id")
            if bid: opp_types[bid] = _classify(stats, bid)

    n_maniac  = sum(1 for t in opp_types.values() if t == "maniac")
    n_station = sum(1 for t in opp_types.values() if t == "station")
    n_nit     = sum(1 for t in opp_types.values() if t == "nit")
    has_maniac     = n_maniac >= 1
    mostly_station = n_station >= 2
    mostly_nit     = n_nit >= 2 and n_maniac == 0

    street = state["street"]
    owed   = state["amount_owed"]
    pot    = max(0, state["pot"])
    stack  = state["your_stack"]

    if stack <= 0:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}

    if street == "preflop":
        return _preflop(state, pos_label, pos_score, has_maniac, mostly_nit)

    equity = _postflop_equity(state)
    opps   = _active_opp_count(state)
    sh     = opps < 3  # shorthanded

    # Opponent-aware value/call bars
    if mostly_station:
        vbar      = 0.52 if sh else 0.58   # value bet thinner vs stations
        cadd      = 0.06 if sh else 0.09
        bfreq     = 0.00                   # never bluff stations
    elif has_maniac:
        vbar      = 0.54 if sh else 0.63   # tighten for value, call wide
        cadd      = 0.12 if sh else 0.16   # they over-bluff → call wider
        bfreq     = 0.02
    elif mostly_nit:
        vbar      = 0.50 if sh else 0.60   # widen value vs tight folds
        cadd      = 0.07 if sh else 0.09
        bfreq     = 0.12                   # nits fold to c-bets more
    else:
        vbar  = BASE["value_bar_shorthanded"] if sh else BASE["value_bar_multi"]
        cadd  = BASE["call_add_shorthanded"]  if sh else BASE["call_add_multi"]
        bfreq = BASE["bluff_freq"]

    # Board texture modulates bluff frequency
    tex = _texture(state.get("community_cards", []))
    if tex == 2:   bfreq *= 0.5   # wet board → less bluff
    elif tex == 0: bfreq *= 1.6   # dry board → more bluff

    call_bar = _pot_odds(owed, pot) + cadd

    if state["can_check"]:
        if equity >= vbar:
            return _raise_to(state, BASE["value_bet"])
        if (equity >= BASE["bluff_equity"]
                and pos_score > BASE["bluff_pos_thresh"]
                and _roll(state, "bluff") < bfreq):
            return _raise_to(state, BASE["bluff_bet"])
        return {"action": "check"}

    if equity >= max(vbar + 0.06, call_bar + 0.18):
        return _raise_to(state, BASE["facing_raise"])
    if equity >= call_bar:
        return {"action": "call"}
    if owed <= max(20, pot * 0.06):
        return {"action": "call"}
    return {"action": "fold"}


# ---------------------------------------------------------------------------
# Preflop
# ---------------------------------------------------------------------------

def _preflop(state, pos_label, pos_score, has_maniac, mostly_nit):
    cards = state["your_cards"]
    r1, r2 = cards[0][0], cards[1][0]
    suited = cards[0][1] == cards[1][1]
    high, low = sorted((r1, r2), key=lambda r: RANK_VALUE[r], reverse=True)
    label = high + low
    pair  = (high == low)
    owed  = state["amount_owed"]
    pot   = max(1, state["pot"])
    stack = state["your_stack"]
    is_late  = pos_label in ("BTN", "CO")
    is_early = pos_label in ("UTG", "LJ")
    is_bb    = pos_label == "BB"

    # Premium pairs — always raise
    if pair and high in PREMIUM_PAIRS:
        return _raise_to(state, BASE["strong_pre_bet"])

    # Strong broadway
    if label in STRONG_BWAY and (suited or pos_score > 0.25):
        return _raise_to(state, BASE["bway_pre_bet"])

    # Small pairs — set-mine; tighter early
    if pair and high in SMALL_PAIRS:
        cap = (0.22 + 0.10 * pos_score) * (0.70 if is_early else 1.0)
        if owed <= pot * cap:
            return {"action": "call"}
        return {"action": "check"} if state["can_check"] else {"action": "fold"}

    # Playable hands
    playable = (
        label in PLAYABLE_BWAY
        or (suited and RANK_VALUE[high] >= 10 and RANK_VALUE[low] >= 8)
        or (suited and RANK_VALUE[high] - RANK_VALUE[low] <= 2 and RANK_VALUE[high] >= 9)
    )
    # BTN/CO: add all suited aces + suited one-gappers
    if is_late:
        playable = playable or (suited and high == "A") \
                            or (suited and RANK_VALUE[high] >= 9 and RANK_VALUE[low] >= 4)

    # Widen steal range vs nits
    steal_thresh = BASE["playable_pos_thresh"] - (0.15 if mostly_nit else 0.0)

    if playable and pos_score > steal_thresh:
        if owed == 0 and _roll(state, "steal") < 0.45:
            return _raise_to(state, BASE["steal_pre_bet"])
        if owed <= pot * BASE["playable_call_pot"]:
            return {"action": "call"}

    # BB defense vs small raise
    if is_bb and owed > 0 and owed <= pot * 0.22 and (playable or pair):
        return {"action": "call"}

    if state["can_check"]: return {"action": "check"}
    if owed <= min(80, stack * 0.025): return {"action": "call"}
    return {"action": "fold"}
