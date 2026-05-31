"""
╔══════════════════════════════════════════════════════════════╗
║              FULLHOUSE — "Equilibrium Opus" v28              ║
║   Bulletproof Profiling + Correct SPR + Correct MC Iters     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import random
import time
import json

import eval7
import numpy as np

BOT_NAME = "Equilibrium Opus v28"
BOT_AVATAR = "robot_1"

# ---------------------------------------------------------------------------
# Engine constants & Globals
# ---------------------------------------------------------------------------

SMALL_BLIND = 50
BIG_BLIND = 100
STARTING_STACK = 10_000

RANK_ORDER = "23456789TJQKA"
RANK_VAL = {r: i + 2 for i, r in enumerate(RANK_ORDER)}
SUITS = "shdc"

_buckets = []
for r1 in RANK_ORDER:
    for r2 in RANK_ORDER:
        if r1 == r2:
            _buckets.append(r1 + r2)
        else:
            hi, lo = (r1, r2) if RANK_VAL[r1] > RANK_VAL[r2] else (r2, r1)
            _buckets.append(hi + lo + "s")
            _buckets.append(hi + lo + "o")
_buckets = sorted(list(set(_buckets)))
HAND_TO_IDX = {b: i for i, b in enumerate(_buckets)}

# ---------------------------------------------------------------------------
# Load Precomputed Data
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

PREFLOP_EQUITY_TABLE = None
POSTFLOP_EQUITY_TABLE = None
CFR_STRATEGY = None
BUCKET_IDX = None

try:
    _pf_path = os.path.join(DATA_DIR, "preflop_equity.npy")
    if os.path.exists(_pf_path): PREFLOP_EQUITY_TABLE = np.load(_pf_path).astype(np.float32)
    
    _post_path = os.path.join(DATA_DIR, "equity_table_v2.npy")
    if os.path.exists(_post_path): POSTFLOP_EQUITY_TABLE = np.load(_post_path).astype(np.float32)
    
    # -> ADDED: Load the compressed CFR dictionary
    _cfr_path = os.path.join(DATA_DIR, "preflop_cfr_strategy.npz")
    if os.path.exists(_cfr_path): CFR_STRATEGY = np.load(_cfr_path)
    
    _bi_path = os.path.join(DATA_DIR, "preflop_buckets.json")
    if os.path.exists(_bi_path):
        with open(_bi_path) as f: BUCKET_IDX = json.load(f)
except Exception: pass

if BUCKET_IDX is None: BUCKET_IDX = HAND_TO_IDX

# ---------------------------------------------------------------------------
# Hand Tracking
# ---------------------------------------------------------------------------

HAND_TRACKER = {}

def track_hand(state):
    hand_id = state["hand_id"]
    if hand_id not in HAND_TRACKER:
        HAND_TRACKER[hand_id] = {
            "raises_preflop": 0, "postflop_aggro_actions": 0,
            "i_was_pf_aggressor": False, "preflop_aggressor": None
        }
        if len(HAND_TRACKER) > 20: del HAND_TRACKER[list(HAND_TRACKER.keys())[0]]
            
    t = HAND_TRACKER[hand_id]
    
    total_raises = 0
    last_raiser = None
    for a in state.get("action_log", []):
        if a.get("action") in ("raise", "all_in"):
            total_raises += 1
            last_raiser = a.get("seat")
            
    if state["street"] == "preflop":
        t["raises_preflop"] = total_raises
        t["preflop_aggressor"] = last_raiser
        t["i_was_pf_aggressor"] = (last_raiser == state["seat_to_act"])
    else:
        t["postflop_aggro_actions"] = max(0, total_raises - t["raises_preflop"])
        
    return t

# ---------------------------------------------------------------------------
# Math & Equity
# ---------------------------------------------------------------------------

def _board_texture_bin(board_strs):
    if len(board_strs) < 3: return 0
    ranks, suits = [RANK_VAL[c[0]] for c in board_strs], [c[1] for c in board_strs]
    paired = len(set(ranks)) < len(ranks)
    flushy = max(suits.count(s) for s in set(suits)) >= 3
    unique_ranks = sorted(set(ranks))
    straighty = False
    if len(unique_ranks) >= 3:
        for i in range(len(unique_ranks) - 2):
            if unique_ranks[i+2] - unique_ranks[i] <= 4: straighty = True; break
        if {14, 2, 3}.issubset(set(unique_ranks)): straighty = True 
    if paired: return 1
    if flushy and straighty: return 4
    if straighty: return 3
    if flushy: return 2
    return 0

def _hand_bucket(card1: str, card2: str) -> str:
    r1, s1, r2, s2 = card1[0], card1[1], card2[0], card2[1]
    v1, v2 = RANK_VAL[r1], RANK_VAL[r2]
    if v1 < v2: r1, r2, s1, s2 = r2, r1, s2, s1
    if RANK_VAL[r1] == RANK_VAL[r2]: return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")

_PREFLOP_MC = {}
_FULL_DECK = [eval7.Card(r + s) for r in RANK_ORDER for s in SUITS]

def preflop_equity(cards):
    bucket = _hand_bucket(cards[0], cards[1])
    idx = BUCKET_IDX.get(bucket)
    if PREFLOP_EQUITY_TABLE is not None and idx is not None:
        return float(PREFLOP_EQUITY_TABLE[idx, 0])
    return _PREFLOP_MC.get(bucket, 0.5)

def _build_preflop_mc(iters_per_rep=500):
    rng = random.Random(0xC0FFEE)
    def reps(b):
        if len(b) == 2: return [(eval7.Card(b[0]+"s"), eval7.Card(b[0]+"h")), (eval7.Card(b[0]+"d"), eval7.Card(b[0]+"c"))]
        if b[2] == "s": return [(eval7.Card(b[0]+"s"), eval7.Card(b[1]+"s")), (eval7.Card(b[0]+"h"), eval7.Card(b[1]+"h"))]
        return [(eval7.Card(b[0]+"s"), eval7.Card(b[1]+"h")), (eval7.Card(b[0]+"d"), eval7.Card(b[1]+"c"))]

    for bucket in _buckets:
        wins, total = 0.0, 0
        for c1, c2 in reps(bucket):
            used = {str(c1), str(c2)}
            pool = [c for c in _FULL_DECK if str(c) not in used]
            for _ in range(iters_per_rep):
                sample = rng.sample(pool, 7)
                my = eval7.evaluate([c1, c2] + sample[2:7])
                opp = eval7.evaluate(sample[:2] + sample[2:7])
                if my > opp: wins += 1
                elif my == opp: wins += 0.5
                total += 1
        _PREFLOP_MC[bucket] = wins / total

def equity_vs_range(hole, board, n_opponents=1, iters=300, deadline=None, opp_min_equity=0.0):
    if not hole: return 0.0
    hole_e, board_e = [eval7.Card(s) for s in hole], [eval7.Card(s) for s in board]
    used = {str(c) for c in hole_e + board_e}
    remaining = [c for c in _FULL_DECK if str(c) not in used]
    needed_total = 2 * n_opponents + (5 - len(board_e))
    if len(remaining) < needed_total: return 0.5

    wins, actual, rng = 0.0, 0, random.Random()
    attempt = 0
    while actual < iters:
        attempt += 1
        if (attempt & 31) == 0 and deadline is not None and time.time() > deadline: break
        sample = rng.sample(remaining, needed_total)
        opp_hands = [sample[2 * j: 2 * j + 2] for j in range(n_opponents)]

        if opp_min_equity > 0.0:
            keep = True
            for oh in opp_hands:
                if preflop_equity([str(oh[0]), str(oh[1])]) < opp_min_equity: keep = False; break
            if not keep:
                if attempt > iters * 12: break
                continue

        full_board = board_e + sample[2 * n_opponents:]
        my_score = eval7.evaluate(hole_e + full_board)
        best_opp = max(eval7.evaluate(oh + full_board) for oh in opp_hands)
        if my_score > best_opp: wins += 1
        elif my_score == best_opp: wins += 0.5
        actual += 1

    return wins / max(actual, 1)

def _lookup_equity(hole, board, n_opps, street, base_iters, deadline, opp_floor):
    if opp_floor > 0.48:
        return equity_vs_range(hole, board, n_opponents=n_opps, iters=base_iters, deadline=deadline, opp_min_equity=opp_floor)
    
    if POSTFLOP_EQUITY_TABLE is not None:
        try:
            h_idx = BUCKET_IDX.get(_hand_bucket(hole[0], hole[1]))
            if h_idx is not None:
                s_idx = {"flop": 0, "turn": 1, "river": 2}.get(street, 0)
                return float(POSTFLOP_EQUITY_TABLE[h_idx, _board_texture_bin(board), s_idx, min(max(1, n_opps) - 1, 2)])
        except Exception: pass
    return equity_vs_range(hole, board, n_opps, base_iters, deadline, opp_floor)

# ---------------------------------------------------------------------------
# Opponent Profiling
# ---------------------------------------------------------------------------

def model_opponents(match_log, my_seat):
    stats, hands = {}, {}
    for a in match_log: hands.setdefault(a.get("hand_num", 0), []).append(a)

    for actions in hands.values():
        seen, vpip, pfr = set(), set(), set()
        for a in actions:
            seat, act = a.get("seat"), a.get("action")
            if seat is None or act is None: continue
            s = stats.setdefault(seat, {"hands": 0, "vpip": 0, "pfr": 0, "aggr": 0, "passive": 0})
            if seat not in seen: seen.add(seat); s["hands"] += 1
            if act in ("call", "raise", "all_in") and seat not in vpip: vpip.add(seat); s["vpip"] += 1
            if act in ("raise", "all_in") and seat not in pfr: pfr.add(seat); s["pfr"] += 1
            if act in ("raise", "all_in"): s["aggr"] += 1
            elif act == "call": s["passive"] += 1

    return {seat: {"vpip": s["vpip"]/max(s["hands"],1), "agg_freq": s["aggr"]/max(s["aggr"]+s["passive"],1), "hands_seen": s["hands"]} 
            for seat, s in stats.items() if seat != my_seat}

def get_aggressor_tag(state, opp_models, analysis):
    last_aggro_seat = next((a.get("seat") for a in reversed(state.get("action_log", [])) if a.get("action") in ("raise", "all_in")), analysis.get("preflop_aggressor"))
    if last_aggro_seat in opp_models:
        stats = opp_models[last_aggro_seat]
        if stats["hands_seen"] >= 15:
            if stats["vpip"] > 0.60 and stats["agg_freq"] > 0.45: return "maniac"
            if stats["vpip"] < 0.22: return "nit"
            if stats["vpip"] > 0.50: return "loose"
    return "normal"

# ---------------------------------------------------------------------------
# Strategy Execution
# ---------------------------------------------------------------------------

def _safe_raise(target_total: int, state) -> dict:
    target_total = max(int(target_total), state["min_raise_to"])
    if target_total >= state["your_stack"] + state["your_bet_this_street"]: return {"action": "all_in"}
    return {"action": "raise", "amount": target_total}

def _fold_or_check(state) -> dict:
    return {"action": "check"} if state["can_check"] else {"action": "fold"}

def get_cfr_key(state, analysis):
    """Maps the table layout to the CFR solver's strict positional keys."""
    action_log = state.get("action_log", [])
    bb_seat = next((a["seat"] for a in action_log if a["action"] == "big_blind"), None)
    if bb_seat is None: return None
    
    # Get active players in action order preflop (UTG first, BB last)
    n = len(state["players"])
    active = []
    for offset in range(1, n + 1):
        s = (bb_seat + offset) % n
        if state["players"][s]["state"] != "busted":
            active.append(s)
            
    my_seat = state["seat_to_act"]
    if my_seat not in active: return None
    
    raises = analysis["raises_preflop"]
    limps = sum(1 for a in action_log if a["action"] == "call" and a.get("amount", 0) <= BIG_BLIND)
    
    # Fallback to exploitative if it's multiway (limped) or 4-bet/5-bet chaos
    if limps > 0 or raises > 1: return None 
    
    my_idx = active.index(my_seat)
    behind = (len(active) - 1) - my_idx
    
    # Opening Spot
    if raises == 0: 
        if behind == 0: return None # Walked to BB, you win anyway
        elif behind == 1: spot = "sb_vs_bb"
        elif behind == 2: spot = "btn_vs_bb"
        elif behind == 3: spot = "co_vs_bb"
        elif behind == 4: spot = "mp_vs_bb"
        else: spot = "ep_vs_bb"
        return f"{spot}____opener"
        
    # Defending against a single raise
    if raises == 1: 
        opener = next((a["seat"] for a in action_log if a["action"] in ("raise", "all_in")), None)
        if opener is None: return None
        op_idx = active.index(opener)
        op_behind = (len(active) - 1) - op_idx
        
        if op_behind == 1: spot = "sb_vs_bb"
        elif op_behind == 2: spot = "btn_vs_bb"
        elif op_behind == 3: spot = "co_vs_bb"
        elif op_behind == 4: spot = "mp_vs_bb"
        else: spot = "ep_vs_bb"
        return f"{spot}__O__defender"
        
    return None

def sample_cfr_action(state, cfr_key):
    """Uses a random number to sample perfectly mixed GTO frequencies."""
    if CFR_STRATEGY is None or cfr_key not in CFR_STRATEGY: return None
    
    cards = state["your_cards"]
    bucket_idx = BUCKET_IDX.get(_hand_bucket(cards[0], cards[1]))
    if bucket_idx is None: return None
    
    strat = CFR_STRATEGY[cfr_key][bucket_idx]
    history = cfr_key.split("__")[1]
    
    actions = ["fold", "raise"] if history == "" else ["fold", "call", "raise"]
    if len(strat) != len(actions): return None
    
    r = random.random()
    cumulative = 0.0
    for i, prob in enumerate(strat):
        cumulative += prob
        if r <= cumulative:
            chosen = actions[i]
            break
    else:
        chosen = actions[-1]
        
    if chosen == "fold":
        return _fold_or_check(state)
    elif chosen == "call":
        return {"action": "check"} if state["can_check"] else {"action": "call"}
    elif chosen == "raise":
        bb = BIG_BLIND
        if history == "": # Open
            position = state["seat_to_act"] / max(len(state["players"]) - 1, 1)
            return _safe_raise(int(bb * (3.0 if position <= 0.4 else 2.5)), state)
        elif history == "O": # 3-bet
            return _safe_raise(state["current_bet"] * 3, state)
    
    return None

def preflop_decision(state, aggro_tag, analysis, opp_models, active_opps):

    # --- ADDED: THE CFR GTO INJECTION ---
    cfr_key = get_cfr_key(state, analysis)
    if cfr_key:
        cfr_action = sample_cfr_action(state, cfr_key)
        if cfr_action:
            return cfr_action

    eq = preflop_equity(state["your_cards"])
    pot, owed, stack, my_bet = state["pot"], state["amount_owed"], state["your_stack"], state["your_bet_this_street"]
    bb, raises_before = BIG_BLIND, analysis["raises_preflop"]
    position = state["seat_to_act"] / max(len(state["players"]) - 1, 1)

    t_open, t_3bet, t_call = 0.52 - (0.04 * position), 0.62, 0.48 - (0.03 * position)

    if aggro_tag == "nit": t_3bet += 0.02; t_call += 0.04
    elif aggro_tag == "maniac": t_3bet -= 0.02; t_call -= 0.04

    # PROACTIVE PROFILING (GUARDED) - Scan active opponents remaining
    if raises_before == 0 and active_opps:
        # Confirmed nits: ONLY steal if EVERY remaining player is known AND a nit.
        nits_only = all(
            p["seat"] in opp_models and 
            opp_models[p["seat"]]["hands_seen"] >= 15 and 
            opp_models[p["seat"]]["vpip"] < 0.25 
            for p in active_opps
        )
        
        # Confirmed maniacs: Tighten if ANY remaining player is known AND a maniac.
        maniac_behind = any(
            p["seat"] in opp_models and 
            opp_models[p["seat"]]["hands_seen"] >= 15 and 
            opp_models[p["seat"]]["vpip"] > 0.50 and 
            opp_models[p["seat"]]["agg_freq"] > 0.45 
            for p in active_opps
        )
        
        if nits_only and len(active_opps) <= 3: t_open -= 0.06
        elif maniac_behind: t_open += 0.04

    if owed >= stack * 0.85:
        required = (owed / max(pot + owed, 1)) + 0.06
        if aggro_tag == "maniac": required -= 0.02
        elif aggro_tag == "nit": required += 0.04
        return {"action": "call"} if eq >= required else _fold_or_check(state)

    if my_bet == bb and not state["can_check"] and raises_before == 1 and owed <= 3 * bb:
        if eq >= t_3bet: return _safe_raise(state["current_bet"] * 3, state)
        return {"action": "call"} if eq >= 0.45 else _fold_or_check(state)

    if raises_before == 0:
        if eq >= t_open:
            limps = sum(1 for a in state.get("action_log", []) if a.get("action") == "call")
            return _safe_raise(int(bb * (3.0 if position <= 0.4 else 2.5) + limps * bb), state)
        if state["can_check"]: return {"action": "check"}
        return {"action": "call"} if eq >= t_call and owed <= 2 * bb and (position < 0.3 or position > 0.5) else {"action": "fold"}

    if raises_before == 1:
        if eq >= t_3bet: return _safe_raise(min(max(int(state["current_bet"] * 3), state["min_raise_to"]), max(int(stack * 0.33) + my_bet, state["min_raise_to"]) if eq < 0.66 else 99999), state)
        if eq >= 0.56 and owed <= stack * 0.10: return {"action": "call"}
        if eq >= t_call and position > 0.4 and owed <= stack * 0.06 and owed <= pot * 0.5: return {"action": "call"}
        return _fold_or_check(state)

    if eq >= 0.80: return _safe_raise(state["current_bet"] * 2.3, state)
    return {"action": "call"} if eq >= 0.66 and owed <= stack * 0.25 else _fold_or_check(state)

def postflop_decision(state, aggro_tag, analysis, deadline, active_opps, opp_models):
    pot, owed, stack, my_bet = state["pot"], state["amount_owed"], state["your_stack"], state["your_bet_this_street"]
    n_opps = max(len(active_opps), 1)
    
    # SPR: use shortest non-all-in stack (stack already reflects chips committed this street)
    non_allin = [p["stack"] for p in active_opps if not p.get("is_all_in")]
    eff_stack = min(non_allin + [stack]) if non_allin else stack
    spr = max(0, eff_stack) / max(pot + owed, 1)

    opp_floor = 0.45 + (analysis["raises_preflop"] * 0.08) + (analysis.get("postflop_aggro_actions", 0) * 0.06)
    if (not state["can_check"]) and owed >= max(pot * 0.5, BIG_BLIND): opp_floor += 0.05
    if (not state["can_check"]) and owed >= stack * 0.85: opp_floor += 0.10
    if aggro_tag == "maniac": opp_floor -= 0.10
    elif aggro_tag == "nit": opp_floor += 0.06
    
    base_iters = {1: 1200, 2: 800}.get(n_opps, 500)
    eq = _lookup_equity(state["your_cards"], state["community_cards"], n_opps, state["street"], base_iters, deadline, max(0.0, min(0.78, opp_floor)))

    if not state["can_check"]:
        required = (owed / max(pot + owed, 1)) + 0.04 + (n_opps - 1) * 0.015
        if owed >= stack * 0.4: required += 0.04
        if owed >= stack * 0.75: required += 0.08

        if eq >= 0.82 and n_opps == 1: return {"action": "all_in"} if spr < 1.5 else _safe_raise(my_bet + pot, state)
        if eq >= 0.68: return _safe_raise(my_bet + pot * 0.75, state)
        return {"action": "call"} if eq >= required else {"action": "fold"}

    size_frac = 0.0
    if eq >= 0.78: 
        if state["street"] == "flop" and random.random() < 0.20: return {"action": "check"}
        size_frac = 0.85 + random.uniform(-0.12, 0.12)
    elif eq >= (0.72 if n_opps >= 3 else 0.65): size_frac = 0.66 + random.uniform(-0.12, 0.12)
    elif eq >= 0.55: size_frac = 0.50 + random.uniform(-0.12, 0.12)

    loose_table = any(opp_models.get(p["seat"], {}).get("vpip", 0) > 0.50 for p in active_opps if p["seat"] in opp_models and opp_models[p["seat"]]["hands_seen"] >= 15)

    if size_frac > 0:
        # INELASTIC SIZING - Scale up value against calling stations
        if loose_table: size_frac = min(1.0, size_frac + 0.25)
        return {"action": "all_in"} if spr <= 1.2 else _safe_raise(my_bet + pot * size_frac, state)

    if state["can_check"] and size_frac == 0.0:
        if state["street"] == "flop" and analysis.get("i_was_pf_aggressor") and n_opps <= 2 and eq >= 0.28:
            if not loose_table and aggro_tag != "maniac": # Never bluff calling stations
                if _board_texture_bin(state["community_cards"]) in (0, 1):
                    if random.random() < 0.55: return _safe_raise(my_bet + pot * random.uniform(0.45, 0.55), state)
                    
    return {"action": "check"}

def decide(state: dict) -> dict:
    if state.get("type") == "warmup":
        try:
            if PREFLOP_EQUITY_TABLE is None: _build_preflop_mc(1000)
        except Exception: pass
        return {"action": "check"}

    if PREFLOP_EQUITY_TABLE is None and not _PREFLOP_MC:
        try: _build_preflop_mc(200)
        except Exception: pass

    try:
        my_seat = state["seat_to_act"]
        opp_models = model_opponents(state.get("match_action_log", []) or [], my_seat)
        active_opps = [p for p in state["players"] if not p.get("is_folded") and p.get("seat") != my_seat]
        analysis = track_hand(state)
        aggro_tag = get_aggressor_tag(state, opp_models, analysis)

        if state["street"] == "preflop":
            return preflop_decision(state, aggro_tag, analysis, opp_models, active_opps)
        return postflop_decision(state, aggro_tag, analysis, time.time() + 1.5, active_opps, opp_models)
    except Exception:
        return _fold_or_check(state)