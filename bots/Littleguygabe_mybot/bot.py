import bisect
import os
import json
import traceback
import sys
from typing import *

BOT_NAME = "MyBot"
BOT_AVATAR = "robot_1"

# Create/clear the log file when the bot boots up
LOG_FILE = "bots/mybot/debug.log"
with open(LOG_FILE, "w") as f:
    f.write("--- MATCH START ---\n")

def log(message):
    with open(LOG_FILE, "a") as f:
        f.write(str(message) + "\n")

####### loading json data at import time
RANGES = {}
DATA_DIR = 'bots/mybot/data'
if os.path.exists(DATA_DIR):
    for fn in os.listdir(DATA_DIR):
        if fn.endswith('.json'):
            scenario = fn.split('.')[0]
            with open(os.path.join(DATA_DIR, fn)) as f:
                RANGES[scenario] = json.load(f)

###### STACK_BINS must be ascending for bisect to work
STACK_BINS = [8, 10, 12, 15, 20, 25, 30, 40, 60, 100]

def cards_to_key(cards: List[str]) -> str:
    # Converts ['As', 'Kh'] -> 'AKo', ['8s', '8h'] -> '88'
    ranks = "23456789TJQKA"
    r1, s1 = cards[0][0], cards[0][1]
    r2, s2 = cards[1][0], cards[1][1]
    
    i1, i2 = ranks.index(r1), ranks.index(r2)
    if i1 < i2:
        r1, r2, s1, s2 = r2, r1, s2, s1
        
    if r1 == r2:
        return r1 + r2
    
    suited = "s" if s1 == s2 else "o"
    return r1 + r2 + suited

def get_position_name(game_state: dict):
    my_seat = game_state['seat_to_act']
    num_players = len(game_state['players'])
    try:
        sb_seat = next(a['seat'] for a in game_state['action_log'] if a['action'] == 'small_blind')
    except (StopIteration, KeyError):
        return "Unknown"
    btn_seat = sb_seat if num_players == 2 else (sb_seat - 1) % num_players
    dist = (my_seat - btn_seat) % num_players
    if num_players == 2:
        return {0: "BTN", 1: "BB"}.get(dist, "Unknown")
    return {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}.get(dist, f"Seat_{dist}")

def get_stack_as_bb(game_state):
    bb_amount = next((a['amount'] for a in game_state['action_log'] if a['action'] == 'big_blind'), 100)
    return game_state.get('your_stack', 1) / bb_amount

def floor_to_custom_bin(stack_size):
    index = bisect.bisect_right(STACK_BINS, stack_size) - 1
    if index < 0: return STACK_BINS[0]
    return STACK_BINS[index]

def get_range(pos: str, stack_size: float, scenario: str) -> List[str]:
    stack_bin = floor_to_custom_bin(stack_size)
    scenario_data = RANGES.get(scenario, {})
    bin_data = scenario_data.get(f'{stack_bin}bb', {})
    pos_data = bin_data.get(pos, {})
    return list(pos_data.keys())

def get_preflop_scenario(game_state: dict) -> str:
    actions = game_state.get('action_log', [])
    raises = [a for a in actions if a.get('action') in ('raise', 'all_in')]
    n_raises = len(raises)
    if n_raises == 0: 
        return 'RFI'
    
    # ? Need to find who raised
        # as the range changes depending on who raised
    if n_raises == 1: 
        return 'FRFI'
    
    if n_raises == 2: 
        return '3BET'
    return '3BET_PLUS'


def handle_preflop(game_state) -> dict:
    pos = get_position_name(game_state)
    stack = get_stack_as_bb(game_state)
    scenario = get_preflop_scenario(game_state)
    hand_key = cards_to_key(game_state['your_cards'])
            
    hand_range = get_range(pos, stack, scenario)
    in_range = hand_key in hand_range
            

    # very simple betting logic - if in range raise to 1.5* pot
    pot_size = game_state.get('pot',0)
    bet_size = 1.5 * pot_size

    if in_range:
        log(f"Hand: {hand_key}\t| Pos: {pos}\t| Board: {scenario}\t| Range: {in_range}\t| Stack : {stack}\t| Bet : call \t| Hand ID : {game_state.get('hand_id')}")
        return {'action':'call'}

    else:
        log(f"Hand: {hand_key}\t| Pos: {pos}\t| Board: {scenario}\t| Range: {in_range}\t| Stack : {stack}\t| Bet : fold \t| Hand ID : {game_state.get('hand_id')}")
        return {'action':'fold'}

def decide(game_state: dict) -> dict:
    # ? bet sizing
    # ? implement a proper logger
        # like the one from imc
    # ? maybe vibe code a visualiser from the logger


    try:
        if game_state.get('type') == 'warmup':
            return {"ok": True}

        if game_state.get('street') == 'preflop':
            return handle_preflop(game_state)

        return {"action": "fold"}

    except Exception:
        err = traceback.format_exc()
        log("!!! BOT CRASHED !!!")
        log(err)
        return {"action": "fold"}
    
    

