"""
╔══════════════════════════════════════════════════════════════╗
║         FULLHOUSE HACKATHON — BOT TEMPLATE v1.0             ║
║         No-Limit Texas Hold'em, 6-max                        ║
╚══════════════════════════════════════════════════════════════╝

RULES:
  - Implement the decide() function below. That's it.
  - You may import any stdlib module and any library in requirements.txt
  - You have 2 seconds to return an action or you auto-fold
  - If your function crashes, it auto-folds for that hand

NOT ALLOWED (will DQ your bot):
  - External API calls: no Claude/OpenAI/Anthropic/Google/any HTTP. Network is
    blocked at the container level; trying anyway is a DQ.
  - File writes during gameplay; data/ is read-only and only at import time.
  - subprocess / os.system / shell commands.
  - Threading or async tricks to dodge the 2s/action signal timer.
  - Reflection: __import__('socket'), getattr(__builtins__, 'open'),
    eval(), exec(), compile() — all flagged by the validator.
  - Collusion between bots you've registered with friends — bots must play
    independently; coordinated soft-play or chip-dumping = both DQ'd.
  - Reading other bots' code or hole cards (you can't anyway, but trying = DQ).

OPTIONAL DATA FILES (NEW):
  Submit a .zip archive containing:
    bot.py        (this file, required at root)
    data/         (optional directory with .npz, .pkl, .bin, etc.)

  At module-import time only, you can read from a sibling 'data/' directory:

      import os
      DATA_DIR = os.environ.get("BOT_DATA_DIR",
                                os.path.join(os.path.dirname(__file__), "data"))
      with open(os.path.join(DATA_DIR, "blueprint.npz"), "rb") as f:
          BLUEPRINT = ...load(f)

  Limits:
    - Total submission (bot.py + data/) <= 250 MB
    - data/ alone <= 200 MB
    - bot.py <= 5 MB
    - File access during decide() is blocked at the OS level

CARD FORMAT:
  Cards are strings like "As" (Ace of spades), "Td" (Ten of diamonds)
  Ranks: 2 3 4 5 6 7 8 9 T J Q K A
  Suits: s (spades) h (hearts) d (diamonds) c (clubs)

RETURN FORMAT:
  {"action": "fold"}
  {"action": "check"}          # only valid when amount_owed == 0
  {"action": "call"}
  {"action": "raise", "amount": 1200}   # amount = TOTAL bet, not raise-by
  {"action": "all_in"}

  Invalid actions default to fold. Raises below min_raise_to are snapped up.
"""

# ── You may add imports here ──────────────────────────────────────────────────
import random
import time

import eval7
from eval7 import evaluate

# ─────────────────────────────────────────────────────────────────────────────

BOT_NAME = "The House"          # Show name on the leaderboard
BOT_AVATAR = "robot_1"         # Chosen in the portal, not here
RANKS = "23456789TJKA"
SUITS = "shdc"
ALL_CARDS = [r + s for r in RANKS for s in SUITS]

OPPONENT_PROFILES = dict()


def decide(game_state: dict) -> dict:
    """
    Called once per action. Must return within 2 seconds.

    game_state keys:
      hand_id          str   — unique hand identifier
      street           str   — "preflop" | "flop" | "turn" | "river"
      seat_to_act      int   — your seat number (0-5)
      pot              int   — total chips in pot
      community_cards  list  — e.g. ["As", "Kd", "7h"] (empty preflop)
      current_bet      int   — highest bet on this street
      min_raise_to     int   — minimum legal raise total
      amount_owed      int   — chips you need to put in to call (0 = free check)
      can_check        bool  — True when amount_owed == 0
      your_cards       list  — your two hole cards, e.g. ["Ah", "Kh"]
      your_stack       int   — your remaining chips
      your_bet_this_street int — chips you've already put in this street
      players          list  — public info on all seats (see below)
      action_log       list  — all actions so far this hand

    players[i] keys (public info only, no hole cards):
      seat, bot_id, stack, state, is_folded, is_all_in, bet_this_street
    """

    # ── Your strategy goes here ───────────────────────────────────────────────

    my_cards: list[str] = game_state["your_cards"]
    board_cards: list[str] = game_state["community_cards"]
    rest_cards = [card for card in ALL_CARDS if card not in my_cards and card not in board_cards]
    amount_owed: int = game_state['amount_owed']
    pot: int = game_state["pot"]
    my_stack: int = game_state["your_stack"]
    street: str = game_state["street"]
    active_players = sum(player["state"] == "active" for player in game_state["players"])
    assert active_players >= 2

    update_opponent_profiles(game_state)

    equity = monte_carlo_equity(my_cards, board_cards, rest_cards, num_opponents=active_players - 1)
    current_bet = game_state["current_bet"]
    pot_odds = current_bet / (pot + current_bet)
    if equity > pot_odds:
        return {"action": "call"}

    if game_state["can_check"]:
        return {"action": "check"}

    return {"action": "fold"}

    if active_players == 2:
        return one_on_one(game_state)
    elif street == "preflop":
        return preflop(game_state)
    return gto(game_state)


    # Pocket aces or kings — raise big
    ranks = [c[0] for c in my_cards]
    if ranks.count("A") == 2 or ranks.count("K") == 2:
        raise_to = min(pot * 3, my_stack + game_state["your_bet_this_street"])
        raise_to = max(raise_to, game_state["min_raise_to"])
        return {"action": "raise", "amount": raise_to}

    # Free check — always take it
    if game_state["can_check"]:
        return {"action": "check"}

    # Small price to call — call
    if amount_owed < pot * 0.25:
        return {"action": "call"}

    # Otherwise fold
    return {"action": "fold"}

    # ─────────────────────────────────────────────────────────────────────────

def one_on_one(game_state: dict) -> dict:
    return dict()

def gto(game_state: dict) -> dict:
    return dict()

# Given the game_state, described in topmost comment
# update OPPONENT_PROFILES such that it maps bot_id to a dict storing
# cumulative behaviour characteristics such as VPIP, AggFreq, etc.
def update_opponent_profiles(game_state: dict):
    """
    Track opponent behavior: VPIP (Voluntarily Put In Pot), aggression frequency, etc.
    """
    action_log = game_state.get("action_log", [])

    for action in action_log:
        bot_id = action.get("bot_id")
        if bot_id is None or bot_id == game_state.get("seat_to_act"):
            continue  # Skip if no bot_id or if it's us

        OPPONENT_PROFILES[bot_id] = {
            "hands_seen": 0,
            "vpip_count": 0,
            "raise_count": 0,
            "call_count": 0,
            "fold_count": 0,
            "check_count": 0,
        }

        profile = OPPONENT_PROFILES[bot_id]
        action_type = action.get("action")

        # Track action
        if action_type == "raise" or action_type == "all_in":
            profile["raise_count"] += 1
            profile["vpip_count"] += 1
        elif action_type == "call":
            profile["call_count"] += 1
            profile["vpip_count"] += 1
        elif action_type == "check":
            profile["check_count"] += 1
        elif action_type == "fold":
            profile["fold_count"] += 1

def monte_carlo_equity(hole_cards: list[str], board_cards: list[str], remaining_cards: list[str], num_opponents=1, time_limit=0.1) -> float:
    start_time = time.time()
    wins = 0
    iterations = 0

    while time.time() - start_time < time_limit:
        random.shuffle(remaining_cards)
        opponent_hole_cards: list[list[str]] = []
        for i in range(num_opponents):
            opponent_hole_cards.append([remaining_cards[i * 2], remaining_cards[i * 2 + 1]])

        full_board_cards = board_cards
        for i in range(5 - len(board_cards)):
            full_board_cards.append(remaining_cards[2 * num_opponents + i])

        player_score = evaluate_cards(hole_cards + full_board_cards)
        opponent_scores = [evaluate_cards(opp_hole + full_board_cards) for opp_hole in opponent_hole_cards]
        best_score = max(player_score, max(opponent_scores))

        N = opponent_scores.count(best_score)

        if player_score == best_score:
            wins += 1 / (N + 1)

        iterations += 1
    return wins / iterations


def evaluate_cards(cards: list[str]):
    return evaluate(list(map(lambda s : eval7.Card(s), cards)))

