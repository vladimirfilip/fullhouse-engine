"""

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

from eval7 import Card, evaluate

# ─────────────────────────────────────────────────────────────────────────────

BOT_NAME = "The House"  # Show name on the leaderboard
BOT_AVATAR = "robot_1"  # Chosen in the portal, not here
RANKS = "23456789TJQKA"
SUITS = "shdc"
ALL_CARDS = [Card(r + s) for r in RANKS for s in SUITS]

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

    my_cards: list[Card] = list(map(Card, game_state["your_cards"]))
    board_cards: list[Card] = list(map(Card, game_state["community_cards"]))
    rest_cards = [card for card in ALL_CARDS if card not in my_cards and card not in board_cards]
    amount_owed: int = game_state['amount_owed']
    assert amount_owed == game_state["current_bet"] - game_state["your_bet_this_street"], \
        f"amount_owed: {amount_owed}, current_bet: {game_state['current_bet']}, your_bet_this_street: {game_state['your_bet_this_street']}"
    pot: int = game_state["pot"]
    my_stack: int = game_state["your_stack"]
    street: str = game_state["street"]
    active_players = sum(player["state"] == "active" for player in game_state["players"])
    assert active_players >= 2

    # print(game_state)
    equity = monte_carlo_equity(my_cards, board_cards, rest_cards, active_players - 1)
    return choose_action(equity, pot, amount_owed, game_state["your_bet_this_street"], game_state["min_raise_to"], my_stack, active_players)
    # ─────────────────────────────────────────────────────────────────────────


def monte_carlo_equity(hole_cards: list[str], board_cards: list[str], remaining_cards: list[str], num_opponents=1,
        time_limit=0.5) -> float:
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

        player_score = evaluate(hole_cards + full_board_cards)
        opponent_scores = [evaluate(opp_hole + full_board_cards) for opp_hole in opponent_hole_cards]
        best_score = max(player_score, max(opponent_scores))

        N = opponent_scores.count(best_score)

        if player_score == best_score:
            wins += 1 / (N + 1)

        iterations += 1
    return wins / iterations


def choose_action(mc_equity, pot, amount_owed, already_bet, min_raise_to, your_stack, n_players):
    buffer = 0.1 if n_players == 2 else (0.15 if n_players <= 4 else 0.3)
    required_equity = amount_owed / (pot + amount_owed) if amount_owed > 0 else 0

    if mc_equity < required_equity + buffer:
        if amount_owed == 0:
            return {"action": "check"}
        return {"action": "fold"}
    elif mc_equity > 0.80 or mc_equity > required_equity + buffer + 0.15:
        all_chips = already_bet + your_stack
        if all_chips < pot * 2:  # short stack, just go all-in
            return {"action": "all_in"}
        else:
            raise_amount = max(min_raise_to, already_bet + amount_owed)
            raise_amount = min(raise_amount, all_chips)
            if raise_amount == all_chips:
                return {"action": "all_in"}
            return {"action": "raise", "amount": raise_amount}
    else:
        if amount_owed == 0:
            return {"action": "check"}
        else:
            return {"action": "call"}
