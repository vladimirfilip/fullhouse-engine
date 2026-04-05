"""
╔══════════════════════════════════════════════════════════════╗
║         FULLHOUSE HACKATHON — BOT TEMPLATE v1.0             ║
║         No-Limit Texas Hold'em, 6-max                        ║
╚══════════════════════════════════════════════════════════════╝

RULES:
  - Implement the decide() function below. That's it.
  - You may import any stdlib module and any library in requirements.txt
  - You may NOT make network calls or read/write files
  - You have 2 seconds to return an action or you auto-fold
  - If your function crashes, it auto-folds for that hand

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
# ─────────────────────────────────────────────────────────────────────────────

BOT_NAME = "MyBot"          # Show name on the leaderboard
BOT_AVATAR = "robot_1"      # Chosen in the portal, not here


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
      seat, bot_id, stack, is_active, is_folded, is_all_in, bet_this_street
    """

    # ── Your strategy goes here ───────────────────────────────────────────────

    # Example: a very basic strategy
    my_cards = game_state["your_cards"]
    amount_owed = game_state["amount_owed"]
    pot = game_state["pot"]
    my_stack = game_state["your_stack"]

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
