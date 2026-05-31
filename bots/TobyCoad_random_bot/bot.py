"""
Random Bot — uniform random over legal actions.
Serves as a pure chaos baseline; any strategy that can't beat this is broken.
"""

import random

BOT_NAME = "RandomBot"


def decide(game_state: dict) -> dict:
    actions = ["fold", "call"]

    if game_state["can_check"]:
        actions.append("check")

    stack = game_state["your_stack"]
    min_r = game_state["min_raise_to"]
    bet_in = game_state["your_bet_this_street"]
    if stack > 0 and min_r <= stack + bet_in:
        actions.append("raise")
        actions.append("all_in")

    choice = random.choice(actions)

    if choice == "raise":
        # Random raise size between min and 3x min
        max_r = min(min_r * 3, stack + bet_in)
        amount = random.randint(min_r, max(min_r, max_r))
        return {"action": "raise", "amount": amount}

    return {"action": choice}
