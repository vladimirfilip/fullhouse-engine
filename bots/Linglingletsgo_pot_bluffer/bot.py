import random

BOT_NAME = "Benchmark Pot Bluffer"


def decide(state):
    stack = state.get("your_stack", 0)
    already = state.get("your_bet_this_street", 0)
    if state.get("can_check") and random.random() < 0.55:
        amount = min(max(state.get("min_raise_to", 0), state.get("pot", 1)), stack + already)
        return {"action": "all_in"} if amount >= stack + already else {"action": "raise", "amount": int(amount)}
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"} if state.get("amount_owed", 0) <= state.get("pot", 1) * 0.18 else {"action": "fold"}
