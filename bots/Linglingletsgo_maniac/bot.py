import random

BOT_NAME = "Benchmark Maniac"


def decide(state):
    stack = state.get("your_stack", 0)
    already = state.get("your_bet_this_street", 0)
    if stack <= 0:
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}
    if random.random() < 0.68:
        amount = max(state.get("min_raise_to", 0), int(max(300, state.get("pot", 1) * 1.25)))
        amount = min(amount, stack + already)
        return {"action": "all_in"} if amount >= stack + already else {"action": "raise", "amount": amount}
    return {"action": "check"} if state.get("can_check") else {"action": "call"}
