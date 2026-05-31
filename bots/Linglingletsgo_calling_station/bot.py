BOT_NAME = "Benchmark Calling Station"


def decide(state):
    if state.get("can_check"):
        return {"action": "check"}
    owed = state.get("amount_owed", 0)
    pot = max(1, state.get("pot", 1))
    if owed <= pot * 0.75 or owed <= 300:
        return {"action": "call"}
    return {"action": "fold"}
