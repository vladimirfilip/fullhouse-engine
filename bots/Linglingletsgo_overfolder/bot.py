BOT_NAME = "Benchmark Overfolder"


def decide(state):
    if state.get("can_check"):
        return {"action": "check"}
    hand = "".join(sorted([card[0] for card in state.get("your_cards", [])], reverse=True))
    if hand in {"AA", "KK", "QQ"}:
        return {"action": "call"}
    return {"action": "fold"}
