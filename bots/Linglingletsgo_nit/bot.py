BOT_NAME = "Benchmark Nit"

STRONG = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"}


def decide(state):
    hand = _key(state.get("your_cards", []))
    if state.get("street") == "preflop" and hand in STRONG:
        return {"action": "raise", "amount": min(max(state["min_raise_to"], 450), state["your_stack"] + state["your_bet_this_street"])}
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"} if state.get("amount_owed", 0) <= max(100, state.get("pot", 1) * 0.08) else {"action": "fold"}


def _key(cards):
    if len(cards) != 2:
        return "72o"
    ranks = "23456789TJQKA"
    a, b = sorted([cards[0][0], cards[1][0]], key=ranks.index, reverse=True)
    if a == b:
        return a + b
    return a + b + ("s" if cards[0][1] == cards[1][1] else "o")
