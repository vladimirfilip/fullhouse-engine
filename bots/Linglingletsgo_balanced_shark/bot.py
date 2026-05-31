BOT_NAME = "Benchmark Balanced Shark"

STRONG = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs", "KQs"}
MEDIUM = {"99", "88", "77", "ATs", "KJs", "QJs", "JTs", "T9s"}


def decide(state):
    hand = _key(state.get("your_cards", []))
    stack = state.get("your_stack", 0)
    already = state.get("your_bet_this_street", 0)
    pot = max(1, state.get("pot", 1))
    if state.get("street") == "preflop":
        if hand in STRONG:
            amount = min(max(state.get("min_raise_to", 0), 550), stack + already)
            return {"action": "raise", "amount": amount} if amount < stack + already else {"action": "all_in"}
        if hand in MEDIUM and state.get("amount_owed", 0) <= pot * 0.3:
            return {"action": "call"} if not state.get("can_check") else {"action": "check"}
        return {"action": "check"} if state.get("can_check") else {"action": "fold"}
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "call"} if state.get("amount_owed", 0) <= pot * 0.28 else {"action": "fold"}


def _key(cards):
    if len(cards) != 2:
        return "72o"
    ranks = "23456789TJQKA"
    a, b = sorted([cards[0][0], cards[1][0]], key=ranks.index, reverse=True)
    if a == b:
        return a + b
    return a + b + ("s" if cards[0][1] == cards[1][1] else "o")
