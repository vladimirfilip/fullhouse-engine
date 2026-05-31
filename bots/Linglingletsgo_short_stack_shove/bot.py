BOT_NAME = "Benchmark Short Stack Shove"

PLAY = {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AJs", "KQs"}


def decide(state):
    hand = _key(state.get("your_cards", []))
    stack = state.get("your_stack", 0)
    pot = max(1, state.get("pot", 1))
    if hand in PLAY or stack <= pot * 1.4:
        return {"action": "all_in"}
    return {"action": "check"} if state.get("can_check") else {"action": "fold"}


def _key(cards):
    if len(cards) != 2:
        return "72o"
    ranks = "23456789TJQKA"
    a, b = sorted([cards[0][0], cards[1][0]], key=ranks.index, reverse=True)
    if a == b:
        return a + b
    return a + b + ("s" if cards[0][1] == cards[1][1] else "o")
