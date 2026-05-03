"""
Swiss-tournament pairing + standings + finalist selection.

Pure-Python helpers used by the demo server and any local tournament harness.
The production match queue (BullMQ workers) ships its own copy in the private
ops repo; this module is the canonical reference for participants who want to
run their own multi-bot bake-off locally.
"""


def swiss_pairing(standings, table_size=6):
    """
    Pair bots into tables. `standings` is a list of dicts with bot_id, bot_path,
    cumulative_delta. Returns a list of tables (each table is a list of bot dicts).

    Stragglers (fewer bots left than `table_size`) are folded into the last full
    table rather than playing a short-handed match.
    """
    sorted_bots = sorted(standings, key=lambda b: -b.get("cumulative_delta", 0))

    tables = []
    i = 0
    while i < len(sorted_bots):
        remaining = len(sorted_bots) - i
        if remaining < table_size and tables:
            tables[-1].extend(sorted_bots[i:])
            break
        tables.append(sorted_bots[i:i + table_size])
        i += table_size

    return tables


def compute_standings(all_results):
    """
    `all_results`: list of {bot_id, bot_path, chip_delta}.
    Returns a sorted list of {bot_id, bot_path, cumulative_delta, matches_played},
    highest cumulative_delta first.
    """
    totals = {}
    for r in all_results:
        bid = r["bot_id"]
        if bid not in totals:
            totals[bid] = {
                "bot_id":           bid,
                "bot_path":         r.get("bot_path", ""),
                "cumulative_delta": 0,
                "matches_played":   0,
            }
        totals[bid]["cumulative_delta"] += r["chip_delta"]
        totals[bid]["matches_played"]   += 1

    return sorted(totals.values(), key=lambda b: -b["cumulative_delta"])


def select_finalists(standings, n=64):
    """Top-n bots by cumulative chip delta. Default n=64 matches the qualifier."""
    return standings[:n]
