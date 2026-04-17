# Fullhouse Engine

> The UK's first quantitative poker bot competition — 1 June 2026, One Canada Square, London
> **£3,000 prize pool · Sponsored by Quadrature Capital**

Build a Python bot that plays No-Limit Texas Hold'em. This repo has everything you need to write and test your bot locally — the full game engine, sandbox runner, reference bots, and validator.

---

## How it works

You submit one file — `bot.py` — with one function:

```python
def decide(game_state: dict) -> dict:
    # your entire strategy goes here
    return {"action": "call"}
```

The engine calls `decide()` once per action. You get the full game state — your cards, community cards, pot size, stack sizes, betting history, position. You return one action. That's it.

---

## Getting started

```bash
git clone https://github.com/uzlez/fullhouse-engine
cd fullhouse-engine
pip3 install eval7 flask
python3 demo.py
```

Open `http://localhost:5000` — you'll see 6 reference bots playing each other live with a real-time leaderboard and hand replay.

---

## Writing your bot

Copy the template and edit the `decide()` function:

```bash
cp -r bots/template bots/mybot
# edit bots/mybot/bot.py
```

**Game state your bot receives:**

| Key | Type | Description |
|-----|------|-------------|
| `your_cards` | `list[str]` | Your two hole cards e.g. `["As", "Kh"]` |
| `community_cards` | `list[str]` | Board cards e.g. `["7d", "Tc", "2s"]` |
| `street` | `str` | `preflop` / `flop` / `turn` / `river` |
| `pot` | `int` | Total chips in the pot |
| `your_stack` | `int` | Your remaining chips |
| `amount_owed` | `int` | Chips needed to call (0 = free check) |
| `can_check` | `bool` | True when no bet to call |
| `current_bet` | `int` | Highest bet this street |
| `min_raise_to` | `int` | Minimum legal raise total |
| `players` | `list` | Public info on all seats |
| `action_log` | `list` | Every action taken this hand |

**Valid return values:**

```python
{"action": "fold"}
{"action": "check"}                       # only when can_check is True
{"action": "call"}
{"action": "raise", "amount": 1200}       # amount = total bet, not raise-by
{"action": "all_in"}
```

Invalid or missing actions default to fold. Raises below the minimum are snapped up automatically.

**Rules:**
- 2 seconds to return an action or your bot auto-folds
- No network calls
- No file I/O
- 256 MB RAM, 0.5 CPU core per bot
- Crashes and exceptions auto-fold for that hand — your bot stays in the tournament

**Available libraries:** `eval7` `numpy` `scipy` `treys` — request others before the event

---

## Running matches

```bash
# single match, 200 hands
python3 sandbox/match.py bots/mybot/bot.py bots/shark/bot.py --hands 200

# full tournament simulation (3 Swiss rounds)
# use the demo UI at http://localhost:5000
```

---

## Tournament format

**Day 1 — Qualification**
All bots play in a Swiss-system tournament (3 rounds, 200 hands each). Bots are paired by similar standing after each round. Top 32 qualify.

**Day 2 — Patch window + second qualifier**
Submit an updated bot. Second qualification round runs. Standings update.

**Day 3 — The Finale**
Top 32 bots, live-streamed bracket. Winner takes the prize pool.

---

## Reference bots

Four bots are included to test against:

| Bot | Strategy |
|-----|----------|
| `bots/template/bot.py` | Pocket pairs + basic pot odds |
| `bots/aggressor/bot.py` | Raises constantly regardless of hand |
| `bots/mathematician/bot.py` | Calls only when getting 3:1 pot odds |
| `bots/shark/bot.py` | Tight preflop, position-aware, value bets |

---

## Repo structure

```
engine/         Game engine — NLHE rules, hand evaluation, chip tracking
sandbox/        Validator + local match runner
bots/           Reference bots and starter template
tests/          Engine unit tests
demo.py         Quick local demo
```

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Game engine | Python 3.9+ |
| Hand evaluation | eval7 (same as MIT Pokerbots) |
| Bot isolation | 2s time limit, no network, no file I/O |

---

## Event details

**Fullhouse Hackathon** — 1 June 2026, One Canada Square, London
Prize pool: £3,000 · Lead sponsor: Quadrature Capital

[fullhousehackathon.com](https://fullhousehackathon.com)

---

## Questions

Open an issue or reach out via [fullhousehackathon.com](https://fullhousehackathon.com).
