# Contributing

This is the official engine for Fullhouse Hackathon. The engine itself is frozen before the event — we don't want rule changes mid-competition.

## What you CAN contribute

- Bug reports (open an issue)
- Improvements to the local demo UI
- Additional reference bots in `bots/`
- Documentation improvements

## What stays frozen

- `engine/game.py` — the poker rules
- `sandbox/runner.py` — the bot protocol
- `db/schema.sql` — the database schema

## Adding a reference bot

Copy `bots/template/bot.py` into a new folder under `bots/`, implement `decide()`, and open a PR. Good reference bots that demonstrate interesting strategies are welcome.

## Questions

Open an issue or reach out via [fullhousehackathon.com](https://fullhousehackathon.com).
