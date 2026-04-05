"""
Fullhouse Match Orchestrator
Runs a multi-hand match between N bots (2-9).

Dev mode (USE_DOCKER=false):  bots run as local subprocesses via runner.py
Prod mode (USE_DOCKER=true):  bots run in isolated Docker containers

The game engine is pure Python — this file handles all I/O and process management.
"""

import json
import subprocess
import sys
import os
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.game import PokerEngine, STARTING_STACK

RUNNER_PATH  = Path(__file__).parent / "runner.py"
SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "fullhouse-sandbox:latest")
USE_DOCKER   = os.environ.get("USE_DOCKER", "false").lower() == "true"
ACTION_TIMEOUT = int(os.environ.get("ACTION_TIMEOUT", "2"))


# ---------------------------------------------------------------------------
# Bot process wrapper
# ---------------------------------------------------------------------------

class BotProcess:
    """
    Wraps one bot running in a subprocess or Docker container.
    Communication: newline-delimited JSON over stdin/stdout.
    """

    def __init__(self, bot_id: str, bot_path: str):
        self.bot_id   = bot_id
        self.bot_path = bot_path
        self.errors   = []
        self._proc    = self._start()

    def _start(self) -> subprocess.Popen:
        if USE_DOCKER:
            cmd = [
                "docker", "run",
                "--rm",                          # delete container on exit
                "-i",                            # keep stdin open
                "--network", "none",             # no internet
                "--memory",  "256m",             # OOM limit
                "--memory-swap", "256m",         # no swap
                "--cpus",    "0.5",              # half a core
                "--read-only",                   # immutable filesystem
                "--no-new-privileges",           # no privilege escalation
                "--user",    "1000:1000",        # non-root
                "--tmpfs",   "/tmp:size=10m",    # tiny writable tmp
                "-v", f"{os.path.abspath(self.bot_path)}:/bot/bot.py:ro",
                "-e", f"ACTION_TIMEOUT={ACTION_TIMEOUT}",
                SANDBOX_IMAGE,
            ]
        else:
            # Dev mode: plain subprocess, same runner.py
            cmd = [sys.executable, "-u", str(RUNNER_PATH)]

        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "BOT_PATH": self.bot_path,
                 "ACTION_TIMEOUT": str(ACTION_TIMEOUT)},
        )

    def act(self, game_state: dict) -> dict:
        """Send state, get action. Returns fold on any failure."""
        try:
            self._proc.stdin.write(json.dumps(game_state) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                raise EOFError("Bot process died")
            action = json.loads(line.strip())
            if "error" in action:
                self.errors.append(action["error"])
            return action
        except Exception as e:
            self.errors.append(str(e))
            return {"action": "fold", "error": str(e)}

    def stderr_lines(self) -> list:
        """Non-blocking stderr drain for logging."""
        lines = []
        try:
            self._proc.stderr.flush()
        except Exception:
            pass
        return lines

    def stop(self):
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def run_match(
    match_id: str,
    bot_paths: dict,          # {bot_id: "/path/to/bot.py"}
    n_hands: int = 200,
    verbose: bool = False,
    seed: int = None,         # set for deterministic/reproducible match
) -> dict:
    """
    Run a complete match. Returns structured result dict with:
      - final chip counts
      - chip delta per bot (positive = profit)
      - full hand-by-hand log
    """
    bot_ids  = list(bot_paths.keys())
    n        = len(bot_ids)
    assert 2 <= n <= 9, f"Need 2-9 bots, got {n}"

    procs    = {bid: BotProcess(bid, path) for bid, path in bot_paths.items()}
    stacks   = {bid: STARTING_STACK for bid in bot_ids}
    hand_log = []
    dealer   = 0
    start_ts = time.time()

    try:
        for hand_num in range(n_hands):
            # Drop busted bots
            alive = [bid for bid in bot_ids if stacks[bid] > 0]
            if len(alive) < 2:
                break

            hand_id = f"{match_id}_h{hand_num:04d}"
            hand_seed = (seed * 1000003 + hand_num) if seed is not None else None
            engine  = PokerEngine(
                hand_id        = hand_id,
                bot_ids        = alive,
                dealer_seat    = dealer % len(alive),
                starting_stacks= {bid: stacks[bid] for bid in alive},
                seed           = hand_seed,
            )

            result  = _play_hand(engine, procs, alive, verbose)
            hand_log.append({"hand_num": hand_num, "hand_id": hand_id, **result})

            for bid, s in result["final_stacks"].items():
                stacks[bid] = s

            dealer += 1

            if verbose and hand_num % 25 == 0:
                _print_stacks(hand_num, n_hands, stacks)

    finally:
        for p in procs.values():
            p.stop()

    return {
        "match_id":     match_id,
        "bot_ids":      bot_ids,
        "n_hands":      len(hand_log),
        "duration_s":   round(time.time() - start_ts, 2),
        "final_stacks": stacks,
        "chip_delta":   {bid: stacks[bid] - STARTING_STACK for bid in bot_ids},
        "bot_errors":   {bid: procs[bid].errors for bid in bot_ids},
        "hands":        hand_log,
    }


def _play_hand(engine: PokerEngine, procs: dict,
               active_bots: list, verbose: bool) -> dict:
    state = engine.start_hand()
    steps = 0

    while state.get("type") == "action_request":
        seat   = state["seat_to_act"]
        bot_id = active_bots[seat]
        action = procs[bot_id].act(state)

        if verbose:
            print(f"  [{bot_id}] {action}", file=sys.stderr)

        state  = engine.apply_action(seat, action)
        steps += 1

        if steps > 1000:
            # Should never happen — safety valve
            raise RuntimeError(f"Hand exceeded 1000 steps: {engine.hand_id}")

    return state


def _print_stacks(hand_num: int, total: int, stacks: dict):
    print(f"\n  === Hand {hand_num}/{total} ===", file=sys.stderr)
    for bid, s in sorted(stacks.items(), key=lambda x: -x[1]):
        bar = "█" * (s // 1000)
        print(f"  {bid:20s} {s:7,}  {bar}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entrypoint for local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, glob

    parser = argparse.ArgumentParser(description="Run a Fullhouse match locally")
    parser.add_argument("bots", nargs="+", help="Paths to bot.py files")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output result as JSON (for worker)")
    parser.add_argument("--match-id", default=None)
    args = parser.parse_args()

    paths = {}
    for i, path in enumerate(args.bots):
        bot_id = Path(path).parent.name or f"bot_{i}"
        paths[bot_id] = path

    match_id = args.match_id or os.environ.get("MATCH_ID") or f"local_{uuid.uuid4().hex[:8]}"

    if not args.json:
        print(f"Starting match {match_id} with {len(paths)} bots, {args.hands} hands\n")

    result = run_match(match_id, paths, n_hands=args.hands, verbose=args.verbose)

    if args.json:
        # Clean structured output for the worker to parse
        print(json.dumps({
            "match_id":     result["match_id"],
            "n_hands":      result["n_hands"],
            "duration_s":   result["duration_s"],
            "final_stacks": result["final_stacks"],
            "chip_delta":   result["chip_delta"],
            "bot_errors":   result["bot_errors"],
        }))
        sys.exit(0)

    print(f"\n{'='*50}")
    print(f"Match complete in {result['duration_s']}s")
    print(f"{'='*50}")
    print(f"{'Bot':<25} {'Final Stack':>12} {'Delta':>10}")
    print("-" * 50)
    for bid in sorted(result["bot_ids"],
                      key=lambda b: -result["final_stacks"][b]):
        delta = result["chip_delta"][bid]
        sign  = "+" if delta >= 0 else ""
        print(f"{bid:<25} {result['final_stacks'][bid]:>12,} {sign}{delta:>9,}")
    print(f"\nHands played: {result['n_hands']}")

    errs = {b: e for b, e in result["bot_errors"].items() if e}
    if errs:
        print(f"\nBot errors: {errs}")
