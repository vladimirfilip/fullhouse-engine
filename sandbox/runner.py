"""
Fullhouse Bot Runner — executes inside the Docker sandbox.
Loads /bot/bot.py, reads game states from stdin, writes actions to stdout.
All communication is newline-delimited JSON.
"""

import sys
import json
import importlib.util
import traceback
import signal
import os

BOT_PATH        = os.environ.get("BOT_PATH", "/bot/bot.py")
TIMEOUT         = int(os.environ.get("ACTION_TIMEOUT", "2"))
WARMUP_TIMEOUT  = int(os.environ.get("WARMUP_TIMEOUT", "30"))


def load_bot(path: str):
    spec   = importlib.util.spec_from_file_location("bot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "decide"):
        raise AttributeError("bot.py must define a decide() function")
    return module


def _timeout(signum, frame):
    raise TimeoutError()


def emit(obj: dict):
    print(json.dumps(obj), flush=True)


def main():
    try:
        bot = load_bot(BOT_PATH)
    except Exception as e:
        sys.stderr.write(f"[runner] LOAD ERROR: {e}\n")
        for line in sys.stdin:
            emit({"action": "fold", "error": "load_failed"})
        return

    signal.signal(signal.SIGALRM, _timeout)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            state = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[runner] BAD JSON: {e}\n")
            emit({"action": "fold", "error": "bad_json"})
            continue

        # Warm-up: one-shot call before hand 1 with a longer alarm so bots
        # loading CFR blueprints / NN weights don't get killed at 2 s.
        # We don't penalise warmup errors — failure here just means hand 1
        # will run with whatever's loaded.
        is_warmup = state.get("type") == "warmup"
        signal.alarm(WARMUP_TIMEOUT if is_warmup else TIMEOUT)
        try:
            action = bot.decide(state)
            signal.alarm(0)
            if is_warmup:
                emit({"ok": True})
                continue
            if not isinstance(action, dict) or "action" not in action:
                raise ValueError("decide() must return dict with 'action' key")
            emit(action)
        except TimeoutError:
            signal.alarm(0)
            if is_warmup:
                sys.stderr.write("[runner] WARMUP TIMEOUT\n")
                emit({"ok": False, "error": "warmup_timeout"})
                continue
            sys.stderr.write("[runner] TIMEOUT\n")
            emit({"action": "fold", "error": "timeout"})
        except Exception:
            signal.alarm(0)
            if is_warmup:
                sys.stderr.write(f"[runner] WARMUP EXCEPTION:\n{traceback.format_exc()}\n")
                emit({"ok": False, "error": "warmup_exception"})
                continue
            sys.stderr.write(f"[runner] BOT EXCEPTION:\n{traceback.format_exc()}\n")
            emit({"action": "fold", "error": "exception"})


if __name__ == "__main__":
    main()
