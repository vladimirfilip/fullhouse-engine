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

BOT_PATH = os.environ.get("BOT_PATH", "/bot/bot.py")
TIMEOUT  = int(os.environ.get("ACTION_TIMEOUT", "2"))


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

        signal.alarm(TIMEOUT)
        try:
            action = bot.decide(state)
            signal.alarm(0)
            if not isinstance(action, dict) or "action" not in action:
                raise ValueError("decide() must return dict with 'action' key")
            emit(action)
        except TimeoutError:
            signal.alarm(0)
            sys.stderr.write("[runner] TIMEOUT\n")
            emit({"action": "fold", "error": "timeout"})
        except Exception:
            signal.alarm(0)
            sys.stderr.write(f"[runner] BOT EXCEPTION:\n{traceback.format_exc()}\n")
            emit({"action": "fold", "error": "exception"})


if __name__ == "__main__":
    main()
