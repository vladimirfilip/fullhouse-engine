"""
Fullhouse Bot Validator
=======================
Automatically checks a bot.py before it enters the tournament.

Checks performed:
  1. File exists and is valid Python (no syntax errors)
  2. No forbidden imports (network, filesystem, subprocess, os.system)
  3. decide() function exists and accepts one argument
  4. Returns a valid action dict on basic game states
  5. Doesn't crash on edge cases (preflop, all-in, can_check)
  6. Responds within 2-second timeout
  7. Never raises an unhandled exception

Usage:
  python3 sandbox/validator.py bots/mybot/bot.py
  python3 sandbox/validator.py bots/mybot/bot.py --json
"""

import ast
import importlib.util
import json
import signal
import sys
import time
import traceback
from pathlib import Path

TIMEOUT_SECONDS = 2

# Imports that are not allowed in submitted bots
FORBIDDEN_MODULES = {
    "socket", "urllib", "urllib2", "urllib3", "requests", "httpx", "aiohttp",
    "http", "ftplib", "smtplib", "telnetlib", "xmlrpc",
    "subprocess", "multiprocessing", "os",
    "open", "builtins.open",
    "pickle", "shelve",
    "threading",                     # allow for complex strategies but flag
}

# A set of realistic game states to test against
TEST_STATES = [
    {
        "name": "preflop_call_or_fold",
        "state": {
            "type": "action_request",
            "hand_id": "val_001",
            "street": "preflop",
            "seat_to_act": 0,
            "pot": 150,
            "community_cards": [],
            "current_bet": 100,
            "min_raise_to": 200,
            "amount_owed": 100,
            "can_check": False,
            "your_cards": ["As", "Kh"],
            "your_stack": 9900,
            "your_bet_this_street": 0,
            "players": [
                {"seat": 0, "bot_id": "bot_under_test", "stack": 9900,
                 "state": "active", "is_folded": False, "is_all_in": False,
                 "bet_this_street": 0, "hole_cards": None},
                {"seat": 1, "bot_id": "opponent", "stack": 9900,
                 "state": "active", "is_folded": False, "is_all_in": False,
                 "bet_this_street": 100, "hole_cards": None},
            ],
            "action_log": [
                {"seat": 0, "action": "small_blind", "amount": 50},
                {"seat": 1, "action": "big_blind",   "amount": 100},
            ],
        },
    },
    {
        "name": "postflop_can_check",
        "state": {
            "type": "action_request",
            "hand_id": "val_002",
            "street": "flop",
            "seat_to_act": 0,
            "pot": 300,
            "community_cards": ["7s", "Td", "2h"],
            "current_bet": 0,
            "min_raise_to": 100,
            "amount_owed": 0,
            "can_check": True,
            "your_cards": ["Ah", "Kd"],
            "your_stack": 9850,
            "your_bet_this_street": 0,
            "players": [
                {"seat": 0, "bot_id": "bot_under_test", "stack": 9850,
                 "state": "active", "is_folded": False, "is_all_in": False,
                 "bet_this_street": 0, "hole_cards": None},
                {"seat": 1, "bot_id": "opponent", "stack": 9850,
                 "state": "active", "is_folded": False, "is_all_in": False,
                 "bet_this_street": 0, "hole_cards": None},
            ],
            "action_log": [],
        },
    },
    {
        "name": "river_facing_large_bet",
        "state": {
            "type": "action_request",
            "hand_id": "val_003",
            "street": "river",
            "seat_to_act": 0,
            "pot": 4000,
            "community_cards": ["7s", "Td", "2h", "Kc", "5d"],
            "current_bet": 2000,
            "min_raise_to": 4000,
            "amount_owed": 2000,
            "can_check": False,
            "your_cards": ["2c", "3d"],
            "your_stack": 6000,
            "your_bet_this_street": 0,
            "players": [
                {"seat": 0, "bot_id": "bot_under_test", "stack": 6000,
                 "state": "active", "is_folded": False, "is_all_in": False,
                 "bet_this_street": 0, "hole_cards": None},
                {"seat": 1, "bot_id": "opponent", "stack": 4000,
                 "state": "active", "is_folded": False, "is_all_in": False,
                 "bet_this_street": 2000, "hole_cards": None},
            ],
            "action_log": [],
        },
    },
    {
        "name": "short_stack_all_in_decision",
        "state": {
            "type": "action_request",
            "hand_id": "val_004",
            "street": "preflop",
            "seat_to_act": 0,
            "pot": 200,
            "community_cards": [],
            "current_bet": 100,
            "min_raise_to": 200,
            "amount_owed": 100,
            "can_check": False,
            "your_cards": ["Qh", "Qs"],
            "your_stack": 80,
            "your_bet_this_street": 20,
            "players": [
                {"seat": 0, "bot_id": "bot_under_test", "stack": 80,
                 "state": "active", "is_folded": False, "is_all_in": False,
                 "bet_this_street": 20, "hole_cards": None},
                {"seat": 1, "bot_id": "opponent", "stack": 9900,
                 "state": "active", "is_folded": False, "is_all_in": False,
                 "bet_this_street": 100, "hole_cards": None},
            ],
            "action_log": [],
        },
    },
]

VALID_ACTIONS = {"fold", "check", "call", "raise", "all_in"}


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------

def check_static(path: str) -> list:
    """
    Parse bot.py and flag forbidden imports and other issues.
    Returns list of warning dicts.
    """
    warnings = []
    source = Path(path).read_text()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [{"level": "error", "check": "syntax", "message": str(e)}]

    # Walk the AST for imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []
            for name in names:
                if name in FORBIDDEN_MODULES:
                    warnings.append({
                        "level": "error",
                        "check": "forbidden_import",
                        "message": f"Forbidden import: '{name}' — bots may not use network, "
                                   f"filesystem, or subprocess modules.",
                    })

    # Check decide() exists
    has_decide = any(
        isinstance(node, ast.FunctionDef) and node.name == "decide"
        for node in ast.walk(tree)
    )
    if not has_decide:
        warnings.append({
            "level": "error",
            "check": "missing_decide",
            "message": "No decide() function found in bot.py",
        })

    return warnings


# ---------------------------------------------------------------------------
# Runtime checks
# ---------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError("Bot exceeded time limit")


def load_bot(path: str):
    spec   = importlib.util.spec_from_file_location("bot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_test(bot_module, test: dict) -> dict:
    """Run one test state and return result dict."""
    state  = test["state"]
    start  = time.time()

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        result = bot_module.decide(state)
        signal.alarm(0)
        elapsed = time.time() - start
    except TimeoutError:
        signal.alarm(0)
        return {
            "test":    test["name"],
            "passed":  False,
            "error":   f"Timeout: bot did not respond within {TIMEOUT_SECONDS}s",
            "elapsed": TIMEOUT_SECONDS,
        }
    except Exception:
        signal.alarm(0)
        return {
            "test":    test["name"],
            "passed":  False,
            "error":   f"Exception: {traceback.format_exc().strip()}",
            "elapsed": round(time.time() - start, 3),
        }

    # Validate the returned action
    if not isinstance(result, dict):
        return {"test": test["name"], "passed": False,
                "error": f"decide() must return dict, got {type(result).__name__}",
                "elapsed": round(elapsed, 3)}

    action = result.get("action", "").lower()
    if action not in VALID_ACTIONS:
        return {"test": test["name"], "passed": False,
                "error": f"Invalid action '{action}'. Must be one of: {VALID_ACTIONS}",
                "elapsed": round(elapsed, 3)}

    if action == "raise" and "amount" not in result:
        return {"test": test["name"], "passed": False,
                "error": "Raise action missing 'amount' key",
                "elapsed": round(elapsed, 3)}

    return {
        "test":    test["name"],
        "passed":  True,
        "action":  result,
        "elapsed": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate(bot_path: str) -> dict:
    """
    Full validation of a bot.py file.
    Returns a result dict with passed/failed status and details.
    """
    path = Path(bot_path)
    results = {
        "bot_path": str(path),
        "passed":   False,
        "errors":   [],
        "warnings": [],
        "tests":    [],
    }

    # 1. File exists
    if not path.exists():
        results["errors"].append(f"File not found: {bot_path}")
        return results

    if path.suffix != ".py":
        results["errors"].append("Bot file must be a .py file")
        return results

    # 2. Static analysis
    static_issues = check_static(str(path))
    for issue in static_issues:
        if issue["level"] == "error":
            results["errors"].append(issue["message"])
        else:
            results["warnings"].append(issue["message"])

    if results["errors"]:
        return results

    # 3. Load bot
    try:
        bot = load_bot(str(path))
    except Exception:
        results["errors"].append(f"Failed to load bot: {traceback.format_exc().strip()}")
        return results

    if not hasattr(bot, "decide") or not callable(bot.decide):
        results["errors"].append("decide() function not found or not callable")
        return results

    # 4. Runtime tests
    all_passed = True
    for test in TEST_STATES:
        r = run_test(bot, test)
        results["tests"].append(r)
        if not r["passed"]:
            results["errors"].append(f"Test '{r['test']}' failed: {r.get('error', '')}")
            all_passed = False

    results["passed"] = all_passed and not results["errors"]
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate a Fullhouse bot.py")
    parser.add_argument("bot_path", help="Path to bot.py")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = validate(args.bot_path)

    if args.json:
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] else 1)

    # Human-readable output
    status = "✅ PASSED" if result["passed"] else "❌ FAILED"
    print(f"\n{status}  —  {result['bot_path']}\n")

    if result["errors"]:
        print("Errors:")
        for e in result["errors"]:
            print(f"  ✗ {e}")

    if result["warnings"]:
        print("Warnings:")
        for w in result["warnings"]:
            print(f"  ⚠  {w}")

    if result["tests"]:
        print("\nTest results:")
        for t in result["tests"]:
            icon    = "✓" if t["passed"] else "✗"
            action  = t.get("action", {})
            elapsed = t.get("elapsed", 0)
            detail  = f"{action}" if t["passed"] else t.get("error", "")
            print(f"  {icon} [{elapsed:.3f}s] {t['test']}: {detail}")

    print()
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
