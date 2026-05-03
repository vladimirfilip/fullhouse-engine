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
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

TIMEOUT_SECONDS = 2

# Submission-time limits for the bot package.
# Bumped May 2026 to support optional /bot/data/ payloads (CFR blueprints,
# NN weights, lookup tables) loaded at module-import time.
MAX_PACKAGE_SIZE_BYTES = 250 * 1024 * 1024   # 250 MB total (zip or dir)
MAX_DATA_SIZE_BYTES    = 200 * 1024 * 1024   # 200 MB just for data/
MAX_BOT_PY_SIZE_BYTES  =   5 * 1024 * 1024   # 5 MB for the .py file

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

def _dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _resolve_submission(bot_path):
    """Accepts a .py file, a directory, or a .zip archive.
    Returns (bot_py_path, data_dir, cleanup_dir, package_errors).
    cleanup_dir, if not None, must be rmtree'd by the caller after validation.
    """
    p = os.path.abspath(bot_path)
    if not os.path.exists(p):
        return None, None, None, ["File not found: " + bot_path]

    cleanup_dir = None

    # .zip archive: extract into tempdir, then treat as directory.
    if os.path.isfile(p) and p.endswith(".zip"):
        if os.path.getsize(p) > MAX_PACKAGE_SIZE_BYTES:
            return None, None, None, [
                "Archive exceeds " + str(MAX_PACKAGE_SIZE_BYTES // (1024*1024)) + " MB limit"
            ]
        cleanup_dir = tempfile.mkdtemp(prefix="fhval_")
        try:
            with zipfile.ZipFile(p) as zf:
                for member in zf.infolist():
                    name = member.filename
                    if name.startswith("/") or "\\" in name:
                        return None, None, cleanup_dir, ["Unsafe zip path (absolute or backslash): " + repr(name)]
                    norm = os.path.normpath(os.path.join(cleanup_dir, name))
                    if not norm.startswith(cleanup_dir + os.sep) and norm != cleanup_dir:
                        return None, None, cleanup_dir, ["Unsafe zip path (traversal): " + repr(name)]
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        return None, None, cleanup_dir, ["Unsafe zip entry (symlink): " + repr(name)]
                zf.extractall(cleanup_dir)
        except zipfile.BadZipFile as e:
            return None, None, cleanup_dir, ["Invalid zip archive: " + str(e)]
        p = cleanup_dir  # fall through

    # Directory: must contain bot.py at root, optional data/ subdirectory.
    if os.path.isdir(p):
        bot_py = os.path.join(p, "bot.py")
        data_dir = os.path.join(p, "data")
        errors = []
        if not os.path.isfile(bot_py):
            errors.append("Submission must contain bot.py at the root")
            return None, None, cleanup_dir, errors

        # Forbid extra .py files at root and any symlinks
        for entry in os.listdir(p):
            full = os.path.join(p, entry)
            if entry == "bot.py" or entry == "data":
                continue
            if entry.endswith(".py"):
                errors.append("Only bot.py is allowed at the root: '" + entry + "' is forbidden")
            if os.path.islink(full):
                errors.append("Symlinks are not allowed: " + entry)

        # data/ scan: forbid .py files inside, enforce size cap
        if os.path.isdir(data_dir):
            for root, _, files in os.walk(data_dir):
                for fn in files:
                    if fn.endswith(".py"):
                        errors.append("data/ may not contain .py files: " + fn)
                    fp = os.path.join(root, fn)
                    if os.path.islink(fp):
                        errors.append("Symlinks are not allowed: data/" + fn)
            data_size = _dir_size(data_dir)
            if data_size > MAX_DATA_SIZE_BYTES:
                errors.append(
                    "data/ exceeds " + str(MAX_DATA_SIZE_BYTES // (1024*1024)) +
                    " MB limit (got " + str(data_size // (1024*1024)) + " MB)"
                )

        if _dir_size(p) > MAX_PACKAGE_SIZE_BYTES:
            errors.append(
                "Submission exceeds " + str(MAX_PACKAGE_SIZE_BYTES // (1024*1024)) + " MB limit"
            )

        if os.path.getsize(bot_py) > MAX_BOT_PY_SIZE_BYTES:
            errors.append("bot.py exceeds " + str(MAX_BOT_PY_SIZE_BYTES // (1024*1024)) + " MB limit")

        return bot_py, (data_dir if os.path.isdir(data_dir) else None), cleanup_dir, errors

    # Plain .py (legacy)
    if os.path.isfile(p) and p.endswith(".py"):
        if os.path.getsize(p) > MAX_BOT_PY_SIZE_BYTES:
            return None, None, None, ["bot.py exceeds " + str(MAX_BOT_PY_SIZE_BYTES // (1024*1024)) + " MB limit"]
        return p, None, None, []

    return None, None, cleanup_dir, ["Unsupported submission (must be .py, .zip, or directory): " + bot_path]


def validate(bot_path):
    """Full validation of a bot submission (.py file, directory, or .zip).
    Returns a result dict with passed/failed status and details.
    """
    results = {
        "bot_path": str(bot_path),
        "passed":   False,
        "errors":   [],
        "warnings": [],
        "tests":    [],
        "data_dir": None,
    }

    bot_py, data_dir, cleanup_dir, package_errors = _resolve_submission(bot_path)
    try:
        if package_errors:
            results["errors"].extend(package_errors)
            if not bot_py:
                return results
        results["data_dir"] = data_dir

        static_issues = check_static(bot_py)
        for issue in static_issues:
            if issue["level"] == "error":
                results["errors"].append(issue["message"])
            else:
                results["warnings"].append(issue["message"])

        if results["errors"]:
            return results

        try:
            bot = load_bot(bot_py)
        except Exception:
            results["errors"].append("Failed to load bot: " + traceback.format_exc().strip())
            return results

        if not hasattr(bot, "decide") or not callable(bot.decide):
            results["errors"].append("decide() function not found or not callable")
            return results

        all_passed = True
        for test in TEST_STATES:
            r = run_test(bot, test)
            results["tests"].append(r)
            if not r["passed"]:
                results["errors"].append("Test '" + r["test"] + "' failed: " + str(r.get("error", "")))
                all_passed = False

        results["passed"] = all_passed and not results["errors"]
        return results
    finally:
        if cleanup_dir and os.path.isdir(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate a Fullhouse bot submission")
    parser.add_argument("bot_path", help="Path to bot.py, bot directory, or bot.zip")
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
