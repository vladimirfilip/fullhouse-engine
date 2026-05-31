try:
    import eval7
except Exception:
    eval7 = None

import json
import os
import time

BOT_NAME = "MyBot"
BOT_AVATAR = "robot_1"

# The first config groups to tune after match results are usually:
# 1. c-bet / bluff frequencies, 2. river call thresholds, 3. exploit/leak multipliers.
CONFIG = {
    "fallback": {
        "tiny_call_chips": 20,
        "tiny_call_pot_fraction": 0.08,
    },
    "preflop": {
        "open_size_bb": 2.5,
        "three_bet_ip_multiplier": 3.0,
        "three_bet_oop_multiplier": 3.5,
        "table_aggression": {6: 1.00, 5: 1.08, 4: 1.20, 3: 1.40, 2: 1.65},
    },
    "stacks": {
        "stack_buckets_bb": {"micro": 8, "short": 20, "medium": 50, "deep": 100},
        "spr_buckets": {"committed": 1, "low": 3, "medium": 6, "high": 12},
        "commitment_thresholds": {"committed": 0.35, "low": 0.48, "medium": 0.66, "high": 0.89},
    },
    "postflop": {
        "bet_sizes": {
            "small": 0.33,
            "medium": 0.55,
            "large": 0.75,
            "overbet": 1.25,
            "value_default": 0.66,
        },
        "cbet_frequencies": {
            "good": 0.72,
            "neutral": 0.45,
            "bad": 0.12,
            "bad_multiway": 0.02,
            "medium_good": 0.35,
        },
    },
    "monte_carlo": {
        "call_iters": 900,
        "all_in_iters": 1000,
        "default_iters": 800,
        "call_time_budget": 0.28,
        "all_in_time_budget": 0.32,
        "default_time_budget": 0.30,
        "call_margin": 0.02,
        "call_reject_margin": 0.03,
        "all_in_threshold_discount": 0.10,
        "all_in_reject_margin": 0.05,
    },
    "equity_realisation": {
        "in_position_bonus": 0.03,
        "out_of_position_penalty": 0.03,
        "multiway_penalty": 0.05,
        "high_spr_strong_penalty": 0.09,
        "low_spr_strong_bonus": 0.04,
        "high_spr_gutshot_penalty": 0.04,
    },
    "exploit": {
        "low_confidence": 0.20,
        "medium_confidence": 0.55,
        "nit_value_threshold": 0.08,
        "station_value_threshold": -0.08,
        "maniac_value_threshold": 0.02,
        "nit_bluff_factor": 1.12,
        "station_bluff_factor": 0.45,
        "maniac_bluff_factor": 0.72,
        "solid_bluff_factor": 0.98,
        "nit_call_margin": -0.04,
        "station_call_margin": 0.03,
        "maniac_call_margin": 0.025,
        "solid_call_margin": 0.0,
    },
    "opponent_tags": {
        "hands_low_confidence": 20,
        "hands_medium_confidence": 60,
        "nit": {"vpip_max": 0.18, "pfr_max": 0.14, "fold_freq_min": 0.32, "recent_fold_min": 0.22},
        "station": {"call_freq_min": 0.40, "raise_freq_max": 0.18, "vpip_min": 0.24},
        "maniac": {"vpip_min": 0.36, "raise_freq_min": 0.24, "aggression_min": 1.8, "all_in_freq_min": 0.08, "recent_aggression_min": 0.35},
        "solid": {"vpip_min": 0.16, "vpip_max": 0.34, "pfr_min": 0.12, "pfr_max": 0.28, "aggression_min": 0.7},
    },
    "leaks": {
        "sample_smoothing_target": 20,
        "confidence_sample_target": 25,
        "flop_cbet_fold_high": 0.62,
        "turn_barrel_fold_high": 0.64,
        "river_call_high": 0.62,
        "river_raise_high": 0.18,
        "underbluff_high": 0.65,
        "check_raise_flop_high": 0.35,
    },
    "anti_punt": {
        "high_spr_one_pair": 6.0,
        "big_river_call_pot_fraction": 0.50,
        "very_strong_line_pot_fraction": 0.75,
        "deep_multiway_draw_spr": 6.0,
    },
    "river": {
        "strong_call_base": 0.21,
        "medium_tiny_call_base": 0.10,
        "polar_medium_call_cap": 0.18,
        "maniac_medium_call_cap": 0.20,
        "thin_value_factor_station": 1.0,
        "bluff_showdown_cap": 0.22,
    },
    "mix": {
        "good_cbet": 0.70,
        "neutral_cbet": 0.45,
        "marginal_semibluff": 0.30,
        "medium_protection": 0.20,
        "maniac_trap": 0.25,
        "river_bluff": 0.28,
    },
    "variant_features": {
        "use_preflop_tables": False,
        "use_opponent_classifier": False,
        "use_gto_guards": False,
        "use_fold_equity_model": False,
    },
    "variant_models": {
        "M0_SAFE_BASE": {},
        "M1_PREFLOP_TABLES": {"use_preflop_tables": True},
        "M2_OPPONENT_CLASSIFIER": {"use_opponent_classifier": True},
        "M3_GTO_GUARDS": {"use_gto_guards": True},
        "M4_FOLD_EQUITY_MODEL": {"use_fold_equity_model": True},
        "M5_FULL_STACK": {
            "use_preflop_tables": True,
            "use_opponent_classifier": True,
            "use_gto_guards": True,
            "use_fold_equity_model": True,
        },
    },
}


def get_config(path, default=None):
    """Read a nested config value using dotted paths like 'postflop.bet_sizes.small'."""
    current = CONFIG
    for part in str(path).split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


TINY_CALL_CHIPS = get_config("fallback.tiny_call_chips", 20)
TINY_CALL_POT_FRACTION = get_config("fallback.tiny_call_pot_fraction", 0.08)

OPEN_SIZE_BB = get_config("preflop.open_size_bb", 2.5)
THREE_BET_IP_MULTIPLIER = get_config("preflop.three_bet_ip_multiplier", 3.0)
THREE_BET_OOP_MULTIPLIER = get_config("preflop.three_bet_oop_multiplier", 3.5)

EXPLOIT_ADJUSTMENTS = get_config("exploit", {})
MIX_FREQUENCIES = get_config("mix", {})

OPPONENTS = {}
OPPONENT_MODELS = OPPONENTS
SEEN_HANDS = set()
SEEN_ACTION_KEYS = set()
CURRENT_HAND_ID = None

DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
RAW_MODEL_VARIANT = os.environ.get("BOT_MODEL_VARIANT", "M0_SAFE_BASE")


def resolve_variant_name(raw_name=None) -> str:
    """Return a valid variant preset name, falling back safely to SAFE_BASE."""
    candidate = str(raw_name or RAW_MODEL_VARIANT or "M0_SAFE_BASE").strip()
    available = get_config("variant_models", {})
    if isinstance(available, dict) and candidate in available:
        return candidate
    return "M0_SAFE_BASE"


def build_feature_flags(variant_name: str) -> dict:
    """Merge base feature defaults with one named variant preset."""
    defaults = dict(get_config("variant_features", {}) or {})
    presets = get_config("variant_models", {})
    selected = presets.get(variant_name, {}) if isinstance(presets, dict) else {}
    if isinstance(selected, dict):
        defaults.update(selected)
    return {str(key): bool(value) for key, value in defaults.items()}


def load_optional_assets(data_dir: str) -> dict:
    """Load optional assets at import time only; failures disable features safely."""
    assets = {
        "data_dir": str(data_dir or ""),
        "data_dir_exists": False,
        "preflop_tables": None,
        "preflop_tables_data": None,
        "opponent_classifier": None,
        "fold_equity_model": None,
        "load_errors": {},
    }
    try:
        assets["data_dir_exists"] = os.path.isdir(data_dir)
    except Exception as exc:
        assets["load_errors"]["data_dir"] = str(exc)
        return assets

    candidates = {
        "preflop_tables": ["preflop_tables.npz", "preflop_tables.json", "preflop_tables.pkl"],
        "opponent_classifier": ["opponent_classifier.pkl", "opponent_classifier.json"],
        "fold_equity_model": ["fold_equity_model.pkl", "fold_equity_model.json"],
    }
    for asset_name, filenames in candidates.items():
        try:
            for filename in filenames:
                path = os.path.join(data_dir, filename)
                if os.path.exists(path):
                    assets[asset_name] = path
                    break
        except Exception as exc:
            assets["load_errors"][asset_name] = str(exc)
    if assets.get("preflop_tables"):
        try:
            with open(str(assets["preflop_tables"]), "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if _valid_preflop_tables_payload(payload):
                assets["preflop_tables_data"] = payload
            else:
                assets["load_errors"]["preflop_tables_data"] = "invalid_payload_shape"
        except Exception as exc:
            assets["load_errors"]["preflop_tables_data"] = str(exc)
    return assets


IMPLEMENTED_FEATURES = {
    "use_preflop_tables": True,
    "use_opponent_classifier": False,
    "use_gto_guards": False,
    "use_fold_equity_model": False,
}
ASSET_REQUIREMENTS = {
    "use_preflop_tables": "preflop_tables_data",
    "use_opponent_classifier": "opponent_classifier",
    "use_gto_guards": None,
    "use_fold_equity_model": "fold_equity_model",
}


def feature_available(flag_name: str, assets=None) -> bool:
    """A feature is available only when enabled, implemented, and any asset exists."""
    if not bool(FEATURE_FLAGS.get(flag_name)):
        return False
    if not bool(IMPLEMENTED_FEATURES.get(flag_name)):
        return False
    requirement = ASSET_REQUIREMENTS.get(flag_name)
    if not requirement:
        return True
    asset_map = assets if isinstance(assets, dict) else OPTIONAL_ASSETS
    return bool(asset_map.get(requirement))


def build_feature_status(feature_flags: dict, assets: dict) -> dict:
    """Describe why each optional feature is enabled or disabled."""
    status = {}
    for flag_name, enabled in feature_flags.items():
        if not enabled:
            status[flag_name] = "disabled_by_variant"
            continue
        if not IMPLEMENTED_FEATURES.get(flag_name):
            status[flag_name] = "disabled_unimplemented"
            continue
        requirement = ASSET_REQUIREMENTS.get(flag_name)
        if requirement and not assets.get(requirement):
            status[flag_name] = f"disabled_missing_{requirement}"
            continue
        status[flag_name] = "enabled"
    return status


def _valid_preflop_tables_payload(payload) -> bool:
    """Accept only a minimal safe JSON structure for the M1 overlay."""
    if not isinstance(payload, dict):
        return False
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return False
    for rule in rules:
        if not isinstance(rule, dict):
            return False
        if not isinstance(rule.get("action"), str):
            return False
    return True


RESOLVED_VARIANT_NAME = resolve_variant_name(RAW_MODEL_VARIANT)
OPTIONAL_ASSETS = load_optional_assets(DATA_DIR)
FEATURE_FLAGS = build_feature_flags(RESOLVED_VARIANT_NAME)
FEATURE_STATUS = build_feature_status(FEATURE_FLAGS, OPTIONAL_ASSETS)


def decide(game_state: dict) -> dict:
    """Required engine entry point."""
    try:
        update_opponent_model(game_state)
        action = main_decide(game_state)
        action = apply_leak_layer(game_state, action)
        action = apply_anti_punt_layer(game_state, action)
    except Exception:
        action = safe_fallback(game_state)
    return legalise_action(action, game_state)


def main_decide(state: dict) -> dict:
    """
    Stage 3 adds a preflop engine while keeping postflop intentionally simple.

    The parsed context now powers preflop opens, 3-bets, and short-stack jams.
    Postflop remains conservative until later stages.
    """
    ctx = parse_state(state)
    if ctx["street"] == "preflop":
        return variant_preflop_decision(state, ctx)
    info = build_postflop_info(state, ctx)
    return postflop_decide(state, info)


def apply_anti_punt_layer(state: dict, action: dict) -> dict:
    """Run the final anti-punt safeguard pass before legalisation."""
    try:
        ctx = parse_state(state)
        if ctx["street"] == "preflop":
            return action
        hand_info = build_postflop_info(state, ctx)
        return apply_anti_punt_overrides(
            action,
            ctx,
            hand_info.get("stack_ctx"),
            hand_info,
            hand_info.get("board_facts"),
            hand_info.get("opponent_profile"),
        )
    except Exception:
        return action


def apply_leak_layer(state: dict, action: dict) -> dict:
    """Apply Stage 16 leak-vector nudges after exploit logic and before anti-punt."""
    try:
        ctx = parse_state(state)
        if ctx["street"] == "preflop":
            return action
        hand_info = build_postflop_info(state, ctx)
        return apply_leak_adjustments(
            action,
            ctx,
            hand_info.get("stack_ctx"),
            hand_info,
            hand_info.get("board_facts"),
            hand_info.get("opponent_profile"),
            hand_info.get("range_bucket"),
        )
    except Exception:
        return action


def variant_preflop_decision(state: dict, ctx: dict) -> dict:
    """Optional preflop-table hook; currently falls back safely to the baseline."""
    if feature_available("use_preflop_tables"):
        try:
            table_action = preflop_table_decision(state, ctx)
            if isinstance(table_action, dict):
                return table_action
        except Exception:
            pass
    return preflop_decision(state, ctx)


def preflop_table_decision(state: dict, ctx: dict):
    """Conservative M1 overlay: only adjust a few marginal preflop branches."""
    table = OPTIONAL_ASSETS.get("preflop_tables_data")
    if not isinstance(table, dict):
        return None

    hand = normalize_hand(ctx.get("your_cards"))
    if not hand:
        return None

    bucket = preflop_position_bucket(ctx.get("position_info", {}))
    profile = {
        "table_size": classify_table_size(ctx.get("players_at_table", 0)),
        "position_bucket": bucket,
        "stack_bucket": str(ctx.get("stack_bucket") or ""),
        "facing_action": classify_preflop_facing_action(ctx),
        "hand_class": classify_preflop_hand_class(hand),
    }
    if profile["facing_action"] == "other":
        return None

    rule = lookup_preflop_table_rule(table, profile)
    if not isinstance(rule, dict):
        return None
    return apply_preflop_table_rule(state, ctx, hand, rule, profile)


def classify_table_size(players_at_table: int) -> str:
    """Collapse current table size into a few conservative preflop buckets."""
    count = max(0, _as_int(players_at_table))
    if count <= 2:
        return "heads_up"
    if count <= 4:
        return "short"
    return "full"


def classify_preflop_facing_action(ctx: dict) -> str:
    """Describe the main preflop branch the baseline engine is in."""
    if _is_unopened_preflop(ctx):
        return "unopened"
    if _is_facing_open(ctx):
        return "facing_open"
    if str(ctx.get("street") or "") == "preflop":
        return "inflated"
    return "other"


def classify_preflop_hand_class(hand: str) -> str:
    """Map detailed hand codes into compact M1 overlay classes."""
    if not hand:
        return "unknown"
    if hand in {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}:
        return "premium"
    if hand in {"TT", "99", "AQs", "AQo", "AJs", "KQs"}:
        return "strong"
    if len(hand) == 2:
        pair_rank = hand[0]
        if pair_rank in {"8", "7", "6", "5", "4", "3", "2"}:
            return "small_pair"
    if hand.startswith("A") and hand.endswith("s") and len(hand) == 3:
        return "suited_ace"
    if hand.endswith("s") and hand[:2] in {"KQ", "KJ", "QJ", "JT", "T9", "98", "87", "76", "65", "54"}:
        return "suited_connector"
    if hand in {"AJo", "ATo", "KQo", "KJo", "QJo", "KTo", "QTo", "JTo"}:
        return "offsuit_broadway"
    return "other"


def lookup_preflop_table_rule(table: dict, profile: dict):
    """Find the first matching table rule for the current public preflop spot."""
    rules = table.get("rules") if isinstance(table.get("rules"), list) else []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if _rule_matches(rule, profile):
            return rule
    return None


def _rule_matches(rule: dict, profile: dict) -> bool:
    for key in ("table_size", "position_bucket", "stack_bucket", "facing_action", "hand_class"):
        expected = rule.get(key, "any")
        if expected == "any":
            continue
        if isinstance(expected, list):
            if profile.get(key) not in expected:
                return False
            continue
        if profile.get(key) != expected:
            return False
    return True


def apply_preflop_table_rule(state: dict, ctx: dict, hand: str, rule: dict, profile: dict):
    """Apply a small M1 action nudge while preserving the baseline engine and legality."""
    action_name = str(rule.get("action") or "").lower()
    threshold = max(0.0, min(1.0, float(rule.get("call_pot_odds_max", 1.0) or 1.0)))
    if action_name == "prefer_call":
        if _is_facing_open(ctx) and _can_flat_preflop(ctx, hand) and float(ctx.get("pot_odds", 1.0)) <= threshold:
            return {"action": "call"}
        return None
    if action_name == "prefer_fold":
        if _is_facing_open(ctx) or classify_preflop_facing_action(ctx) == "inflated":
            return make_fold_or_check(state)
        return None
    if action_name == "prefer_raise":
        if _is_unopened_preflop(ctx) and can_raise(state):
            return {"action": "raise", "amount": get_open_raise_size(ctx)}
        return None
    if action_name == "prefer_jam":
        if ctx.get("stack_bucket") in {"micro", "short"} and _should_jam_short_stack(ctx):
            return {"action": "all_in"}
        return None
    if action_name == "suppress_bluff_3bet":
        if _is_facing_open(ctx):
            return {"action": "call"} if _can_flat_preflop(ctx, hand) and float(ctx.get("pot_odds", 1.0)) <= threshold else make_fold_or_check(state)
        return None
    return None


def enrich_opponent_profile_variant(ctx: dict, profile: dict) -> dict:
    """Optional classifier hook; currently returns the baseline public profile."""
    safe_profile = dict(profile) if isinstance(profile, dict) else _default_profile()
    if feature_available("use_opponent_classifier"):
        try:
            classified = opponent_classifier_profile(ctx, safe_profile)
            if isinstance(classified, dict):
                return classified
        except Exception:
            pass
    return safe_profile


def opponent_classifier_profile(ctx: dict, profile: dict):
    """Placeholder for future classifier enrichment."""
    return None


def apply_postflop_guard_variant(state: dict, action: dict, info: dict) -> dict:
    """Optional postflop guard hook; currently no-ops back to the chosen action."""
    if feature_available("use_gto_guards"):
        try:
            guarded = gto_guard_action(state, action, info)
            if isinstance(guarded, dict):
                return guarded
        except Exception:
            pass
    return action


def gto_guard_action(state: dict, action: dict, info: dict):
    """Placeholder for future GTO-style guardrails."""
    return None


def fold_equity_multiplier_variant(ctx: dict, info: dict) -> float:
    """Optional fold-equity model hook; defaults to neutral multiplier."""
    if feature_available("use_fold_equity_model"):
        try:
            value = fold_equity_model_multiplier(ctx, info)
            if isinstance(value, (int, float)):
                return max(0.25, min(2.0, float(value)))
        except Exception:
            pass
    return 1.0


def fold_equity_model_multiplier(ctx: dict, info: dict):
    """Placeholder for future fold-equity model outputs."""
    return None


def safe_fallback(state: dict) -> dict:
    """
    Prefer the lowest-risk legal action.

    If checking is free, always check. Otherwise only continue for tiny calls;
    larger unknown spots fold by default.
    """
    if state.get("can_check"):
        return {"action": "check"}
    if _is_tiny_call(state):
        return {"action": "call"}
    return {"action": "fold"}


def legalise_action(action: dict, state: dict) -> dict:
    """Clamp strategy output to a valid engine action."""
    if not isinstance(action, dict):
        return safe_fallback(state)

    action_name = action.get("action")
    if action_name == "check":
        return {"action": "check"} if state.get("can_check") else make_fold_or_check(state)

    if action_name == "fold":
        return make_fold_or_check(state)

    if action_name == "call":
        return make_call_or_check(state)

    if action_name == "all_in":
        if _as_int(state.get("your_stack")) > 0:
            return {"action": "all_in"}
        return make_call_or_check(state)

    if action_name == "raise":
        if not can_raise(state):
            return make_call_or_check(state)

        requested_amount = _as_int(action.get("amount"))
        min_raise_to = _as_int(state.get("min_raise_to"))
        max_raise_to = _max_total_bet(state)

        if max_raise_to <= 0:
            return make_call_or_check(state)
        if max_raise_to <= min_raise_to:
            return {"action": "all_in"}

        legal_amount = max(requested_amount, min_raise_to)
        if legal_amount > max_raise_to:
            return {"action": "all_in"}
        return {"action": "raise", "amount": legal_amount}

    return safe_fallback(state)


def can_raise(state: dict) -> bool:
    """True when we can make a legal non-all-in raise."""
    min_raise_to = _as_int(state.get("min_raise_to"))
    max_raise_to = _max_total_bet(state)
    return min_raise_to > 0 and max_raise_to > min_raise_to


def make_call_or_check(state: dict) -> dict:
    """Use the passive legal option for the spot."""
    if state.get("can_check"):
        return {"action": "check"}
    if _as_int(state.get("amount_owed")) > 0:
        return {"action": "call"}
    return {"action": "check"}


def make_fold_or_check(state: dict) -> dict:
    """Fold when facing a bet, otherwise take the free check."""
    if state.get("can_check"):
        return {"action": "check"}
    return {"action": "fold"}


def _default_open_size(state: dict) -> int:
    """
    Conservative Stage 1 raise sizing.

    The engine expects a total-bet target, so this function returns the final
    amount we want our bet to become, not the amount to add.
    """
    pot = max(0, _as_int(state.get("pot")))
    current_bet = max(0, _as_int(state.get("current_bet")))
    base_size = max(current_bet * 3, pot + max(10, current_bet))
    return max(base_size, _as_int(state.get("min_raise_to")))


def preflop_decision(state: dict, ctx: dict) -> dict:
    """Use simple position-aware preflop ranges without widening into spew."""
    hand = normalize_hand(ctx["your_cards"])
    if not hand:
        return make_fold_or_check(state)

    bucket = preflop_position_bucket(ctx["position_info"])
    open_ranges = get_open_ranges(ctx["players_at_table"])
    defend_ranges = get_defend_ranges()
    three_bet_value = get_three_bet_value_ranges()
    three_bet_bluff = get_three_bet_bluff_ranges()

    if ctx["stack_bucket"] in {"micro", "short"} and hand in get_short_stack_jam_range(bucket):
        if _should_jam_short_stack(ctx):
            return {"action": "all_in"}

    if _is_unopened_preflop(ctx):
        if hand in open_ranges.get(bucket, set()):
            if can_raise(state):
                return {"action": "raise", "amount": get_open_raise_size(ctx)}
            return make_call_or_check(state)
        return make_fold_or_check(state)

    if _is_facing_open(ctx):
        if hand in three_bet_value.get(bucket, set()):
            if _should_jam_short_stack(ctx):
                return {"action": "all_in"}
            if can_raise(state):
                return {"action": "raise", "amount": get_three_bet_size(ctx)}
            return make_call_or_check(state)

        if hand in three_bet_bluff.get(bucket, set()) and can_raise(state):
            return {"action": "raise", "amount": get_three_bet_size(ctx)}

        if hand in defend_ranges.get(bucket, set()) and _can_flat_preflop(ctx, hand):
            return {"action": "call"}
        return make_fold_or_check(state)

    if hand in get_four_bet_continue_range(bucket):
        if _should_jam_short_stack(ctx):
            return {"action": "all_in"}
        return make_call_or_check(state)

    return make_fold_or_check(state)


def build_postflop_info(state: dict, ctx: dict) -> dict:
    """Collect the postflop signals used by the baseline decision layer."""
    draw_info = detect_draws(state)
    texture = classify_board_texture(state)
    hand_bucket = estimate_hand_strength_bucket(state)
    board_facts = analyse_board_texture(state)
    initiative = hero_has_initiative(state)
    stack_ctx = build_stack_context(ctx)
    primary_opponent = get_primary_opponent(state)
    opponent_profile = enrich_opponent_profile_variant(ctx, get_opponent_profile(primary_opponent))
    update_leak_vector(ctx, opponent_profile)
    opponent_tag, opponent_confidence = classify_opponent(opponent_profile)
    adjustment_hint = opponent_adjustment_hint(ctx, opponent_profile)
    primary_villain = get_primary_villain_profile(ctx)
    base_range_bucket = infer_villain_range_bucket(ctx, primary_villain)
    range_bucket = update_range_bucket_for_action(
        base_range_bucket,
        _latest_villain_action(ctx),
        ctx["street"],
        max(0, _as_int(ctx.get("current_bet"))),
        board_facts,
        primary_villain,
        stack_ctx,
    )
    range_adjustments = range_bucket_adjustments(range_bucket, ctx, {"hand_bucket": hand_bucket, "draws": draw_info}, board_facts)
    return {
        "ctx": ctx,
        "stack_ctx": stack_ctx,
        "street": ctx["street"],
        "hand_bucket": hand_bucket,
        "hand_category": get_hand_category(state),
        "draws": draw_info,
        "board_texture": texture,
        "board_facts": board_facts,
        "showdown_value": estimate_showdown_value(state),
        "initiative": initiative,
        "in_position": ctx["position_info"]["is_late_position"],
        "multiway": ctx["active_opponents"] >= 2,
        "players_in_hand": len(ctx["players_in_hand"]),
        "pot_odds": ctx["pot_odds"],
        "spr": ctx["spr"],
        "spr_bucket": ctx["spr_bucket"],
        "stack_bucket": ctx["stack_bucket"],
        "primary_opponent": primary_opponent,
        "opponent_profile": opponent_profile,
        "opponent_tag": opponent_tag,
        "opponent_confidence": opponent_confidence,
        "opponent_adjustment": adjustment_hint,
        "range_bucket": range_bucket,
        "range_adjustments": range_adjustments,
        "bet_facing": ctx["amount_owed"] > 0,
        "cbet_quality": get_cbet_quality(state, hand_bucket, texture, initiative, ctx["active_opponents"]),
        "action_menu": generate_action_menu(ctx, stack_ctx),
        "variant_name": RESOLVED_VARIANT_NAME,
        "feature_flags": FEATURE_FLAGS,
        "feature_status": FEATURE_STATUS,
    }


def postflop_decide(state: dict, info: dict) -> dict:
    """Route to a simple street-specific postflop baseline."""
    street = info.get("street")
    if street == "flop":
        return apply_postflop_guard_variant(state, decide_flop(state, info), info)
    if street == "turn":
        return apply_postflop_guard_variant(state, decide_turn(state, info), info)
    if street == "river":
        action = river_decision(
            info["ctx"],
            info["stack_ctx"],
            info,
            info["board_facts"],
            info.get("opponent_profile"),
            info.get("range_bucket"),
        )
        return apply_postflop_guard_variant(state, action, info)
    return make_fold_or_check(state)


def decide_flop(state: dict, info: dict) -> dict:
    """Baseline flop strategy: value bet good hands, c-bet dry boards, call draws."""
    ctx = info["ctx"]
    stack_ctx = info["stack_ctx"]
    action = {"action": "check"} if ctx["can_check"] else make_fold_or_check(state)
    if ctx["can_check"]:
        if should_value_bet(state, info):
            action = choose_stack_aware_bet_size(ctx, stack_ctx, info, info["board_facts"], "value")
            if info["opponent_tag"] == "MANIAC" and info["hand_bucket"] == "strong" and not info["multiway"]:
                action = maybe_mix_bet_check(
                    ctx,
                    action,
                    {"action": "check"},
                    MIX_FREQUENCIES["maniac_trap"],
                )
        elif should_protection_bet(state, info):
            action = choose_stack_aware_bet_size(ctx, stack_ctx, info, info["board_facts"], "protection")
            if info["hand_bucket"] == "medium":
                action = maybe_mix_bet_check(
                    ctx,
                    action,
                    {"action": "check"},
                    MIX_FREQUENCIES["medium_protection"],
                )
        elif should_cbet(state, info):
            action = choose_stack_aware_bet_size(ctx, stack_ctx, info, info["board_facts"], "cbet")
            cbet_freq = MIX_FREQUENCIES["neutral_cbet"]
            if info.get("cbet_quality") == "good":
                cbet_freq = MIX_FREQUENCIES["good_cbet"]
            action = maybe_mix_bet_check(ctx, action, {"action": "check"}, cbet_freq)
        elif should_semibluff(state, info):
            action = choose_stack_aware_bet_size(ctx, stack_ctx, info, info["board_facts"], "semibluff")
            action = maybe_mix_bet_check(
                ctx,
                action,
                {"action": "check"},
                MIX_FREQUENCIES["marginal_semibluff"],
            )
        return apply_exploit_adjustments(action, ctx, stack_ctx, info, info["board_facts"], info.get("opponent_profile"))

    if should_value_bet(state, info) and can_raise(state):
        if should_commit_stack(ctx, stack_ctx, info, info["board_facts"], info.get("opponent_profile")) or info["spr_bucket"] in {"committed", "low", "medium"}:
            action = choose_stack_aware_raise_size(ctx, stack_ctx, info, info["board_facts"], "value")
    elif should_check_raise(state, info) and can_raise(state):
        purpose = "value" if info["hand_bucket"] in {"very_strong", "strong"} else "semibluff"
        if should_commit_stack(ctx, stack_ctx, info, info["board_facts"], info.get("opponent_profile")) or purpose == "semibluff":
            action = choose_stack_aware_raise_size(ctx, stack_ctx, info, info["board_facts"], purpose)
    elif should_call_postflop(state, info):
        action = {"action": "call"}
    return apply_exploit_adjustments(action, ctx, stack_ctx, info, info["board_facts"], info.get("opponent_profile"))


def decide_turn(state: dict, info: dict) -> dict:
    """Baseline turn strategy tightens medium hands and values strong made hands."""
    ctx = info["ctx"]
    stack_ctx = info["stack_ctx"]
    action = {"action": "check"} if ctx["can_check"] else make_fold_or_check(state)
    if ctx["can_check"]:
        if should_value_bet(state, info):
            action = choose_stack_aware_bet_size(ctx, stack_ctx, info, info["board_facts"], "value")
            if info["opponent_tag"] == "MANIAC" and info["hand_bucket"] == "strong" and not info["multiway"]:
                action = maybe_mix_bet_check(
                    ctx,
                    action,
                    {"action": "check"},
                    MIX_FREQUENCIES["maniac_trap"],
                )
        elif should_protection_bet(state, info):
            action = choose_stack_aware_bet_size(ctx, stack_ctx, info, info["board_facts"], "protection")
            if info["hand_bucket"] == "medium":
                action = maybe_mix_bet_check(
                    ctx,
                    action,
                    {"action": "check"},
                    MIX_FREQUENCIES["medium_protection"],
                )
        elif should_semibluff(state, info) and not info["multiway"]:
            action = choose_stack_aware_bet_size(ctx, stack_ctx, info, info["board_facts"], "semibluff")
            action = maybe_mix_bet_check(
                ctx,
                action,
                {"action": "check"},
                MIX_FREQUENCIES["marginal_semibluff"],
            )
        return apply_exploit_adjustments(action, ctx, stack_ctx, info, info["board_facts"], info.get("opponent_profile"))

    if should_value_bet(state, info) and can_raise(state):
        if should_commit_stack(ctx, stack_ctx, info, info["board_facts"], info.get("opponent_profile")) or info["spr_bucket"] in {"committed", "low"}:
            action = choose_stack_aware_raise_size(ctx, stack_ctx, info, info["board_facts"], "value")
    elif should_check_raise(state, info) and can_raise(state):
        purpose = "value" if info["hand_bucket"] in {"very_strong", "strong"} else "semibluff"
        if should_commit_stack(ctx, stack_ctx, info, info["board_facts"], info.get("opponent_profile")) or purpose == "semibluff":
            action = choose_stack_aware_raise_size(ctx, stack_ctx, info, info["board_facts"], purpose)
    elif should_call_postflop(state, info):
        action = {"action": "call"}
    return apply_exploit_adjustments(action, ctx, stack_ctx, info, info["board_facts"], info.get("opponent_profile"))


def river_decision(ctx, stack_ctx, hand_info, board_info, opp_profile, range_bucket):
    """Dedicated river model: conservative, range-aware, and no Monte Carlo."""
    if bool((ctx or {}).get("can_check")):
        action = river_bet_decision(ctx, stack_ctx, hand_info, board_info, opp_profile, range_bucket)
    else:
        action = river_call_decision(ctx, stack_ctx, hand_info, board_info, opp_profile, range_bucket)
    return apply_exploit_adjustments(action, ctx, stack_ctx, hand_info, board_info, opp_profile)


def river_call_decision(ctx, stack_ctx, hand_info, board_info, opp_profile, range_bucket):
    """Handle river facing-bet decisions with conservative bluff-catch logic."""
    hand_bucket = str((hand_info or {}).get("hand_bucket") or "")
    hand_category = str((hand_info or {}).get("hand_category") or "")
    pot = max(0, _as_int((ctx or {}).get("pot")))
    amount_owed = max(0, _as_int((ctx or {}).get("amount_owed")))
    tag, confidence = classify_opponent(opp_profile)
    bucket = str(range_bucket or "")
    mdf = minimum_defence_frequency(pot, amount_owed)
    pot_odds = float((hand_info or {}).get("pot_odds") or 0.0)
    multiway = bool((hand_info or {}).get("multiway"))
    river_aggression = float((opp_profile or {}).get("river_aggression", 0.0))

    if hand_bucket == "very_strong":
        return {"action": "call"}
    if multiway and hand_bucket != "very_strong":
        return make_fold_or_check(ctx)

    if hand_bucket == "strong":
        if tag in {"NIT"} or bucket in {"value_heavy", "very_tight", "tight"}:
            return {"action": "call"} if pot_odds <= 0.20 else make_fold_or_check(ctx)
        if tag == "MANIAC" or bucket == "polar" or river_aggression >= 0.18:
            threshold = max(0.18, min(0.40, mdf))
            return {"action": "call"} if pot_odds <= threshold else make_fold_or_check(ctx)
        return {"action": "call"} if pot_odds <= float(get_config("river.strong_call_base", 0.28)) else make_fold_or_check(ctx)

    if hand_bucket == "medium":
        if tag == "MANIAC" and confidence >= 0.35 and float((stack_ctx or {}).get("spr") or 0.0) < 6.0:
            if pot_odds <= min(0.20, mdf):
                return maybe_mix_call_fold(ctx, {"action": "call"}, make_fold_or_check(ctx), 0.60)
            return make_fold_or_check(ctx)
        if bucket == "polar" and river_aggression >= 0.20:
            if pot_odds <= min(0.18, mdf):
                return maybe_mix_call_fold(ctx, {"action": "call"}, make_fold_or_check(ctx), 0.55)
            return make_fold_or_check(ctx)
        return {"action": "call"} if pot_odds <= float(get_config("river.medium_tiny_call_base", 0.10)) and _is_tiny_call(ctx) else make_fold_or_check(ctx)

    return make_fold_or_check(ctx)


def river_bet_decision(ctx, stack_ctx, hand_info, board_info, opp_profile, range_bucket):
    """Choose river bet/check lines without solver-style complexity."""
    if river_value_candidate(ctx, hand_info, board_info, opp_profile, range_bucket):
        action = choose_stack_aware_bet_size(ctx, stack_ctx, hand_info, board_info, "river_value")
        if str((hand_info or {}).get("hand_bucket") or "") == "strong" and str((hand_info or {}).get("opponent_tag") or "") == "MANIAC":
            return maybe_mix_bet_check(ctx, action, {"action": "check"}, MIX_FREQUENCIES["maniac_trap"])
        return action
    if river_bluff_candidate(ctx, hand_info, board_info, opp_profile, range_bucket):
        return maybe_mix_bet_check(
            ctx,
            choose_stack_aware_bet_size(ctx, stack_ctx, hand_info, board_info, "semibluff"),
            {"action": "check"},
            MIX_FREQUENCIES["river_bluff"],
        )
    return {"action": "check"}


def river_bluff_candidate(ctx, hand_info, board_info, opp_profile, range_bucket):
    """Allow only selective river bluffs with low showdown value and good targets."""
    if bool((hand_info or {}).get("multiway")):
        return False
    if str((hand_info or {}).get("hand_bucket") or "") != "weak":
        return False
    if float((hand_info or {}).get("showdown_value") or 0.0) > float(get_config("river.bluff_showdown_cap", 0.22)):
        return False
    if str(range_bucket or "") not in {"capped", "loose", "very_loose"}:
        return False
    tag, confidence = classify_opponent(opp_profile)
    if tag == "STATION" and confidence >= 0.25:
        return False
    if tag == "NIT" and confidence >= 0.35:
        return True
    if str((hand_info or {}).get("board_texture") or "") in {"wet", "very_wet"}:
        return False
    return True


def river_value_candidate(ctx, hand_info, board_info, opp_profile, range_bucket):
    """Select clear river value bets and a little thin value versus stations/capped lines."""
    hand_bucket = str((hand_info or {}).get("hand_bucket") or "")
    hand_category = str((hand_info or {}).get("hand_category") or "")
    tag, confidence = classify_opponent(opp_profile)
    bucket = str(range_bucket or "")
    if hand_bucket == "very_strong":
        return True
    if hand_bucket != "strong":
        return False
    if hand_category == "top_pair" and bucket in {"value_heavy", "very_tight", "tight"}:
        return False
    if tag == "STATION" and confidence >= 0.3:
        return hand_category in {"top_pair", "top_pair_good", "overpair", "pocket_pair_over_board"}
    if bucket in {"capped", "loose", "very_loose"}:
        return hand_category in {"top_pair_good", "overpair", "pocket_pair_over_board"}
    return hand_category in {"overpair", "top_pair_good"}


def minimum_defence_frequency(pot, bet) -> float:
    """Simple MDF guardrail for aggressive or polar river lines."""
    pot_value = float(max(0, _as_int(pot)))
    bet_value = float(max(0, _as_int(bet)))
    if pot_value <= 0 or bet_value <= 0:
        return 0.0
    return max(0.0, min(1.0, pot_value / (pot_value + bet_value)))


def should_value_bet(state: dict, info: dict) -> bool:
    """Bet strong made hands more often, but respect wet boards and high SPR."""
    threshold = exploit_adjusted_value_threshold(info["ctx"], info, info.get("opponent_profile"))
    if info["hand_bucket"] == "very_strong":
        return True
    if info["hand_bucket"] != "strong":
        return False
    if info["multiway"] and info["board_texture"] in {"wet", "very_wet"}:
        return threshold <= 0.42
    if info["spr_bucket"] in {"high", "very_high"} and info["board_texture"] in {"wet", "very_wet"}:
        return threshold <= 0.38
    if info["hand_category"] not in {"overpair", "top_pair_good", "top_pair", "pocket_pair_over_board"}:
        return False
    if info["hand_category"] == "top_pair" and info["board_texture"] in {"wet", "very_wet"}:
        return threshold <= 0.40
    return True


def should_protection_bet(state: dict, info: dict) -> bool:
    """Bet medium-strength made hands to deny equity in safe, non-bloated pots."""
    if info["multiway"]:
        return False
    if info["street"] == "river":
        return False
    if info["hand_bucket"] == "strong" and info["board_texture"] in {"wet", "very_wet"}:
        return True
    if info["hand_bucket"] != "medium":
        return False
    if info["spr_bucket"] in {"high", "very_high"}:
        return False
    if info["board_texture"] == "very_wet":
        return False
    return info["hand_category"] in {"middle_pair", "top_pair", "weak_pair"} and info["initiative"]


def should_semibluff(state: dict, info: dict) -> bool:
    """Semi-bluff strong draws mainly in heads-up pots with some fold equity."""
    draws = info["draws"]
    bluff_factor = exploit_adjusted_bluff_frequency(info["ctx"], info["board_facts"], info.get("opponent_profile"))
    bluff_factor *= float(info.get("range_adjustments", {}).get("bluff_factor", 1.0))
    bluff_factor *= fold_equity_multiplier_variant(info["ctx"], info)
    if info["multiway"]:
        return False
    if info["street"] == "river":
        return False
    if bluff_factor < 0.9 and not draws.get("combo_draw"):
        return False
    if draws["combo_draw"]:
        return True
    if draws["flush_draw"] and draws["overcards"] >= 1 and info["initiative"] and bluff_factor >= 0.7:
        return True
    if draws["open_ended"] and info["initiative"] and info["board_texture"] != "very_wet" and bluff_factor >= 0.8:
        return True
    return draws["gutshot"] and draws["overcards"] >= 2 and info["initiative"] and info["board_texture"] == "dry" and bluff_factor >= 1.0


def should_cbet(state: dict, info: dict) -> bool:
    """Allow small continuation bets on dry boards with initiative."""
    if not info["initiative"] or info["multiway"]:
        return False
    if info["street"] not in {"flop", "turn"}:
        return False
    if info["hand_bucket"] in {"very_strong", "strong"}:
        return True
    quality = info.get("cbet_quality", "bad")
    bluff_factor = exploit_adjusted_bluff_frequency(info["ctx"], info["board_facts"], info.get("opponent_profile"))
    bluff_factor *= float(info.get("range_adjustments", {}).get("bluff_factor", 1.0))
    bluff_factor *= fold_equity_multiplier_variant(info["ctx"], info)
    if info["hand_bucket"] == "medium":
        return quality == "good" and _mix_frequency(
            state,
            "medium_cbet",
            float(get_config("postflop.cbet_frequencies.medium_good", 0.35)),
        )
    if quality == "good":
        return _mix_frequency(
            state,
            "good_cbet",
            float(get_config("postflop.cbet_frequencies.good", 0.72)) * bluff_factor,
        )
    if quality == "neutral":
        return _mix_frequency(
            state,
            "neutral_cbet",
            float(get_config("postflop.cbet_frequencies.neutral", 0.45)) * bluff_factor,
        )
    if quality == "bad":
        return _mix_frequency(
            state,
            "bad_cbet",
            float(get_config("postflop.cbet_frequencies.bad", 0.20)) * bluff_factor,
        )
    return _mix_frequency(
        state,
        "bad_multiway_cbet",
        float(get_config("postflop.cbet_frequencies.bad_multiway", 0.05)) * bluff_factor,
    )


def should_check_raise(state: dict, info: dict) -> bool:
    """Allow a simple check-raise baseline with strong value or robust draws."""
    if info["ctx"]["can_check"]:
        return False
    if info["multiway"] and info["hand_bucket"] not in {"very_strong"}:
        return False
    if info["hand_bucket"] == "very_strong":
        return info["spr_bucket"] in {"committed", "low", "medium"}
    if info["hand_bucket"] == "strong":
        return info["board_texture"] in {"wet", "very_wet"} and info["spr_bucket"] in {"low", "medium"}
    return info["draws"]["combo_draw"] and not info["multiway"] and info["spr_bucket"] in {"low", "medium"}


def should_call_postflop(state: dict, info: dict) -> bool:
    """Call with strong made hands, good draws, and modest showdown value at price."""
    ctx = info["ctx"]
    call_margin = exploit_adjusted_call_margin(ctx, info["stack_ctx"], info, info.get("opponent_profile"))
    call_margin += float(info.get("range_adjustments", {}).get("call_margin", 0.0))
    baseline_decision = False
    if ctx["stack_bucket"] == "micro" and info["hand_bucket"] not in {"very_strong", "strong"}:
        return False
    if info["hand_bucket"] == "very_strong":
        return True
    if info["hand_bucket"] == "strong":
        if info["multiway"]:
            baseline_decision = info["pot_odds"] <= 0.24 + call_margin
        elif info["board_texture"] in {"wet", "very_wet"} and info["spr_bucket"] in {"high", "very_high"}:
            baseline_decision = info["pot_odds"] <= 0.22 + call_margin
        else:
            baseline_decision = info["pot_odds"] <= 0.35 + call_margin
    elif info["hand_bucket"] == "medium":
        if info["multiway"]:
            return False
        baseline_decision = info["pot_odds"] <= 0.18 + call_margin or (
            info["showdown_value"] >= 0.5 and _is_tiny_call(state)
        )
    elif info["hand_bucket"] == "draw":
        draws = info["draws"]
        if draws["combo_draw"]:
            baseline_decision = info["pot_odds"] <= 0.42 + call_margin
        elif draws["flush_draw"] or draws["open_ended"]:
            baseline_decision = info["pot_odds"] <= 0.30 + call_margin
        elif draws["gutshot"] and draws["overcards"] >= 2 and not info["multiway"]:
            baseline_decision = info["pot_odds"] <= 0.18 + call_margin
        else:
            baseline_decision = False
    else:
        baseline_decision = False

    if should_use_monte_carlo(ctx, info["stack_ctx"], info, decision_type="call"):
        mc_decision = mc_call_decision(ctx, info["stack_ctx"], info)
        if mc_decision is not None:
            return mc_decision
    return baseline_decision


def choose_bet_size(state: dict, info: dict, purpose: str) -> int:
    """Choose small, medium, large, or jam sizing based on hand and texture."""
    ctx = info["ctx"]
    if info["spr_bucket"] in {"committed", "low"} and purpose in {"value", "semibluff"}:
        if info["hand_bucket"] == "very_strong" or info["draws"]["combo_draw"]:
            return _max_total_bet(state)

    fraction = 0.33
    if purpose == "cbet":
        fraction = 0.33 if info["board_texture"] in {"dry", "semi_dry"} else 0.55
    elif purpose == "protection":
        fraction = 0.55 if info["board_texture"] in {"wet", "very_wet"} else 0.33
    elif purpose == "semibluff":
        fraction = 0.55 if info["draws"]["combo_draw"] else 0.33
    elif purpose == "river_value":
        fraction = 0.55 if info["hand_bucket"] == "strong" else 0.75
    elif purpose == "value":
        if info["board_texture"] in {"wet", "very_wet"} or info["multiway"]:
            fraction = 0.75
        elif info["hand_bucket"] == "strong":
            fraction = 0.55
        else:
            fraction = 0.75

    target = int(round(ctx["pot"] * fraction))
    if target <= 0:
        target = _default_postflop_value_size(ctx)
    return max(ctx["min_raise_to"], target)


def build_stack_context(ctx: dict) -> dict:
    """Create a compact stack context for Stage 8 commitment and sizing."""
    if not isinstance(ctx, dict):
        return {
            "hero_stack_bb": 0.0,
            "effective_stack": 0,
            "spr": 0.0,
            "spr_bucket": "committed",
            "stack_bucket": "micro",
        }
    return {
        "hero_stack_bb": float(ctx.get("stack_bb", 0.0) or 0.0),
        "effective_stack": max(0, _as_int(ctx.get("effective_stack"))),
        "spr": float(ctx.get("spr", 0.0) or 0.0),
        "spr_bucket": str(ctx.get("spr_bucket") or "committed"),
        "stack_bucket": str(ctx.get("stack_bucket") or "micro"),
    }


def generate_action_menu(ctx, stack_ctx) -> list:
    """Return the stack-aware action menu for the current SPR bucket."""
    spr_bucket = str((stack_ctx or {}).get("spr_bucket") or (ctx or {}).get("spr_bucket") or "committed")
    if spr_bucket in {"committed", "low"}:
        return ["check_fold", "call", "bet_33", "bet_66", "jam"]
    if spr_bucket == "medium":
        return ["check_fold", "call", "bet_33", "bet_50", "bet_75", "bet_100", "jam_if_natural"]
    return ["check_fold", "call", "bet_25", "bet_50", "bet_75", "bet_125", "jam_rare"]


def commitment_threshold(ctx, stack_ctx, hand_info, opp_profile=None) -> float:
    """Return the minimum confidence score needed to play for stacks."""
    spr_bucket = str((stack_ctx or {}).get("spr_bucket") or "committed")
    thresholds = get_config("stacks.commitment_thresholds", {})
    threshold = float(thresholds.get("high", 0.82))
    if spr_bucket == "committed":
        threshold = float(thresholds.get("committed", 0.35))
    elif spr_bucket == "low":
        threshold = float(thresholds.get("low", 0.48))
    elif spr_bucket == "medium":
        threshold = float(thresholds.get("medium", 0.66))

    if isinstance(opp_profile, dict):
        tag = str(opp_profile.get("style_tag") or "")
        confidence = float(opp_profile.get("confidence") or 0.0)
        if tag == "MANIAC" and confidence >= 0.4:
            threshold -= 0.08
        elif tag == "NIT" and confidence >= 0.4:
            threshold += 0.08

    return max(0.2, min(0.95, threshold))


def should_commit_stack(ctx, stack_ctx, hand_info, board_info, opp_profile=None) -> bool:
    """Prevent accidental high-SPR one-pair stack-offs while allowing clear commits."""
    hand_bucket = str((hand_info or {}).get("hand_bucket") or "")
    hand_category = str((hand_info or {}).get("hand_category") or "")
    draws = (hand_info or {}).get("draws") if isinstance((hand_info or {}).get("draws"), dict) else {}
    spr_bucket = str((stack_ctx or {}).get("spr_bucket") or "")
    board_texture = str((hand_info or {}).get("board_texture") or "")
    safe_board = board_texture in {"dry", "semi_dry"} and not bool((board_info or {}).get("paired"))

    score = 0.20
    if hand_bucket == "very_strong":
        if hand_category in {"set", "trips", "two_pair"}:
            score = 0.88
        elif hand_category in {"straight", "flush", "full_house", "quads", "straight_flush"}:
            score = 0.96
        else:
            score = 0.85
    elif hand_bucket == "strong":
        if hand_category == "overpair" and safe_board:
            score = 0.72
        elif hand_category == "top_pair_good":
            score = 0.60
        elif hand_category == "top_pair":
            score = 0.48
        else:
            score = 0.55
    elif hand_bucket == "draw":
        if draws.get("combo_draw"):
            score = 0.76
        elif draws.get("flush_draw") and draws.get("open_ended"):
            score = 0.74
        elif draws.get("flush_draw") and draws.get("overcards", 0) >= 1:
            score = 0.62
        elif draws.get("open_ended"):
            score = 0.58
        else:
            score = 0.44

    if spr_bucket == "committed":
        if hand_category in {"top_pair_good", "overpair", "two_pair", "set", "trips", "straight", "flush", "full_house", "quads", "straight_flush"}:
            score = max(score, 0.70)
        if draws.get("combo_draw") or draws.get("flush_draw"):
            score = max(score, 0.64)
    elif spr_bucket == "low":
        if hand_category in {"top_pair_good", "overpair", "two_pair", "set", "trips", "straight", "flush", "full_house", "quads", "straight_flush"}:
            score = max(score, 0.68)
        if draws.get("combo_draw"):
            score = max(score, 0.68)
    elif spr_bucket == "medium":
        if hand_category == "overpair" and safe_board:
            score = max(score, 0.70)
        if hand_category in {"two_pair", "set", "trips", "straight", "flush", "full_house", "quads", "straight_flush"}:
            score = max(score, 0.84)
        if draws.get("combo_draw"):
            score = max(score, 0.71)
        if hand_category in {"top_pair", "top_pair_good"}:
            score = min(score, 0.58)
    else:
        if hand_category in {"set", "trips", "straight", "flush", "full_house", "quads", "straight_flush"}:
            score = max(score, 0.88)
        elif hand_category == "two_pair":
            score = max(score, 0.80)
        elif draws.get("combo_draw"):
            score = max(score, 0.78)
        else:
            score = min(score, 0.58)
        if hand_category in {"top_pair", "top_pair_good", "overpair"}:
            score = min(score, 0.56)

    return score >= commitment_threshold(ctx, stack_ctx, hand_info, opp_profile)


def choose_stack_aware_bet_size(ctx, stack_ctx, hand_info, board_info, purpose) -> dict:
    """Choose unopened postflop bet sizing using the Stage 8 action menu."""
    if is_natural_jam(ctx, stack_ctx, hand_info, board_info, purpose):
        return {"action": "all_in"}
    menu = generate_action_menu(ctx, stack_ctx)
    fraction = _action_menu_fraction(menu, hand_info, board_info, purpose, facing_bet=False)
    return {"action": "raise", "amount": _fraction_to_bet_amount(ctx, fraction)}


def choose_stack_aware_raise_size(ctx, stack_ctx, hand_info, board_info, purpose) -> dict:
    """Choose raise sizing when facing action using the Stage 8 menu."""
    if is_natural_jam(ctx, stack_ctx, hand_info, board_info, purpose):
        return {"action": "all_in"}
    menu = generate_action_menu(ctx, stack_ctx)
    fraction = _action_menu_fraction(menu, hand_info, board_info, purpose, facing_bet=True)
    return {"action": "raise", "amount": _fraction_to_bet_amount(ctx, fraction)}


def is_natural_jam(ctx, stack_ctx, hand_info, board_info, purpose) -> bool:
    """Jam only when SPR is low and the stack-off is naturally justified."""
    if purpose in {"cbet", "protection"}:
        return False
    menu = generate_action_menu(ctx, stack_ctx)
    if "jam" not in menu and "jam_if_natural" not in menu:
        return False
    if max(0, _as_int((ctx or {}).get("your_stack"))) <= 0:
        return False
    if not should_commit_stack(ctx, stack_ctx, hand_info, board_info, (hand_info or {}).get("opponent_profile")):
        return False

    spr = float((stack_ctx or {}).get("spr") or 0.0)
    hand_bucket = str((hand_info or {}).get("hand_bucket") or "")
    draws = (hand_info or {}).get("draws") if isinstance((hand_info or {}).get("draws"), dict) else {}

    if spr <= 1.1:
        baseline = hand_bucket in {"very_strong", "strong"} or draws.get("combo_draw") or draws.get("flush_draw")
        if baseline and should_use_monte_carlo(ctx, stack_ctx, hand_info, decision_type="all_in"):
            mc_decision = mc_all_in_decision(ctx, stack_ctx, hand_info)
            return baseline if mc_decision is None else mc_decision
        return baseline
    if spr <= 2.5:
        baseline = hand_bucket == "very_strong" or draws.get("combo_draw")
        if baseline and should_use_monte_carlo(ctx, stack_ctx, hand_info, decision_type="all_in"):
            mc_decision = mc_all_in_decision(ctx, stack_ctx, hand_info)
            return baseline if mc_decision is None else mc_decision
        return baseline
    if spr <= 4.0:
        baseline = hand_bucket == "very_strong" and str((hand_info or {}).get("board_texture") or "") != "very_wet"
        if baseline and should_use_monte_carlo(ctx, stack_ctx, hand_info, decision_type="all_in"):
            mc_decision = mc_all_in_decision(ctx, stack_ctx, hand_info)
            return baseline if mc_decision is None else mc_decision
        return baseline
    return False


def get_cbet_quality(state: dict, hand_bucket: str, texture: str, initiative: bool, active_opponents: int) -> str:
    """Grade c-bet spots into good, neutral, bad, or bad_multiway buckets."""
    if not initiative:
        return "bad"
    if active_opponents >= 2:
        if texture in {"wet", "very_wet"} and hand_bucket == "weak":
            return "bad_multiway"
        return "bad"
    if hand_bucket in {"very_strong", "strong"}:
        return "good"
    if texture == "dry":
        return "good"
    if texture == "semi_dry":
        return "neutral"
    if texture == "wet":
        return "bad"
    return "bad"


def _action_menu_fraction(menu: list, hand_info: dict, board_info: dict, purpose: str, facing_bet: bool) -> float:
    """Map a strategic purpose to one of the allowed Stage 8 menu fractions."""
    hand_bucket = str((hand_info or {}).get("hand_bucket") or "")
    board_texture = str((hand_info or {}).get("board_texture") or "")
    multiway = bool((hand_info or {}).get("multiway"))
    paired = bool((board_info or {}).get("paired"))

    available = []
    for item in menu or []:
        if item == "bet_25":
            available.append(0.25)
        elif item == "bet_33":
            available.append(0.33)
        elif item == "bet_50":
            available.append(0.50)
        elif item == "bet_66":
            available.append(0.66)
        elif item == "bet_75":
            available.append(0.75)
        elif item == "bet_100":
            available.append(1.00)
        elif item == "bet_125":
            available.append(1.25)

    if not available:
        return 0.50

    target = 0.33
    if purpose == "cbet":
        target = 0.33 if board_texture in {"dry", "semi_dry"} else 0.50
    elif purpose == "protection":
        target = 0.50 if board_texture in {"wet", "very_wet"} else 0.33
    elif purpose == "semibluff":
        if hand_bucket == "draw" and (hand_info or {}).get("draws", {}).get("combo_draw"):
            target = 0.66 if facing_bet else 0.50
        else:
            target = 0.33
    elif purpose == "river_value":
        target = 0.50 if hand_bucket == "strong" else 0.75
    elif purpose == "value":
        if multiway or board_texture in {"wet", "very_wet"}:
            target = 0.75
        elif hand_bucket == "strong" or paired:
            target = 0.50
        else:
            target = 0.66

    return min(available, key=lambda value: abs(value - target))


def _fraction_to_bet_amount(ctx: dict, fraction: float) -> int:
    """Convert a menu fraction into a legal total-bet target."""
    pot = max(0, _as_int((ctx or {}).get("pot")))
    min_raise_to = max(0, _as_int((ctx or {}).get("min_raise_to")))
    target = int(round(pot * float(fraction)))
    if target <= 0:
        target = _default_postflop_value_size(ctx or {})
    return max(min_raise_to, target)


def deterministic_random(ctx, salt="") -> float:
    """Generate a stable pseudo-random float from public hand state."""
    if not isinstance(ctx, dict):
        return 0.0
    parts = [
        str(ctx.get("hand_id", "")),
        str(ctx.get("street", "")),
        str(ctx.get("seat_to_act", "")),
        str(ctx.get("pot", "")),
        str(ctx.get("current_bet", "")),
        str(ctx.get("amount_owed", "")),
        str(len(ctx.get("community_cards", []) if isinstance(ctx.get("community_cards"), list) else [])),
        str(len(ctx.get("action_log", []) if isinstance(ctx.get("action_log"), list) else [])),
        str(salt),
    ]
    text = "|".join(parts)
    accumulator = 0
    for index, char in enumerate(text):
        accumulator = (accumulator * 131 + ord(char) + index) % 1000003
    return (accumulator % 10000) / 10000.0


def mix_actions(weighted_actions, ctx) -> dict:
    """Choose among already reasonable actions using deterministic weights."""
    if not isinstance(weighted_actions, list) or not weighted_actions:
        return {"action": "check"} if bool((ctx or {}).get("can_check")) else make_fold_or_check(ctx or {})
    valid = []
    total = 0.0
    for item in weighted_actions:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        weight = float(item.get("weight", 0.0) or 0.0)
        if not isinstance(action, dict) or weight <= 0:
            continue
        valid.append({"action": action, "weight": weight})
        total += weight
    if total <= 0 or not valid:
        return weighted_actions[0].get("action", {"action": "check"})
    roll = deterministic_random(ctx, "mix_actions")
    running = 0.0
    for item in valid:
        running += item["weight"] / total
        if roll <= running:
            return item["action"]
    return valid[-1]["action"]


def maybe_mix_bet_check(ctx, bet_action, check_action, bet_freq):
    """Mix between a bet and a check when both are already reasonable."""
    if not isinstance(bet_action, dict) or not isinstance(check_action, dict):
        return bet_action if isinstance(bet_action, dict) else check_action
    if str(bet_action.get("action") or "") not in {"raise", "all_in"}:
        return bet_action
    if str(check_action.get("action") or "") != "check":
        return bet_action
    freq = max(0.0, min(1.0, float(bet_freq or 0.0)))
    return mix_actions(
        [
            {"action": bet_action, "weight": freq},
            {"action": check_action, "weight": 1.0 - freq},
        ],
        ctx,
    )


def maybe_mix_call_fold(ctx, call_action, fold_action, call_freq):
    """Mix between a marginal call and a fold in reproducible borderline spots."""
    if not isinstance(call_action, dict) or not isinstance(fold_action, dict):
        return call_action if isinstance(call_action, dict) else fold_action
    if str(call_action.get("action") or "") != "call":
        return call_action
    if str(fold_action.get("action") or "") != "fold":
        return call_action
    freq = max(0.0, min(1.0, float(call_freq or 0.0)))
    return mix_actions(
        [
            {"action": call_action, "weight": freq},
            {"action": fold_action, "weight": 1.0 - freq},
        ],
        ctx,
    )


def should_use_monte_carlo(ctx, stack_ctx, hand_info, decision_type="call") -> bool:
    """Use MC only for close flop/turn calls and natural all-in decisions."""
    if eval7 is None:
        return False
    if not isinstance(ctx, dict) or not isinstance(hand_info, dict):
        return False
    street = str(ctx.get("street") or "")
    if street not in {"flop", "turn"}:
        return False
    if bool(ctx.get("can_check")):
        return False
    if decision_type == "call" and max(0, _as_int(ctx.get("amount_owed"))) <= 0:
        return False

    hand_bucket = str(hand_info.get("hand_bucket") or "")
    hand_category = str(hand_info.get("hand_category") or "")
    if hand_bucket == "very_strong" or hand_category in {"straight_flush", "quads", "full_house"}:
        return False
    if hand_bucket == "weak":
        return False

    if decision_type == "call":
        pot_odds = float(hand_info.get("pot_odds") or 0.0)
        draws = hand_info.get("draws") if isinstance(hand_info.get("draws"), dict) else {}
        is_close_price = 0.15 <= pot_odds <= 0.42
        has_reason = hand_bucket in {"strong", "medium", "draw"} or draws.get("flush_draw") or draws.get("open_ended")
        return is_close_price and has_reason

    if decision_type == "all_in":
        spr = float((stack_ctx or {}).get("spr") or 0.0)
        draws = hand_info.get("draws") if isinstance(hand_info.get("draws"), dict) else {}
        return spr <= 4.0 and (hand_bucket in {"strong", "draw"} or draws.get("combo_draw"))

    return False


def estimate_equity(ctx, stack_ctx, hand_info, opp_ranges=None, max_iters=800, time_budget=0.3) -> float:
    """Estimate raw showdown equity with a small eval7 Monte Carlo sample."""
    if eval7 is None or not isinstance(ctx, dict):
        return 0.0
    if max_iters == 800:
        max_iters = int(get_config("monte_carlo.default_iters", 800))
    if time_budget == 0.3:
        time_budget = float(get_config("monte_carlo.default_time_budget", 0.30))
    hero_cards = parse_cards(ctx.get("your_cards") if isinstance(ctx.get("your_cards"), list) else [])
    board_cards = parse_cards(ctx.get("community_cards") if isinstance(ctx.get("community_cards"), list) else [])
    active_opponents = max(1, _as_int(ctx.get("active_opponents")))

    if len(hero_cards) != 2:
        return 0.0
    if len(board_cards) >= 5:
        return 0.0

    deadline = time.perf_counter() + max(0.05, min(0.35, float(time_budget or 0.3)))
    hero_strings = [str(card) for card in hero_cards]
    board_strings = [str(card) for card in board_cards]
    deck = eval7.Deck()
    known_strings = set(hero_strings + board_strings)
    deck.cards = [card for card in deck.cards if str(card) not in known_strings]

    wins = 0.0
    trials = 0
    opp_count = max(1, min(3, active_opponents))
    target_iters = max(100, min(int(max_iters or 800), 1600))
    needed_board = max(0, 5 - len(board_cards))

    while trials < target_iters and time.perf_counter() < deadline:
        try:
            deck.shuffle()
            draw_count = opp_count * 2 + needed_board
            sample = deck.peek(draw_count)
        except Exception:
            break

        if len(sample) < draw_count:
            break

        offset = 0
        villain_hands = []
        for _ in range(opp_count):
            villain_hands.append(sample[offset:offset + 2])
            offset += 2
        runout = board_cards + sample[offset:offset + needed_board]

        hero_value = eval7.evaluate(hero_cards + runout)
        villain_values = [eval7.evaluate(villain + runout) for villain in villain_hands]
        best_villain = max(villain_values) if villain_values else 0

        if hero_value > best_villain:
            wins += 1.0
        elif hero_value == best_villain:
            tie_count = villain_values.count(best_villain)
            wins += 1.0 / float(tie_count + 1)
        trials += 1

    if trials <= 0:
        return 0.0
    return max(0.0, min(1.0, wins / float(trials)))


def realised_equity(raw_equity, ctx, stack_ctx, hand_info) -> float:
    """Apply simple realization adjustments for position, multiway, and SPR."""
    equity = max(0.0, min(1.0, float(raw_equity or 0.0)))
    if not isinstance(ctx, dict) or not isinstance(hand_info, dict):
        return equity

    if bool(hand_info.get("in_position")):
        equity += float(get_config("equity_realisation.in_position_bonus", 0.03))
    else:
        equity -= float(get_config("equity_realisation.out_of_position_penalty", 0.03))
    if bool(hand_info.get("multiway")):
        equity -= float(get_config("equity_realisation.multiway_penalty", 0.05))

    hand_bucket = str(hand_info.get("hand_bucket") or "")
    draws = hand_info.get("draws") if isinstance(hand_info.get("draws"), dict) else {}
    spr = float((stack_ctx or {}).get("spr") or 0.0)
    if spr >= 6.0 and hand_bucket == "strong":
        equity -= float(get_config("equity_realisation.high_spr_strong_penalty", 0.05))
    if spr <= 2.5 and (hand_bucket == "strong" or draws.get("combo_draw")):
        equity += float(get_config("equity_realisation.low_spr_strong_bonus", 0.04))
    if spr >= 6.0 and draws.get("gutshot") and not draws.get("flush_draw"):
        equity -= float(get_config("equity_realisation.high_spr_gutshot_penalty", 0.04))
    bucket = str(hand_info.get("range_bucket") or "")
    if bucket in {"value_heavy", "very_tight", "tight"}:
        equity -= 0.03
    elif bucket in {"capped", "loose", "very_loose"}:
        equity += 0.02
    elif bucket == "polar" and hand_bucket == "strong":
        equity += 0.015

    return max(0.0, min(1.0, equity))


def mc_call_decision(ctx, stack_ctx, hand_info):
    """Refine a close call decision with quick Monte Carlo equity."""
    try:
        raw_equity = estimate_equity(
            ctx,
            stack_ctx,
            hand_info,
            max_iters=int(get_config("monte_carlo.call_iters", 900)),
            time_budget=float(get_config("monte_carlo.call_time_budget", 0.28)),
        )
        if raw_equity <= 0:
            return None
        realized = realised_equity(raw_equity, ctx, stack_ctx, hand_info)
        required = float(hand_info.get("pot_odds") or 0.0)
        margin = float(get_config("monte_carlo.call_margin", 0.02))
        if hand_info.get("draws", {}).get("combo_draw"):
            margin = 0.0
        if realized >= required + margin:
            return True
        if realized <= required - float(get_config("monte_carlo.call_reject_margin", 0.03)):
            return False
    except Exception:
        return None
    return None


def mc_all_in_decision(ctx, stack_ctx, hand_info):
    """Refine natural jam decisions with a quick equity sanity check."""
    try:
        raw_equity = estimate_equity(
            ctx,
            stack_ctx,
            hand_info,
            max_iters=int(get_config("monte_carlo.all_in_iters", 1000)),
            time_budget=float(get_config("monte_carlo.all_in_time_budget", 0.32)),
        )
        if raw_equity <= 0:
            return None
        realized = realised_equity(raw_equity, ctx, stack_ctx, hand_info)
        required = commitment_threshold(ctx, stack_ctx, hand_info, hand_info.get("opponent_profile")) - float(
            get_config("monte_carlo.all_in_threshold_discount", 0.10)
        )
        if realized >= required:
            return True
        if realized <= required - float(get_config("monte_carlo.all_in_reject_margin", 0.05)):
            return False
    except Exception:
        return None
    return None


def update_opponent_model(state: dict):
    """Update per-player public stats from the visible action log."""
    global CURRENT_HAND_ID

    hand_id = state.get("hand_id")
    if hand_id != CURRENT_HAND_ID:
        CURRENT_HAND_ID = hand_id
        if hand_id not in SEEN_HANDS:
            SEEN_HANDS.add(hand_id)
            _mark_players_seen_for_hand(state)

    players = state.get("players") if isinstance(state.get("players"), list) else []
    hero_seat = _as_int(state.get("seat_to_act"))
    for player in players:
        if not isinstance(player, dict):
            continue
        player_id = _get_player_id(player)
        if not player_id:
            continue
        profile = _ensure_opponent_profile(player_id)
        if _as_int(player.get("seat")) == hero_seat:
            profile["is_hero_like"] = True

    action_log = state.get("action_log") if isinstance(state.get("action_log"), list) else []
    for index, entry in enumerate(action_log):
        if not isinstance(entry, dict):
            continue
        action_key = _action_entry_key(hand_id, index, entry)
        if action_key in SEEN_ACTION_KEYS:
            continue
        SEEN_ACTION_KEYS.add(action_key)
        _apply_action_to_profiles(entry, state)


def get_opponent_profile(player_id) -> dict:
    """Return a safe copy of the stored profile for one player."""
    profile = OPPONENTS.get(str(player_id)) if player_id is not None else None
    if not isinstance(profile, dict):
        return _default_profile()
    safe_profile = dict(profile)
    hands_seen = max(1, _as_int(safe_profile.get("hands_seen")))
    total_actions = max(1, _as_int(safe_profile.get("total_actions")))
    raises = _as_int(safe_profile.get("raise_count"))
    calls = _as_int(safe_profile.get("call_count"))
    all_ins = _as_int(safe_profile.get("all_in_count"))
    samples = max(1, _as_int(safe_profile.get("bet_size_samples")))
    safe_profile["VPIP"] = _safe_ratio(safe_profile.get("vpip_count", 0), hands_seen)
    safe_profile["PFR"] = _safe_ratio(safe_profile.get("pfr_count", 0), hands_seen)
    safe_profile["three_bet_frequency"] = _safe_ratio(safe_profile.get("three_bet_count", 0), max(1, raises))
    safe_profile["call_frequency"] = _safe_ratio(calls, total_actions)
    safe_profile["fold_frequency"] = _safe_ratio(safe_profile.get("fold_count", 0), total_actions)
    safe_profile["raise_frequency"] = _safe_ratio(raises, total_actions)
    safe_profile["aggression_factor"] = _safe_ratio(raises + all_ins, max(1, calls))
    safe_profile["all_in_frequency"] = _safe_ratio(all_ins, total_actions)
    safe_profile["vpip"] = safe_profile["VPIP"]
    safe_profile["pfr"] = safe_profile["PFR"]
    safe_profile["three_bet_freq"] = safe_profile["three_bet_frequency"]
    safe_profile["call_freq"] = safe_profile["call_frequency"]
    safe_profile["fold_freq"] = safe_profile["fold_frequency"]
    safe_profile["raise_freq"] = safe_profile["raise_frequency"]
    safe_profile["aggression"] = safe_profile["aggression_factor"]
    safe_profile["all_in_freq"] = safe_profile["all_in_frequency"]
    safe_profile["fold_to_bet"] = _safe_ratio(safe_profile.get("fold_to_bet_count", 0), hands_seen)
    safe_profile["fold_to_cbet"] = _safe_ratio(safe_profile.get("fold_to_cbet_count", 0), hands_seen)
    safe_profile["check_raise_frequency"] = _safe_ratio(safe_profile.get("check_raise_count", 0), total_actions)
    safe_profile["check_raise_freq"] = safe_profile["check_raise_frequency"]
    safe_profile["river_aggression"] = _safe_ratio(safe_profile.get("river_aggression_count", 0), total_actions)
    safe_profile["average_bet_size"] = _safe_ratio(safe_profile.get("bet_size_total", 0), samples)
    safe_profile["recent_aggression"] = _recent_rate(safe_profile.get("recent_actions", []), {"raise", "bet", "all_in"})
    safe_profile["recent_fold_rate"] = _recent_rate(safe_profile.get("recent_actions", []), {"fold"})
    update_leak_vector({}, safe_profile)
    tag, confidence = classify_opponent(safe_profile)
    safe_profile["style_tag"] = tag
    safe_profile["confidence"] = confidence
    return safe_profile


def classify_opponent(profile) -> tuple:
    """Assign a simple style tag and confidence from cumulative public stats."""
    if not isinstance(profile, dict):
        return "UNKNOWN", 0.0

    hands_seen = max(0, _as_int(profile.get("hands_seen")))
    if hands_seen <= 0:
        return "UNKNOWN", 0.0

    vpip = float(profile.get("vpip", _safe_ratio(profile.get("vpip_count", 0), hands_seen)))
    pfr = float(profile.get("pfr", _safe_ratio(profile.get("pfr_count", 0), hands_seen)))
    call_freq = float(profile.get("call_freq", _safe_ratio(profile.get("call_count", 0), max(1, profile.get("total_actions", 0)))))
    raise_freq = float(profile.get("raise_freq", _safe_ratio(profile.get("raise_count", 0), max(1, profile.get("total_actions", 0)))))
    three_bet_freq = float(profile.get("three_bet_freq", _safe_ratio(profile.get("three_bet_count", 0), max(1, profile.get("raise_count", 0)))))
    fold_freq = float(profile.get("fold_freq", _safe_ratio(profile.get("fold_count", 0), max(1, profile.get("total_actions", 0)))))
    all_in_freq = float(profile.get("all_in_freq", _safe_ratio(profile.get("all_in_count", 0), max(1, profile.get("total_actions", 0)))))
    aggression = float(profile.get("aggression", _safe_ratio(profile.get("raise_count", 0) + profile.get("all_in_count", 0), max(1, profile.get("call_count", 0)))))
    recent_aggression = float(profile.get("recent_aggression", 0.0))
    recent_fold_rate = float(profile.get("recent_fold_rate", 0.0))

    tag_cfg = get_config("opponent_tags", {})
    low_hands = int(tag_cfg.get("hands_low_confidence", 20))
    medium_hands = int(tag_cfg.get("hands_medium_confidence", 60))
    if hands_seen < low_hands:
        confidence = min(0.18, hands_seen / 120.0)
    elif hands_seen < medium_hands:
        confidence = min(0.58, 0.18 + (hands_seen - low_hands) / 100.0)
    else:
        confidence = min(0.88, 0.58 + (hands_seen - medium_hands) / 150.0)

    tag = "SOLID"
    nit_cfg = tag_cfg.get("nit", {})
    station_cfg = tag_cfg.get("station", {})
    maniac_cfg = tag_cfg.get("maniac", {})
    solid_cfg = tag_cfg.get("solid", {})
    if vpip < float(nit_cfg.get("vpip_max", 0.18)) and pfr < float(nit_cfg.get("pfr_max", 0.14)) and fold_freq > float(nit_cfg.get("fold_freq_min", 0.32)) and recent_fold_rate > float(nit_cfg.get("recent_fold_min", 0.22)):
        tag = "NIT"
    elif call_freq > float(station_cfg.get("call_freq_min", 0.40)) and raise_freq < float(station_cfg.get("raise_freq_max", 0.18)) and vpip > float(station_cfg.get("vpip_min", 0.24)):
        tag = "STATION"
    elif vpip > float(maniac_cfg.get("vpip_min", 0.36)) and (
        raise_freq > float(maniac_cfg.get("raise_freq_min", 0.24))
        or aggression > float(maniac_cfg.get("aggression_min", 1.8))
        or all_in_freq > float(maniac_cfg.get("all_in_freq_min", 0.08))
        or recent_aggression > float(maniac_cfg.get("recent_aggression_min", 0.35))
    ):
        tag = "MANIAC"
    elif float(solid_cfg.get("vpip_min", 0.16)) <= vpip <= float(solid_cfg.get("vpip_max", 0.34)) and float(solid_cfg.get("pfr_min", 0.12)) <= pfr <= float(solid_cfg.get("pfr_max", 0.28)) and aggression >= float(solid_cfg.get("aggression_min", 0.7)):
        tag = "SOLID"
    else:
        tag = "UNKNOWN"
    if tag == "UNKNOWN":
        confidence *= 0.5
    if three_bet_freq > 0.2 and tag == "MANIAC":
        confidence = min(0.92, confidence + 0.08)
    return tag, max(0.0, min(0.95, confidence))


def opponent_adjustment_hint(ctx, profile) -> dict:
    """Return tiny Stage 10 nudges based on tag and confidence only."""
    tag, confidence = classify_opponent(profile)
    hint = {
        "tag": tag,
        "confidence": confidence,
        "call_delta": 0.0,
        "fold_delta": 0.0,
        "bluff_factor": 1.0,
        "value_factor": 1.0,
    }
    if confidence < 0.2:
        return hint
    if tag == "NIT":
        hint["fold_delta"] = 0.03 if confidence < 0.6 else 0.05
    elif tag == "STATION":
        hint["call_delta"] = 0.02 if confidence < 0.6 else 0.03
        hint["bluff_factor"] = 0.8
        hint["value_factor"] = 1.05
    elif tag == "MANIAC":
        hint["call_delta"] = 0.015 if confidence < 0.6 else 0.025
        hint["bluff_factor"] = 0.85
    return hint


def update_leak_vector(ctx, profile):
    """Derive conservative leak estimates from public cumulative counters."""
    if not isinstance(profile, dict):
        return
    leaks = dict(profile.get("leaks", {})) if isinstance(profile.get("leaks"), dict) else {}
    samples = dict(profile.get("leak_samples", {})) if isinstance(profile.get("leak_samples"), dict) else {}

    leaks["fold_to_steal"] = _sampled_rate(profile.get("steal_fold_count", 0), profile.get("steal_faced_count", 0), 0.50)
    samples["fold_to_steal"] = max(0, _as_int(profile.get("steal_faced_count")))

    leaks["fold_to_flop_cbet"] = _sampled_rate(profile.get("flop_bet_folds", 0), profile.get("flop_bet_faced", 0), 0.50)
    samples["fold_to_flop_cbet"] = max(0, _as_int(profile.get("flop_bet_faced")))

    leaks["fold_to_turn_barrel"] = _sampled_rate(profile.get("turn_bet_folds", 0), profile.get("turn_bet_faced", 0), 0.50)
    samples["fold_to_turn_barrel"] = max(0, _as_int(profile.get("turn_bet_faced")))

    leaks["river_call_rate"] = _sampled_rate(profile.get("river_call_count", 0), profile.get("river_bet_faced", 0), 0.50)
    samples["river_call_rate"] = max(0, _as_int(profile.get("river_bet_faced")))

    leaks["river_raise_freq"] = _sampled_rate(profile.get("river_raise_action_count", 0), max(1, profile.get("river_bet_faced", 0)), 0.10)
    samples["river_raise_freq"] = max(0, _as_int(profile.get("river_bet_faced")))

    leaks["three_bet_freq"] = float(profile.get("three_bet_freq", _safe_ratio(profile.get("three_bet_count", 0), max(1, profile.get("raise_count", 0)))))
    samples["three_bet_freq"] = max(0, _as_int(profile.get("raise_count")))

    leaks["check_raise_flop"] = _sampled_rate(profile.get("flop_check_raise_count", 0), profile.get("flop_check_count", 0), 0.10)
    samples["check_raise_flop"] = max(0, _as_int(profile.get("flop_check_count")))

    leaks["overbet_freq"] = _sampled_rate(profile.get("overbet_count", 0), profile.get("bet_size_samples", 0), 0.10)
    samples["overbet_freq"] = max(0, _as_int(profile.get("bet_size_samples")))

    river_aggr = float(profile.get("river_aggression", 0.0))
    leaks["underbluffs_river"] = max(
        0.0,
        min(
            1.0,
            float(get_config("leaks.underbluff_high", 0.65)) - river_aggr + (0.10 if str(profile.get("style_tag") or "") == "NIT" else 0.0),
        ),
    )
    samples["underbluffs_river"] = max(0, _as_int(profile.get("river_bet_faced", 0)))

    leaks["overcalls_river"] = max(0.0, min(1.0, leaks["river_call_rate"]))
    samples["overcalls_river"] = samples["river_call_rate"]

    profile["leaks"] = leaks
    profile["leak_samples"] = samples


def get_leak(profile, name, default=0.5):
    """Return one leak value with a conservative fallback."""
    if not isinstance(profile, dict):
        return default
    leaks = profile.get("leaks") if isinstance(profile.get("leaks"), dict) else {}
    value = leaks.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def leak_confidence(profile, name):
    """Return a 0..1 confidence score for a specific derived leak."""
    if not isinstance(profile, dict):
        return 0.0
    samples = profile.get("leak_samples") if isinstance(profile.get("leak_samples"), dict) else {}
    sample = max(0, _as_int(samples.get(name)))
    return max(0.0, min(1.0, sample / float(get_config("leaks.confidence_sample_target", 25))))


def apply_leak_adjustments(action, ctx, stack_ctx, hand_info, board_info, opp_profile, range_bucket):
    """Apply small leak-specific nudges without replacing safety-first logic."""
    if not isinstance(action, dict) or not isinstance(opp_profile, dict):
        return action
    street = str((ctx or {}).get("street") or "")
    hand_bucket = str((hand_info or {}).get("hand_bucket") or "")
    action_name = str(action.get("action") or "")

    if street == "flop":
        fold_to_flop = get_leak(opp_profile, "fold_to_flop_cbet", 0.5)
        cbet_conf = leak_confidence(opp_profile, "fold_to_flop_cbet")
        check_raise = get_leak(opp_profile, "check_raise_flop", 0.1)
        check_raise_conf = leak_confidence(opp_profile, "check_raise_flop")

        if action_name == "check" and bool((ctx or {}).get("can_check")) and cbet_conf >= 0.35 and fold_to_flop >= 0.62:
            if bool((hand_info or {}).get("initiative")) and not bool((hand_info or {}).get("multiway")) and str((hand_info or {}).get("board_texture") or "") != "very_wet":
                return choose_stack_aware_bet_size(ctx, stack_ctx, hand_info, board_info, "cbet")
        if action_name in {"raise", "all_in"} and check_raise_conf >= 0.35 and check_raise >= 0.35:
            if hand_bucket in {"weak", "medium"}:
                return {"action": "check"} if bool((ctx or {}).get("can_check")) else action

    if street == "turn":
        fold_to_turn = get_leak(opp_profile, "fold_to_turn_barrel", 0.5)
        turn_conf = leak_confidence(opp_profile, "fold_to_turn_barrel")
        if action_name == "check" and bool((ctx or {}).get("can_check")) and turn_conf >= 0.35 and fold_to_turn >= 0.64:
            if str((hand_info or {}).get("hand_bucket") or "") in {"draw", "strong"} and not bool((hand_info or {}).get("multiway")):
                return choose_stack_aware_bet_size(ctx, stack_ctx, hand_info, board_info, "semibluff")

    if street == "river":
        river_call_rate = get_leak(opp_profile, "river_call_rate", 0.5)
        river_raise_freq = get_leak(opp_profile, "river_raise_freq", 0.1)
        underbluffs = get_leak(opp_profile, "underbluffs_river", 0.5)
        overcalls = get_leak(opp_profile, "overcalls_river", 0.5)
        river_call_conf = leak_confidence(opp_profile, "river_call_rate")
        river_raise_conf = leak_confidence(opp_profile, "river_raise_freq")

        if action_name in {"raise", "all_in"} and hand_bucket == "weak":
            if river_call_conf >= 0.35 and (river_call_rate >= 0.62 or overcalls >= 0.62):
                return {"action": "check"} if bool((ctx or {}).get("can_check")) else make_fold_or_check(ctx)
        if action_name in {"raise", "all_in"} and hand_bucket == "strong":
            if river_call_conf >= 0.35 and river_call_rate >= 0.62:
                return choose_stack_aware_bet_size(ctx, stack_ctx, hand_info, board_info, "river_value")
        if action_name == "call" and river_raise_conf >= 0.35 and (river_raise_freq >= 0.18 or underbluffs >= 0.65):
            if hand_bucket in {"medium", "strong"} and float((stack_ctx or {}).get("spr") or 0.0) >= 3.0:
                return make_fold_or_check(ctx)

    return action


def exploit_adjusted_value_threshold(ctx, hand_info, opp_profile) -> float:
    """Return a small value-betting threshold shift by opponent style."""
    tag, confidence = classify_opponent(opp_profile)
    if confidence < EXPLOIT_ADJUSTMENTS["low_confidence"]:
        return 0.50
    threshold = 0.50
    if tag == "NIT":
        threshold += EXPLOIT_ADJUSTMENTS["nit_value_threshold"] * _confidence_scale(confidence)
    elif tag == "STATION":
        threshold += EXPLOIT_ADJUSTMENTS["station_value_threshold"] * _confidence_scale(confidence)
    elif tag == "MANIAC":
        threshold += EXPLOIT_ADJUSTMENTS["maniac_value_threshold"] * _confidence_scale(confidence)
    return max(0.25, min(0.80, threshold))


def exploit_adjusted_bluff_frequency(ctx, board_info, opp_profile) -> float:
    """Return a multiplier for bluffing and c-bet air frequency."""
    tag, confidence = classify_opponent(opp_profile)
    if confidence < EXPLOIT_ADJUSTMENTS["low_confidence"]:
        return 1.0
    factor = 1.0
    if tag == "NIT":
        factor = EXPLOIT_ADJUSTMENTS["nit_bluff_factor"]
    elif tag == "STATION":
        factor = EXPLOIT_ADJUSTMENTS["station_bluff_factor"]
    elif tag == "MANIAC":
        factor = EXPLOIT_ADJUSTMENTS["maniac_bluff_factor"]
    elif tag == "SOLID":
        factor = EXPLOIT_ADJUSTMENTS["solid_bluff_factor"]
    if bool((board_info or {}).get("paired")) and factor > 1.0:
        factor += 0.05
    return max(0.25, min(1.20, 1.0 + (factor - 1.0) * _confidence_scale(confidence)))


def exploit_adjusted_call_margin(ctx, stack_ctx, hand_info, opp_profile) -> float:
    """Return a small call-threshold adjustment by style and stack context."""
    tag, confidence = classify_opponent(opp_profile)
    if confidence < EXPLOIT_ADJUSTMENTS["low_confidence"]:
        return 0.0
    margin = 0.0
    if tag == "NIT":
        margin = EXPLOIT_ADJUSTMENTS["nit_call_margin"]
    elif tag == "STATION":
        margin = EXPLOIT_ADJUSTMENTS["station_call_margin"]
    elif tag == "MANIAC":
        margin = EXPLOIT_ADJUSTMENTS["maniac_call_margin"]
        if float((stack_ctx or {}).get("spr") or 0.0) >= 6.0 and str((hand_info or {}).get("hand_bucket") or "") == "medium":
            margin -= 0.02
    elif tag == "SOLID":
        margin = EXPLOIT_ADJUSTMENTS["solid_call_margin"]
    return max(-0.08, min(0.06, margin * _confidence_scale(confidence)))


def apply_exploit_adjustments(action, ctx, stack_ctx, hand_info, board_info, opp_profile):
    """Apply small opponent-style nudges without overriding core safety rules."""
    if not isinstance(action, dict):
        return action
    tag, confidence = classify_opponent(opp_profile)
    if confidence < EXPLOIT_ADJUSTMENTS["low_confidence"]:
        return action

    action_name = str(action.get("action") or "")
    hand_bucket = str((hand_info or {}).get("hand_bucket") or "")
    street = str((ctx or {}).get("street") or "")
    can_check_now = bool((ctx or {}).get("can_check"))
    amount_owed = max(0, _as_int((ctx or {}).get("amount_owed")))
    pot = max(0, _as_int((ctx or {}).get("pot")))
    facing_big_aggression = amount_owed > max(0, int(pot * 0.6))

    if tag == "STATION":
        if action_name == "raise" and hand_bucket in {"weak", "draw"} and street == "river":
            return {"action": "check"} if can_check_now else make_call_or_check(ctx) if hand_bucket == "draw" and amount_owed <= int(pot * 0.2) else make_fold_or_check(ctx)
        if action_name == "raise" and hand_bucket == "draw" and str((hand_info or {}).get("board_texture") or "") == "very_wet":
            return {"action": "check"} if can_check_now else make_call_or_check(ctx)
        if action_name == "call" and facing_big_aggression and hand_bucket in {"medium", "draw"} and street in {"turn", "river"}:
            return make_fold_or_check(ctx)

    if tag == "NIT":
        if action_name == "call" and facing_big_aggression and hand_bucket not in {"very_strong"}:
            return make_fold_or_check(ctx)
        if action_name == "raise" and street == "river" and hand_bucket == "strong":
            return {"action": "check"} if can_check_now else {"action": "call"}

    if tag == "MANIAC":
        if action_name == "raise" and can_check_now and hand_bucket == "strong" and float((stack_ctx or {}).get("spr") or 0.0) >= 4.0:
            return {"action": "check"}
        if action_name == "call" and hand_bucket == "medium" and float((stack_ctx or {}).get("spr") or 0.0) >= 8.0 and facing_big_aggression:
            return make_fold_or_check(ctx)

    return action


def apply_anti_punt_overrides(action, ctx, stack_ctx, hand_info, board_info, opp_profile):
    """Prefer safer lines when the chosen action violates hard practical guardrails."""
    if not isinstance(action, dict):
        return action

    if is_high_spr_one_pair_punt(ctx, stack_ctx, hand_info, board_info, action):
        return make_call_or_check(ctx) if _action_is_aggressive(action) else make_fold_or_check(ctx)
    if is_bad_river_bluff(ctx, hand_info, board_info, opp_profile, action):
        return {"action": "check"} if bool((ctx or {}).get("can_check")) else make_fold_or_check(ctx)
    if is_bad_multiway_cbet(ctx, hand_info, board_info, action):
        return {"action": "check"} if bool((ctx or {}).get("can_check")) else make_fold_or_check(ctx)
    if is_bad_big_river_call(ctx, stack_ctx, hand_info, board_info, opp_profile, action):
        return make_fold_or_check(ctx)
    if is_bad_non_nut_draw_chase(ctx, stack_ctx, hand_info, board_info, action):
        return make_fold_or_check(ctx)
    if _low_confidence_exploit_push(opp_profile, action, hand_info):
        return {"action": "check"} if bool((ctx or {}).get("can_check")) else make_call_or_check(ctx)
    if _line_too_strong_for_mc_override(ctx, hand_info, opp_profile, action):
        return make_fold_or_check(ctx) if str(action.get("action") or "") == "call" else action
    return action


def is_high_spr_one_pair_punt(ctx, stack_ctx, hand_info, board_info, action) -> bool:
    """Block high-SPR one-pair raise/jam lines against passive or nitty aggression."""
    if not _action_is_aggressive(action):
        return False
    if float((stack_ctx or {}).get("spr") or 0.0) < 6.0:
        return False
    if str((hand_info or {}).get("hand_bucket") or "") != "strong":
        return False
    if str((hand_info or {}).get("hand_category") or "") not in {"top_pair", "top_pair_good", "overpair", "pocket_pair_over_board"}:
        return False
    if bool((ctx or {}).get("can_check")):
        return False
    profile = hand_info.get("opponent_profile") if isinstance(hand_info, dict) else {}
    tag, confidence = classify_opponent(profile)
    passive = float((profile or {}).get("raise_freq", 0.0)) < 0.18 and float((profile or {}).get("aggression", 0.0)) < 1.0
    return tag == "NIT" or (confidence >= 0.45 and passive)


def is_bad_river_bluff(ctx, hand_info, board_info, opp_profile, action) -> bool:
    """Block river air bluffs into stations and other low-fold profiles."""
    if str((ctx or {}).get("street") or "") != "river":
        return False
    if not _action_is_aggressive(action):
        return False
    if str((hand_info or {}).get("hand_bucket") or "") in {"very_strong", "strong"}:
        return False
    draws = (hand_info or {}).get("draws") if isinstance((hand_info or {}).get("draws"), dict) else {}
    if draws.get("flush_draw") or draws.get("open_ended") or draws.get("combo_draw"):
        return False
    tag, confidence = classify_opponent(opp_profile)
    return tag == "STATION" and confidence >= 0.35


def is_bad_multiway_cbet(ctx, hand_info, board_info, action) -> bool:
    """Block low-equity multiway c-bets on wet boards."""
    if not _action_is_aggressive(action):
        return False
    if not bool((hand_info or {}).get("multiway")):
        return False
    if str((ctx or {}).get("street") or "") not in {"flop", "turn"}:
        return False
    if str((hand_info or {}).get("board_texture") or "") not in {"wet", "very_wet"}:
        return False
    draws = (hand_info or {}).get("draws") if isinstance((hand_info or {}).get("draws"), dict) else {}
    has_equity = str((hand_info or {}).get("hand_bucket") or "") in {"very_strong", "strong", "draw", "medium"}
    has_draw = draws.get("flush_draw") or draws.get("open_ended") or draws.get("combo_draw") or draws.get("gutshot")
    return not has_equity and not has_draw


def is_bad_big_river_call(ctx, stack_ctx, hand_info, board_info, opp_profile, action) -> bool:
    """Block large river bluff-catches versus nits or passive strong-looking lines."""
    if str((ctx or {}).get("street") or "") != "river":
        return False
    if str((action or {}).get("action") or "") != "call":
        return False
    amount_owed = max(0, _as_int((ctx or {}).get("amount_owed")))
    pot = max(0, _as_int((ctx or {}).get("pot")))
    if amount_owed <= max(0, int(pot * 0.5)):
        return False
    if str((hand_info or {}).get("hand_bucket") or "") == "very_strong":
        return False
    if str((hand_info or {}).get("hand_category") or "") in {"straight", "flush", "full_house", "quads", "straight_flush", "set", "trips", "two_pair"}:
        return False
    tag, confidence = classify_opponent(opp_profile)
    passive = float((opp_profile or {}).get("raise_freq", 0.0)) < 0.18 and float((opp_profile or {}).get("river_aggression", 0.0)) < 0.15
    return tag == "NIT" or (confidence >= 0.45 and passive)


def is_bad_non_nut_draw_chase(ctx, stack_ctx, hand_info, board_info, action) -> bool:
    """Block deep multiway draw calls with weak/non-nut draw structures."""
    if str((action or {}).get("action") or "") != "call":
        return False
    if not bool((hand_info or {}).get("multiway")):
        return False
    if float((stack_ctx or {}).get("spr") or 0.0) < 6.0:
        return False
    if str((hand_info or {}).get("hand_bucket") or "") != "draw":
        return False
    draws = (hand_info or {}).get("draws") if isinstance((hand_info or {}).get("draws"), dict) else {}
    non_nut_like = draws.get("gutshot") or (draws.get("flush_draw") and draws.get("overcards", 0) == 0)
    combo = draws.get("combo_draw")
    return non_nut_like and not combo


def _low_confidence_exploit_push(opp_profile, action, hand_info) -> bool:
    """Disable strong exploit-style bluff raises when the read quality is weak."""
    tag, confidence = classify_opponent(opp_profile)
    if confidence >= 0.2:
        return False
    if tag not in {"UNKNOWN", "SOLID", "NIT", "STATION", "MANIAC"}:
        return False
    if not _action_is_aggressive(action):
        return False
    return str((hand_info or {}).get("hand_bucket") or "") in {"weak", "draw"}


def _line_too_strong_for_mc_override(ctx, hand_info, opp_profile, action) -> bool:
    """Do not let MC rescue thin calls when the public line already looks very strong."""
    if str((action or {}).get("action") or "") != "call":
        return False
    if str((ctx or {}).get("street") or "") not in {"turn", "river"}:
        return False
    amount_owed = max(0, _as_int((ctx or {}).get("amount_owed")))
    pot = max(0, _as_int((ctx or {}).get("pot")))
    if amount_owed <= max(0, int(pot * 0.75)):
        return False
    tag, confidence = classify_opponent(opp_profile)
    if tag == "NIT" and confidence >= 0.35:
        return True
    passive = float((opp_profile or {}).get("raise_freq", 0.0)) < 0.18 and float((opp_profile or {}).get("aggression", 0.0)) < 1.0
    return confidence >= 0.45 and passive and str((hand_info or {}).get("hand_bucket") or "") not in {"very_strong"}


def _action_is_aggressive(action: dict) -> bool:
    """True for raise or all-in actions."""
    return str((action or {}).get("action") or "") in {"raise", "all_in"}


def get_primary_opponent(state: dict):
    """Pick one likely villain for small Stage 6 adjustments."""
    players_in_hand = get_players_in_hand(state)
    hero_seat = _as_int(state.get("seat_to_act"))
    candidates = []
    for player in players_in_hand:
        if not isinstance(player, dict):
            continue
        if _as_int(player.get("seat")) == hero_seat:
            continue
        candidates.append(player)
    if not candidates:
        return None
    candidates.sort(
        key=lambda player: (
            -max(0, _as_int(player.get("bet_this_street"))),
            -max(0, _as_int(player.get("stack"))),
        )
    )
    return _get_player_id(candidates[0])


def get_primary_villain_profile(ctx):
    """Return the most relevant villain profile for the current public spot."""
    if not isinstance(ctx, dict):
        return _default_profile()
    primary_id = get_primary_opponent(ctx)
    return get_opponent_profile(primary_id)


def infer_villain_range_bucket(ctx, opp_profile) -> str:
    """Infer a coarse villain range bucket from position, stack, and style."""
    if not isinstance(ctx, dict):
        return "normal"
    position = preflop_position_bucket(ctx.get("position_info", {}))
    players_at_table = max(2, _as_int(ctx.get("players_at_table")))
    stack_bb = float(ctx.get("stack_bb", 0.0) or 0.0)
    tag, confidence = classify_opponent(opp_profile)
    street = str(ctx.get("street") or "")
    current_bet = max(0, _as_int(ctx.get("current_bet")))
    big_blind = max(1, _as_int(ctx.get("big_blind")))

    if street == "preflop":
        if current_bet >= big_blind * 8 and stack_bb >= 40:
            return "very_tight"
        if stack_bb <= 12 and current_bet >= big_blind * 6:
            return "loose"
        if position == "early":
            return "very_tight" if current_bet >= big_blind * 4 else "tight"
        if position == "late" and players_at_table <= 4:
            return "very_loose" if tag == "MANIAC" and confidence >= 0.35 else "loose"
        if position == "blind":
            return "capped"
        return "normal"

    if tag == "NIT" and confidence >= 0.35:
        return "tight"
    if tag == "MANIAC" and confidence >= 0.35:
        return "very_loose"
    if tag == "STATION" and confidence >= 0.35:
        return "capped"
    return "normal"


def update_range_bucket_for_action(range_bucket, action, street, bet_size, board_info, opp_profile, stack_ctx) -> str:
    """Update a coarse range bucket from the latest public action and context."""
    bucket = str(range_bucket or "normal")
    action_name = str(action or "").lower()
    street = str(street or "")
    bet_size = max(0, _as_int(bet_size))
    tag, confidence = classify_opponent(opp_profile)
    spr = float((stack_ctx or {}).get("spr") or 0.0)
    paired = bool((board_info or {}).get("paired"))
    connected = bool((board_info or {}).get("connected"))

    if street == "preflop":
        if action_name in {"call", "check"} and bucket in {"normal", "loose"}:
            return "capped"
        if action_name in {"raise", "all_in"} and spr <= 2.0:
            return "loose"
        if action_name in {"raise", "all_in"} and spr >= 6.0 and bet_size >= 800:
            return "value_heavy" if tag == "NIT" else "very_tight"
        return bucket

    if action_name in {"call", "check"}:
        if street in {"turn", "river"} and not paired:
            return "capped"
        return bucket
    if action_name not in {"raise", "bet", "all_in"}:
        return bucket

    if street in {"turn", "river"} and tag == "NIT" and confidence >= 0.35:
        return "value_heavy"
    if street in {"turn", "river"} and float((opp_profile or {}).get("raise_freq", 0.0)) < 0.18 and confidence >= 0.45:
        return "value_heavy"
    if street in {"flop", "turn"} and tag == "MANIAC" and confidence >= 0.35:
        return "draw_heavy" if connected else "polar"
    if street in {"flop", "turn"} and connected and not paired:
        return "draw_heavy"
    if street == "river":
        return "polar" if tag == "MANIAC" else "value_heavy"
    return bucket


def range_bucket_adjustments(range_bucket, ctx, hand_info, board_info) -> dict:
    """Return small strategy nudges from a coarse range bucket."""
    bucket = str(range_bucket or "normal")
    adjustments = {
        "bluff_factor": 1.0,
        "call_margin": 0.0,
        "value_factor": 1.0,
    }
    if bucket in {"very_tight", "tight", "value_heavy"}:
        adjustments["bluff_factor"] = 0.82
        adjustments["call_margin"] = -0.03
    elif bucket in {"capped", "loose", "very_loose"}:
        adjustments["bluff_factor"] = 1.08
        adjustments["call_margin"] = 0.02
    elif bucket == "draw_heavy":
        adjustments["bluff_factor"] = 0.92
        adjustments["call_margin"] = -0.01
        adjustments["value_factor"] = 1.08
    elif bucket == "polar":
        if str((hand_info or {}).get("hand_bucket") or "") == "strong":
            adjustments["call_margin"] = 0.015
        adjustments["bluff_factor"] = 0.9

    if bool((board_info or {}).get("paired")) and bucket in {"capped", "loose"}:
        adjustments["bluff_factor"] += 0.04
    return adjustments


def _latest_villain_action(ctx) -> str:
    """Extract the most recent non-hero public action name when possible."""
    if not isinstance(ctx, dict):
        return ""
    action_log = ctx.get("action_log") if isinstance(ctx.get("action_log"), list) else []
    hero_seat = _as_int(ctx.get("seat_to_act"))
    for entry in reversed(action_log):
        if not isinstance(entry, dict):
            continue
        seat = _extract_action_seat(entry)
        if seat is not None and seat == hero_seat:
            continue
        return str(entry.get("action") or "").lower()
    return ""


def hero_has_initiative(state: dict) -> bool:
    """Infer initiative from the last aggressive action in the public log."""
    action_log = state.get("action_log") if isinstance(state.get("action_log"), list) else []
    hero_seat = _as_int(state.get("seat_to_act"))
    last_aggressor = None
    for entry in action_log:
        if not isinstance(entry, dict):
            continue
        action_name = str(entry.get("action", "")).lower()
        if action_name not in {"raise", "bet", "all_in"}:
            continue
        seat = _extract_action_seat(entry)
        if seat is not None:
            last_aggressor = seat
    return last_aggressor == hero_seat if last_aggressor is not None else False


def _extract_action_seat(entry: dict):
    """Best-effort seat extraction from public action log entries."""
    for key in ("seat", "seat_id", "player_seat", "actor_seat"):
        if key in entry:
            try:
                return int(entry.get(key))
            except (TypeError, ValueError):
                return None
    return None


def _mark_players_seen_for_hand(state: dict):
    """Increment hands seen once per player at the start of a new hand."""
    players = state.get("players") if isinstance(state.get("players"), list) else []
    for player in players:
        if not isinstance(player, dict):
            continue
        player_id = _get_player_id(player)
        if not player_id:
            continue
        profile = _ensure_opponent_profile(player_id)
        profile["hands_seen"] += 1
        profile["vpip_this_hand"] = False
        profile["pfr_this_hand"] = False
        profile["last_street"] = None


def _apply_action_to_profiles(entry: dict, state: dict):
    """Apply one public action-log entry to the relevant player profile."""
    player_id = _entry_player_id(entry, state)
    if not player_id:
        return
    profile = _ensure_opponent_profile(player_id)

    action_name = str(entry.get("action", "")).lower()
    if not action_name:
        return

    amount = _entry_amount(entry)
    street = str(entry.get("street") or state.get("street") or "")
    seat = _extract_action_seat(entry)
    current_bet = max(0, _as_int(state.get("current_bet")))
    big_blind = max(1, estimate_big_blind(state))
    profile["total_actions"] += 1
    profile["last_street"] = street
    profile["recent_actions"].append(action_name)
    if len(profile["recent_actions"]) > 12:
        profile["recent_actions"] = profile["recent_actions"][-12:]

    if action_name in {"call"}:
        profile["call_count"] += 1
        if street == "river" and current_bet > 0:
            profile["river_call_count"] += 1
    elif action_name in {"fold"}:
        profile["fold_count"] += 1
        if _facing_bet_from_entry(entry, state):
            profile["fold_to_bet_count"] += 1
            if street == "flop":
                profile["fold_to_cbet_count"] += 1
                profile["flop_bet_folds"] += 1
                profile["flop_bet_faced"] += 1
            elif street == "turn":
                profile["turn_bet_folds"] += 1
                profile["turn_bet_faced"] += 1
            elif street == "river":
                profile["river_bet_faced"] += 1
    elif action_name in {"raise", "bet"}:
        profile["raise_count"] += 1
        profile["bet_size_total"] += max(0, amount)
        profile["bet_size_samples"] += 1
        if max(0, amount) > max(1, _as_int(state.get("pot"))) * 1.1:
            profile["overbet_count"] += 1
        if street == "river":
            profile["river_aggression_count"] += 1
            profile["river_raise_action_count"] += 1
        if _is_check_raise_entry(entry, state):
            profile["check_raise_count"] += 1
            if street == "flop":
                profile["flop_check_raise_count"] += 1
        if _is_three_bet_entry(entry, state):
            profile["three_bet_count"] += 1
    elif action_name in {"all_in"}:
        profile["all_in_count"] += 1
        profile["raise_count"] += 1
        profile["bet_size_total"] += max(0, amount)
        profile["bet_size_samples"] += 1
        if max(0, amount) > max(1, _as_int(state.get("pot"))) * 1.1:
            profile["overbet_count"] += 1
        if street == "river":
            profile["river_aggression_count"] += 1
            profile["river_raise_action_count"] += 1
    elif action_name in {"check"}:
        profile["check_count"] += 1
        if street == "flop":
            profile["flop_check_count"] += 1

    if action_name in {"call", "raise", "bet", "all_in"} and not profile["vpip_this_hand"] and street == "preflop":
        profile["vpip_count"] += 1
        profile["vpip_this_hand"] = True
    if action_name in {"raise", "bet", "all_in"} and not profile["pfr_this_hand"] and street == "preflop":
        profile["pfr_count"] += 1
        profile["pfr_this_hand"] = True

    if street == "preflop" and seat is not None and _is_blind_seat(state, seat) and current_bet > big_blind:
        profile["steal_faced_count"] += 1
        if action_name == "fold":
            profile["steal_fold_count"] += 1
    if street == "flop" and current_bet > 0 and action_name in {"call", "fold"}:
        profile["flop_bet_faced"] += 1
    if street == "turn" and current_bet > 0 and action_name in {"call", "fold"}:
        profile["turn_bet_faced"] += 1
    if street == "river" and current_bet > 0 and action_name in {"call", "fold", "raise", "bet", "all_in"}:
        profile["river_bet_faced"] += 1


def _ensure_opponent_profile(player_id) -> dict:
    """Create the stats bucket for a player if it does not exist."""
    key = str(player_id)
    if key not in OPPONENTS:
        OPPONENTS[key] = _default_profile()
    return OPPONENTS[key]


def _default_profile() -> dict:
    """Default opponent profile with cumulative counters."""
    return {
        "hands_seen": 0,
        "vpip_count": 0,
        "pfr_count": 0,
        "three_bet_count": 0,
        "call_count": 0,
        "fold_count": 0,
        "raise_count": 0,
        "check_count": 0,
        "all_in_count": 0,
        "fold_to_bet_count": 0,
        "fold_to_cbet_count": 0,
        "check_raise_count": 0,
        "river_aggression_count": 0,
        "bet_size_total": 0,
        "bet_size_samples": 0,
        "total_actions": 0,
        "vpip_this_hand": False,
        "pfr_this_hand": False,
        "last_street": None,
        "is_hero_like": False,
        "recent_actions": [],
        "steal_faced_count": 0,
        "steal_fold_count": 0,
        "flop_bet_faced": 0,
        "flop_bet_folds": 0,
        "turn_bet_faced": 0,
        "turn_bet_folds": 0,
        "river_bet_faced": 0,
        "river_call_count": 0,
        "river_raise_action_count": 0,
        "flop_check_count": 0,
        "flop_check_raise_count": 0,
        "overbet_count": 0,
        "leaks": {},
        "leak_samples": {},
    }


def _recent_rate(actions, targets) -> float:
    """Compute a simple recent action rate over a short rolling window."""
    if not isinstance(actions, list) or not actions:
        return 0.0
    matches = 0
    total = 0
    for action in actions[-12:]:
        if not isinstance(action, str):
            continue
        total += 1
        if action in targets:
            matches += 1
    if total <= 0:
        return 0.0
    return matches / float(total)


def _sampled_rate(successes, samples, default=0.5) -> float:
    """Blend sparse observations toward a conservative default."""
    sample_count = max(0, _as_int(samples))
    if sample_count <= 0:
        return float(default)
    success_count = max(0, _as_int(successes))
    weight = min(1.0, sample_count / 20.0)
    observed = success_count / float(sample_count)
    return max(0.0, min(1.0, default * (1.0 - weight) + observed * weight))


def _is_blind_seat(state, seat: int) -> bool:
    """Best-effort blind-seat inference from current active seat ordering."""
    players = state.get("players") if isinstance(state.get("players"), list) else []
    seats = sorted(_as_int(player.get("seat")) for player in players if isinstance(player, dict))
    if len(seats) < 2:
        return False
    return seat in {seats[-1], seats[-2]}


def _confidence_scale(confidence: float) -> float:
    """Scale exploit adjustments so low-confidence reads stay close to baseline."""
    return max(0.0, min(1.0, (float(confidence or 0.0) - 0.2) / 0.6))


def _get_player_id(player: dict):
    """Extract a stable player identifier from public player info."""
    for key in ("bot_id", "player_id", "id", "name"):
        value = player.get(key) if isinstance(player, dict) else None
        if value not in (None, ""):
            return str(value)
    seat = player.get("seat") if isinstance(player, dict) else None
    return "seat_{}".format(seat) if seat is not None else None


def _entry_player_id(entry: dict, state: dict):
    """Resolve the acting player from a log entry using id or seat."""
    for key in ("bot_id", "player_id", "id", "name"):
        value = entry.get(key)
        if value not in (None, ""):
            return str(value)

    seat = _extract_action_seat(entry)
    if seat is None:
        return None
    players = state.get("players") if isinstance(state.get("players"), list) else []
    for player in players:
        if not isinstance(player, dict):
            continue
        if _as_int(player.get("seat")) == seat:
            return _get_player_id(player)
    return "seat_{}".format(seat)


def _action_entry_key(hand_id, index: int, entry: dict):
    """Build a stable key so repeated decide() calls do not double-count actions."""
    seat = _extract_action_seat(entry)
    action_name = entry.get("action")
    amount = _entry_amount(entry)
    street = entry.get("street")
    return (str(hand_id), index, seat, str(action_name), amount, str(street))


def _entry_amount(entry: dict) -> int:
    """Best-effort action size extraction from a public log entry."""
    for key in ("amount", "raise_to", "bet_to", "total", "chips"):
        if key in entry:
            return max(0, _as_int(entry.get(key)))
    return 0


def _facing_bet_from_entry(entry: dict, state: dict) -> bool:
    """Approximate whether a fold happened facing a meaningful wager."""
    amount = _entry_amount(entry)
    if amount > 0:
        return True
    current_bet = max(0, _as_int(state.get("current_bet")))
    return current_bet > 0


def _is_three_bet_entry(entry: dict, state: dict) -> bool:
    """Approximate 3-bets from preflop re-raise sizing in the public log."""
    street = str(entry.get("street") or state.get("street") or "")
    if street != "preflop":
        return False
    amount = _entry_amount(entry)
    big_blind = estimate_big_blind(state)
    return amount >= big_blind * 6 if big_blind > 0 else False


def _is_check_raise_entry(entry: dict, state: dict) -> bool:
    """Best-effort check-raise detection from recent same-street log entries."""
    seat = _extract_action_seat(entry)
    if seat is None:
        return False
    street = str(entry.get("street") or state.get("street") or "")
    action_log = state.get("action_log") if isinstance(state.get("action_log"), list) else []
    seen_check = False
    for previous in action_log:
        if previous is entry:
            break
        if not isinstance(previous, dict):
            continue
        if str(previous.get("street") or state.get("street") or "") != street:
            continue
        if _extract_action_seat(previous) != seat:
            continue
        if str(previous.get("action", "")).lower() == "check":
            seen_check = True
    return seen_check and str(entry.get("action", "")).lower() in {"raise", "bet", "all_in"}


def parse_cards(card_strings) -> list:
    """Safely parse card strings into eval7 cards when available."""
    if eval7 is None or not isinstance(card_strings, list):
        return []
    parsed = []
    for card in card_strings:
        if not isinstance(card, str) or len(card) < 2:
            continue
        try:
            parsed.append(eval7.Card(card))
        except Exception:
            continue
    return parsed


def evaluate_made_hand(state: dict) -> dict:
    """Summarize the visible made hand using simple rank-pattern logic."""
    hole = state.get("your_cards") if isinstance(state.get("your_cards"), list) else []
    board = state.get("community_cards") if isinstance(state.get("community_cards"), list) else []
    all_cards = hole + board
    ranks = _card_ranks(all_cards)
    rank_counts = _rank_count_map(ranks)
    count_values = sorted(rank_counts.values(), reverse=True)
    straight = _has_straight(set(ranks))
    flush = _has_flush(all_cards)

    category = "high_card"
    if flush and straight:
        category = "straight_flush"
    elif count_values and count_values[0] >= 4:
        category = "quads"
    elif len(count_values) >= 2 and count_values[0] >= 3 and count_values[1] >= 2:
        category = "full_house"
    elif flush:
        category = "flush"
    elif straight:
        category = "straight"
    elif count_values and count_values[0] >= 3:
        category = "set" if _hero_pair_rank(hole, rank_counts) else "trips"
    elif len(count_values) >= 2 and count_values[0] >= 2 and count_values[1] >= 2:
        category = "two_pair"
    elif count_values and count_values[0] >= 2:
        category = _pair_category(hole, board, rank_counts)

    eval7_value = None
    parsed_cards = parse_cards(all_cards)
    if eval7 is not None and len(parsed_cards) >= 5:
        try:
            eval7_value = eval7.evaluate(parsed_cards)
        except Exception:
            eval7_value = None

    return {
        "category": category,
        "rank_counts": rank_counts,
        "is_flush": flush,
        "is_straight": straight,
        "eval7_value": eval7_value,
    }


def get_hand_category(state: dict) -> str:
    """Return a simple human-readable made-hand category."""
    return evaluate_made_hand(state).get("category", "high_card")


def has_pair_or_better(state: dict) -> bool:
    """True when the current hand has at least one made pair."""
    return get_hand_category(state) != "high_card"


def detect_draws(state: dict) -> dict:
    """Collect lightweight draw flags for early postflop decisions."""
    flush_info = detect_flush_draw(state)
    straight_info = detect_straight_draw(state)
    overcards = detect_overcards(state)
    combo_draw = (
        (flush_info["flush_draw"] and (straight_info["open_ended"] or straight_info["gutshot"]))
        or (flush_info["flush_draw"] and overcards >= 2)
    )
    return {
        "flush_draw": flush_info["flush_draw"],
        "backdoor_flush_draw": flush_info["backdoor_flush_draw"],
        "open_ended": straight_info["open_ended"],
        "gutshot": straight_info["gutshot"],
        "overcards": overcards,
        "combo_draw": combo_draw,
    }


def detect_flush_draw(state: dict) -> dict:
    """Detect direct and backdoor flush potential from suit counts."""
    hole = state.get("your_cards") if isinstance(state.get("your_cards"), list) else []
    board = state.get("community_cards") if isinstance(state.get("community_cards"), list) else []
    cards = hole + board
    suits = [card[1] for card in cards if isinstance(card, str) and len(card) >= 2]
    if not suits:
        return {"flush_draw": False, "backdoor_flush_draw": False}

    max_suit = max(suits.count(suit) for suit in "shdc")
    return {
        "flush_draw": max_suit == 4,
        "backdoor_flush_draw": len(board) == 3 and max_suit == 3,
    }


def detect_straight_draw(state: dict) -> dict:
    """Detect open-ended and gutshot straight draws from visible ranks."""
    hole = state.get("your_cards") if isinstance(state.get("your_cards"), list) else []
    board = state.get("community_cards") if isinstance(state.get("community_cards"), list) else []
    rank_values = sorted(set(_card_rank_values(hole + board)))
    if len(rank_values) < 4:
        return {"open_ended": False, "gutshot": False}

    open_ended = False
    gutshot = False
    for start in range(1, 11):
        window = {start, start + 1, start + 2, start + 3, start + 4}
        window_values = set(rank_values)
        if 14 in window_values:
            window_values.add(1)
        overlap = len(window.intersection(window_values))
        if overlap != 4:
            continue
        missing = sorted(window.difference(window_values))
        if not missing:
            continue
        if missing[0] in {min(window), max(window)}:
            open_ended = True
        else:
            gutshot = True

    return {"open_ended": open_ended, "gutshot": gutshot and not open_ended}


def detect_overcards(state: dict) -> int:
    """Count how many hole cards outrank the current top board card."""
    hole = state.get("your_cards") if isinstance(state.get("your_cards"), list) else []
    board = state.get("community_cards") if isinstance(state.get("community_cards"), list) else []
    if not hole or not board:
        return 0
    board_top = max(_card_rank_values(board), default=0)
    return sum(1 for value in _card_rank_values(hole) if value > board_top)


def analyse_board_texture(state: dict) -> dict:
    """Return simple board features used to classify texture."""
    board = state.get("community_cards") if isinstance(state.get("community_cards"), list) else []
    return {
        "paired": is_board_paired(board),
        "monotone": is_board_monotone(board),
        "two_tone": is_board_two_tone(board),
        "connected": is_board_connected(board),
        "broadway_heavy": sum(1 for rank in _card_rank_values(board) if rank >= 10) >= 2,
        "low_connected": _is_low_connected_board(board),
        "high_card": board_high_card_rank(board),
    }


def is_board_paired(board) -> bool:
    """True when the board contains any repeated rank."""
    ranks = _card_ranks(board)
    return len(ranks) != len(set(ranks))


def is_board_monotone(board) -> bool:
    """True when every visible board card shares the same suit."""
    suits = [card[1] for card in board if isinstance(card, str) and len(card) >= 2]
    return len(suits) >= 3 and len(set(suits)) == 1


def is_board_two_tone(board) -> bool:
    """True when exactly two suits appear on a three-plus card board."""
    suits = [card[1] for card in board if isinstance(card, str) and len(card) >= 2]
    return len(suits) >= 3 and len(set(suits)) == 2


def is_board_connected(board) -> bool:
    """True when rank spacing suggests easy straight interaction."""
    values = sorted(set(_card_rank_values(board)))
    if len(values) < 2:
        return False
    gaps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    return max(gaps, default=99) <= 2 or _is_low_connected_board(board)


def board_high_card_rank(board) -> int:
    """Return the highest board rank as an integer value."""
    return max(_card_rank_values(board), default=0)


def classify_board_texture(state: dict) -> str:
    """Collapse board features into dry to very-wet texture buckets."""
    texture = analyse_board_texture(state)
    wet_score = 0
    if texture["monotone"]:
        wet_score += 3
    elif texture["two_tone"]:
        wet_score += 2
    if texture["connected"]:
        wet_score += 2
    if texture["low_connected"]:
        wet_score += 1
    if texture["broadway_heavy"]:
        wet_score += 1
    if texture["paired"]:
        wet_score -= 1

    if wet_score <= 0:
        return "dry"
    if wet_score <= 2:
        return "semi_dry"
    if wet_score <= 4:
        return "wet"
    return "very_wet"


def estimate_showdown_value(state: dict) -> float:
    """Assign a rough showdown score for conservative call and check logic."""
    category = get_hand_category(state)
    draws = detect_draws(state)
    if category in {"straight_flush", "quads", "full_house", "flush", "straight", "set", "trips", "two_pair"}:
        return 0.95
    if category in {"overpair", "top_pair_good", "top_pair", "pocket_pair_over_board"}:
        return 0.72
    if category in {"middle_pair", "weak_pair", "underpair"}:
        return 0.52
    if draws["combo_draw"]:
        return 0.46
    if draws["flush_draw"] or draws["open_ended"]:
        return 0.38
    if _has_ace_high(state):
        return 0.26
    return 0.08


def estimate_hand_strength_bucket(state: dict) -> str:
    """Group hands into broad action buckets for the early postflop policy."""
    category = get_hand_category(state)
    draws = detect_draws(state)
    if category in {"straight_flush", "quads", "full_house", "flush", "straight", "set", "trips", "two_pair"}:
        return "very_strong"
    if category in {"overpair", "top_pair_good", "top_pair", "pocket_pair_over_board"}:
        return "strong"
    if draws["combo_draw"] or draws["flush_draw"] or draws["open_ended"]:
        return "draw"
    if category in {"middle_pair", "weak_pair", "underpair"} or _has_ace_high(state):
        return "medium"
    return "weak"


def parse_state(state: dict) -> dict:
    """Build a defensive strategy context from public engine fields."""
    street = state.get("street") or "preflop"
    pot = max(0, _as_int(state.get("pot")))
    amount_owed = max(0, _as_int(state.get("amount_owed")))
    your_stack = max(0, _as_int(state.get("your_stack")))
    current_bet = max(0, _as_int(state.get("current_bet")))
    min_raise_to = max(0, _as_int(state.get("min_raise_to")))
    your_cards = state.get("your_cards") if isinstance(state.get("your_cards"), list) else []
    community_cards = (
        state.get("community_cards") if isinstance(state.get("community_cards"), list) else []
    )
    players = state.get("players") if isinstance(state.get("players"), list) else []
    action_log = state.get("action_log") if isinstance(state.get("action_log"), list) else []
    can_check = bool(state.get("can_check")) or amount_owed == 0
    active_players = get_active_players(state)
    players_in_hand = get_players_in_hand(state)
    players_at_table = len(players)
    effective_stack = get_effective_stack(state)
    stack_bb = get_stack_bb(state)
    spr = get_spr(state)

    return {
        "street": street,
        "pot": pot,
        "amount_owed": amount_owed,
        "can_check": can_check,
        "current_bet": current_bet,
        "min_raise_to": min_raise_to,
        "your_stack": your_stack,
        "your_cards": your_cards,
        "community_cards": community_cards,
        "players": players,
        "action_log": action_log,
        "players_at_table": players_at_table,
        "active_players": active_players,
        "players_in_hand": players_in_hand,
        "active_opponents": max(0, len(players_in_hand) - 1),
        "big_blind": estimate_big_blind(state),
        "effective_stack": effective_stack,
        "stack_bb": stack_bb,
        "spr": spr,
        "stack_bucket": get_stack_bucket(stack_bb),
        "spr_bucket": get_spr_bucket(spr),
        "position_info": get_position_info(state),
        "preflop_position_bucket": preflop_position_bucket(get_position_info(state)),
        "pot_odds": _safe_ratio(amount_owed, pot + amount_owed),
    }


def get_active_players(state: dict) -> list:
    """Return seated players who still have chips or are still marked active."""
    players = state.get("players") if isinstance(state.get("players"), list) else []
    active = []
    for player in players:
        if not isinstance(player, dict):
            continue
        stack = max(0, _as_int(player.get("stack")))
        is_active = bool(player.get("is_active"))
        is_all_in = bool(player.get("is_all_in"))
        if is_active or is_all_in or stack > 0:
            active.append(player)
    return active


def get_players_in_hand(state: dict) -> list:
    """Return players who appear to still be contesting the current hand."""
    players = get_active_players(state)
    in_hand = []
    for player in players:
        if bool(player.get("is_folded")):
            continue
        if bool(player.get("is_active")) or bool(player.get("is_all_in")):
            in_hand.append(player)
    return in_hand


def estimate_big_blind(state: dict) -> int:
    """Infer the big blind from visible betting data without assuming structure."""
    min_raise_to = max(0, _as_int(state.get("min_raise_to")))
    current_bet = max(0, _as_int(state.get("current_bet")))
    amount_owed = max(0, _as_int(state.get("amount_owed")))
    candidates = [value for value in (min_raise_to // 2, current_bet, amount_owed) if value > 0]
    if not candidates:
        return 100
    return max(1, min(candidates))


def get_effective_stack(state: dict) -> int:
    """Estimate the smallest relevant stack among opponents still in the hand."""
    hero_stack = max(0, _as_int(state.get("your_stack")))
    players_in_hand = get_players_in_hand(state)
    opponent_stacks = []
    hero_seat = _as_int(state.get("seat_to_act"))

    for player in players_in_hand:
        seat = _as_int(player.get("seat"))
        if seat == hero_seat:
            continue
        opponent_stacks.append(max(0, _as_int(player.get("stack"))))

    if not opponent_stacks:
        return hero_stack
    return min([hero_stack] + opponent_stacks)


def get_stack_bb(state: dict) -> float:
    """Express our stack in big blinds using the inferred blind size."""
    big_blind = estimate_big_blind(state)
    if big_blind <= 0:
        return 0.0
    return max(0.0, _as_int(state.get("your_stack")) / float(big_blind))


def get_spr(state: dict) -> float:
    """Compute stack-to-pot ratio from the effective stack and current pot."""
    pot = max(0, _as_int(state.get("pot")))
    if pot <= 0:
        return 0.0
    return max(0.0, get_effective_stack(state) / float(pot))


def get_stack_bucket(stack_bb) -> str:
    """Bucket stack depth for later strategy stages."""
    thresholds = get_config("stacks.stack_buckets_bb", {})
    if stack_bb < float(thresholds.get("micro", 8)):
        return "micro"
    if stack_bb < float(thresholds.get("short", 20)):
        return "short"
    if stack_bb < float(thresholds.get("medium", 50)):
        return "medium"
    if stack_bb < float(thresholds.get("deep", 100)):
        return "deep"
    return "very_deep"


def get_spr_bucket(spr) -> str:
    """Bucket SPR so later code can reason about commitment safely."""
    thresholds = get_config("stacks.spr_buckets", {})
    if spr < float(thresholds.get("committed", 1)):
        return "committed"
    if spr < float(thresholds.get("low", 3)):
        return "low"
    if spr < float(thresholds.get("medium", 6)):
        return "medium"
    if spr < float(thresholds.get("high", 12)):
        return "high"
    return "very_high"


def get_position_info(state: dict) -> dict:
    """Infer a simple positional label from seat order and table size."""
    players = get_active_players(state)
    total_players = len(players)
    seat_to_act = _as_int(state.get("seat_to_act"))
    ordered_seats = sorted(_as_int(player.get("seat")) for player in players)

    if seat_to_act not in ordered_seats:
        ordered_seats.append(seat_to_act)
        ordered_seats.sort()

    seat_index = ordered_seats.index(seat_to_act) if ordered_seats else 0
    players_left = max(0, total_players - seat_index - 1)

    label = "unknown"
    is_late_position = False
    if total_players <= 2:
        label = "heads_up_button" if seat_index == 0 else "heads_up_bb"
        is_late_position = seat_index == 0
    elif seat_index == total_players - 1:
        label = "big_blind"
    elif seat_index == total_players - 2:
        label = "small_blind"
    elif seat_index >= max(0, total_players - 3):
        label = "late"
        is_late_position = True
    elif seat_index >= max(0, total_players - 4):
        label = "middle"
    else:
        label = "early"

    return {
        "label": label,
        "seat_index": seat_index,
        "players_left_to_act": players_left,
        "is_late_position": is_late_position,
        "is_blind": label in {"small_blind", "big_blind", "heads_up_bb"},
        "is_heads_up": total_players <= 2,
    }


def preflop_position_bucket(position_info: dict) -> str:
    """Collapse detailed seat info into a small set of preflop buckets."""
    label = position_info.get("label")
    if label in {"heads_up_button", "late"}:
        return "late"
    if label == "middle":
        return "middle"
    if label in {"small_blind", "big_blind", "heads_up_bb"}:
        return "blind"
    return "early"


def normalize_hand(hole_cards) -> str:
    """Convert two cards into a compact preflop code like AKo or T9s."""
    if not isinstance(hole_cards, list) or len(hole_cards) != 2:
        return ""
    first, second = hole_cards
    if not all(isinstance(card, str) and len(card) >= 2 for card in (first, second)):
        return ""

    rank_order = "23456789TJQKA"
    ranks = sorted((first[0], second[0]), key=rank_order.index, reverse=True)
    if ranks[0] == ranks[1]:
        return "".join(ranks)

    suited = first[1] == second[1]
    return "{}{}{}".format(ranks[0], ranks[1], "s" if suited else "o")


def get_open_ranges(players_at_table: int) -> dict:
    """Return raise-first-in ranges widened slightly for shorter tables."""
    early = {
        "AA", "KK", "QQ", "JJ", "TT", "99",
        "AKs", "AQs", "AJs", "ATs", "KQs", "KJs", "QJs",
        "AKo", "AQo",
    }
    middle = early | {
        "88", "77", "66",
        "A9s", "A8s", "KTs", "QTs", "JTs", "T9s", "98s",
        "AJo", "KQo",
    }
    late = middle | {
        "55", "44", "33", "22",
        "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "K9s", "Q9s", "J9s", "87s", "76s", "65s", "54s",
        "ATo", "KJo", "QJo",
    }
    blind = middle | {"A5o", "KTo", "QTo", "97s", "86s"}

    if players_at_table <= 5:
        middle |= {"A5s", "K9s", "Q9s", "J9s", "87s"}
        late |= {"KTo", "QTo", "JTo", "96s", "75s", "64s"}
        blind |= {"A8o", "KJo", "T8s"}
    if players_at_table <= 4:
        early |= {"88", "77", "KQs", "AJo"}
        middle |= {"A4s", "A3s", "A2s", "KTo", "QTo", "JTo"}
        late |= {"A9o", "K9o", "Q9s", "T8s", "97s", "86s"}
        blind |= {"QJo", "JTo", "76s", "65s"}
    if players_at_table <= 3:
        early |= {"66", "55", "ATs", "A9s", "KJs", "QJs", "ATo"}
        middle |= {"A9o", "K9s", "Q9s", "T9s", "98s", "87s"}
        late |= {"A8o", "K9o", "Q9o", "J9s", "75s", "53s"}
        blind |= {"A7o", "K8s", "Q8s", "J9o"}

    return {
        "early": early,
        "middle": middle,
        "late": late,
        "blind": blind,
    }


def get_defend_ranges() -> dict:
    """Reasonable flatting ranges that avoid dominated offsuit trash OOP."""
    return {
        "early": {
            "QQ", "JJ", "TT", "99", "88",
            "AKs", "AQs", "AJs", "KQs", "QJs", "JTs",
            "AKo", "AQo",
        },
        "middle": {
            "QQ", "JJ", "TT", "99", "88", "77",
            "AKs", "AQs", "AJs", "ATs", "KQs", "KJs", "QJs", "JTs", "T9s",
            "AKo", "AQo", "AJo", "KQo",
        },
        "late": {
            "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
            "AKs", "AQs", "AJs", "ATs", "A9s", "KQs", "KJs", "KTs",
            "QJs", "QTs", "JTs", "T9s", "98s", "87s",
            "AKo", "AQo", "AJo", "KQo",
        },
        "blind": {
            "QQ", "JJ", "TT", "99", "88", "77", "66",
            "AKs", "AQs", "AJs", "ATs", "A9s", "KQs", "KJs", "KTs",
            "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s",
            "AKo", "AQo", "AJo", "KQo",
        },
    }


def get_three_bet_value_ranges() -> dict:
    """Strong value hands that are happy building the pot preflop."""
    return {
        "early": {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"},
        "middle": {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs"},
        "late": {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AJs"},
        "blind": {"AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AJs"},
    }


def get_three_bet_bluff_ranges() -> dict:
    """Late-position blocker and suited wheel hands for light 3-bets."""
    return {
        "early": set(),
        "middle": {"A5s", "A4s"},
        "late": {"A5s", "A4s", "A3s", "KTs", "QTs", "JTs"},
        "blind": {"A5s", "A4s", "KTs", "QTs"},
    }


def get_short_stack_jam_range(bucket: str) -> set:
    """Hands strong enough to jam when stacks are shallow."""
    base = {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88",
        "AKs", "AQs", "AJs", "ATs", "AKo", "AQo",
    }
    if bucket in {"late", "blind"}:
        return base | {"77", "66", "A9s", "A5s", "KQs", "AJo"}
    if bucket == "middle":
        return base | {"77", "KQs", "AJo"}
    return base


def get_four_bet_continue_range(bucket: str) -> set:
    """Very tight continue range once the pot is already heavily inflated."""
    base = {"AA", "KK", "QQ", "AKs", "AKo"}
    if bucket in {"late", "blind"}:
        return base | {"JJ"}
    return base


def get_open_raise_size(ctx: dict) -> int:
    """Open to 2.5bb, expressed as a total bet target."""
    big_blind = max(1, ctx["big_blind"])
    return max(ctx["min_raise_to"], int(round(OPEN_SIZE_BB * big_blind)))


def get_three_bet_size(ctx: dict) -> int:
    """3-bet to roughly 3x in position or 3.5x out of position."""
    multiplier = THREE_BET_IP_MULTIPLIER if ctx["position_info"]["is_late_position"] else THREE_BET_OOP_MULTIPLIER
    target = int(round(max(ctx["current_bet"], ctx["big_blind"]) * multiplier))
    return max(ctx["min_raise_to"], target)


def _is_unopened_preflop(ctx: dict) -> bool:
    """Treat the pot as unopened when the biggest bet is still only the blind."""
    return ctx["street"] == "preflop" and ctx["current_bet"] <= ctx["big_blind"]


def _is_facing_open(ctx: dict) -> bool:
    """Detect the first meaningful preflop raise without overfitting to logs."""
    return ctx["street"] == "preflop" and ctx["current_bet"] > ctx["big_blind"] and ctx["current_bet"] <= ctx["big_blind"] * 6


def _should_jam_short_stack(ctx: dict) -> bool:
    """Jam shallow stacks where raise-folding would burn too much equity."""
    return ctx["stack_bb"] <= 12 or (ctx["stack_bb"] <= 16 and ctx["amount_owed"] > 0)


def _can_flat_preflop(ctx: dict, hand: str) -> bool:
    """Prevent loose cold-calls and dominated offsuit continues out of position."""
    if ctx["position_info"]["is_blind"] and hand.endswith("o") and hand[0] in {"K", "Q", "J"}:
        return False
    if not ctx["position_info"]["is_late_position"] and hand in {"A9o", "A8o", "KJo", "KTo", "QJo", "QTo", "JTo"}:
        return False
    if ctx["stack_bucket"] == "micro":
        return False
    if ctx["amount_owed"] > max(ctx["big_blind"] * 4, int(ctx["pot"] * 0.25)):
        return False
    return True


def _is_tiny_call(state: dict) -> bool:
    """Allow only very cheap bluff-catch/call-downs in the fallback shell."""
    amount_owed = max(0, _as_int(state.get("amount_owed")))
    if amount_owed <= 0:
        return False
    pot = max(0, _as_int(state.get("pot")))
    return amount_owed <= TINY_CALL_CHIPS or amount_owed <= int(pot * TINY_CALL_POT_FRACTION)


def _default_postflop_value_size(ctx: dict) -> int:
    """Use a simple value size around two-thirds pot when betting postflop."""
    target = max(ctx["min_raise_to"], int(round(ctx["pot"] * 0.66)))
    if target <= 0:
        target = max(ctx["min_raise_to"], ctx["current_bet"] + ctx["big_blind"] * 2)
    return target


def _mix_frequency(state: dict, label: str, frequency: float) -> bool:
    """Use a stable hand-based roll for coarse action frequencies."""
    hand_id = str(state.get("hand_id", "0"))
    seed = "{}:{}".format(hand_id, label)
    bucket = sum(ord(char) for char in seed) % 100
    threshold = max(0, min(100, int(round(frequency * 100))))
    return bucket < threshold


def _card_ranks(cards) -> list:
    """Extract rank characters from valid card strings."""
    return [card[0] for card in cards if isinstance(card, str) and len(card) >= 2]


def _card_rank_values(cards) -> list:
    """Map visible cards to integer ranks for simple comparisons."""
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    values = []
    for rank in _card_ranks(cards):
        if rank in rank_map:
            values.append(rank_map[rank])
    return values


def _rank_count_map(ranks) -> dict:
    """Count rank frequency without assuming clean inputs."""
    counts = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    return counts


def _has_flush(cards) -> bool:
    """Check for any five-card flush among visible hole and board cards."""
    suits = [card[1] for card in cards if isinstance(card, str) and len(card) >= 2]
    return any(suits.count(suit) >= 5 for suit in "shdc")


def _has_straight(unique_ranks) -> bool:
    """Check for a five-rank straight, including wheel support."""
    value_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    values = {value_map[rank] for rank in unique_ranks if rank in value_map}
    if 14 in values:
        values.add(1)
    for start in range(1, 11):
        if all(rank in values for rank in range(start, start + 5)):
            return True
    return False


def _hero_pair_rank(hole, rank_counts) -> str:
    """Return the hero pocket-pair rank when it improved to trips or better."""
    if not isinstance(hole, list) or len(hole) != 2:
        return ""
    if any(not isinstance(card, str) or len(card) < 2 for card in hole):
        return ""
    first_rank = hole[0][0]
    second_rank = hole[1][0]
    if first_rank == second_rank and rank_counts.get(first_rank, 0) >= 3:
        return first_rank
    return ""


def _pair_category(hole, board, rank_counts) -> str:
    """Split one-pair hands into coarse strength groups."""
    board_values = _card_rank_values(board)
    hole_values = _card_rank_values(hole)
    if not hole_values:
        return "weak_pair"

    pair_ranks = [rank for rank, count in rank_counts.items() if count >= 2]
    if not pair_ranks:
        return "weak_pair"

    value_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    best_pair_value = max(value_map.get(rank, 0) for rank in pair_ranks)
    board_high = max(board_values, default=0)

    if len(hole) == 2 and hole[0][0] == hole[1][0]:
        if hole_values[0] > board_high:
            return "overpair"
        return "underpair"

    hero_pair_on_board = any(card[0] in pair_ranks for card in hole if isinstance(card, str) and len(card) >= 2)
    if hero_pair_on_board and board_high == best_pair_value:
        kicker = max(value for value in hole_values if value != best_pair_value) if any(
            value != best_pair_value for value in hole_values
        ) else min(hole_values)
        return "top_pair_good" if kicker >= 11 else "top_pair"
    if hero_pair_on_board:
        sorted_board = sorted(board_values, reverse=True)
        second_board = sorted_board[1] if len(sorted_board) > 1 else 0
        if best_pair_value >= second_board:
            return "middle_pair"
        return "weak_pair"

    if max(hole_values) > board_high:
        return "pocket_pair_over_board"
    return "weak_pair"


def _is_low_connected_board(board) -> bool:
    """Flag low runouts where straight interaction is naturally high."""
    values = sorted(set(_card_rank_values(board)))
    if len(values) < 3:
        return False
    return max(values) <= 10 and (max(values) - min(values) <= 4)


def _has_ace_high(state: dict) -> bool:
    """Treat ace-high as a little showdown value when unimproved."""
    hole = state.get("your_cards") if isinstance(state.get("your_cards"), list) else []
    board = state.get("community_cards") if isinstance(state.get("community_cards"), list) else []
    if has_pair_or_better(state):
        return False
    return "A" in _card_ranks(hole) and "A" not in _card_ranks(board)


def _max_total_bet(state: dict) -> int:
    """Maximum total bet we can legally reach with our remaining stack."""
    return max(
        0,
        _as_int(state.get("your_bet_this_street")) + _as_int(state.get("your_stack")),
    )


def _safe_ratio(numerator, denominator) -> float:
    """Divide defensively and return zero when the denominator is empty."""
    denominator_value = float(denominator) if denominator else 0.0
    if denominator_value <= 0:
        return 0.0
    return max(0.0, float(numerator) / denominator_value)


def _as_int(value) -> int:
    """Best-effort integer conversion for engine values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
