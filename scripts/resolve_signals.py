#!/usr/bin/env python3
"""Resolve past signals against actual weather data and compute hypothetical PnL.

Reads signals from data/signals/history.json, fetches actual temperatures
for any whose target_date has passed, determines YES/NO outcomes, and
calculates PnL. Results are saved to data/signals/outcomes.json.
"""

import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.weather_history import WeatherHistoryCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SIGNALS_DIR = Path("data/signals")
HISTORY_PATH = SIGNALS_DIR / "history.json"
OUTCOMES_PATH = SIGNALS_DIR / "outcomes.json"


def load_history() -> list[dict]:
    """Load all signals from history.json.

    Returns:
        Flat list of signal dicts across all scans.
    """
    if not HISTORY_PATH.exists():
        logger.warning("No history.json found")
        return []

    with open(HISTORY_PATH) as f:
        history = json.load(f)

    signals = []
    for scan in history.get("scans", []):
        for sig in scan.get("signals", []):
            signals.append(sig)
    return signals


def load_existing_outcomes() -> dict[str, dict]:
    """Load already-resolved outcomes keyed by signal id.

    Returns:
        Dict mapping signal key to resolved signal dict.
    """
    if not OUTCOMES_PATH.exists():
        return {}

    with open(OUTCOMES_PATH) as f:
        data = json.load(f)

    resolved = {}
    for sig in data.get("resolved_signals", []):
        key = _signal_key(sig)
        resolved[key] = sig
    return resolved


def _signal_key(sig: dict) -> str:
    """Create a unique key for a signal to detect duplicates."""
    return f"{sig.get('market_id')}|{sig.get('target_date')}"


def determine_outcome(actual_max: float, signal: dict) -> str:
    """Determine YES/NO outcome for a signal given actual temperature.

    Args:
        actual_max: Actual observed max temperature in Celsius.
        signal: Signal dict with comparison and threshold fields.

    Returns:
        "YES" or "NO".
    """
    comparison = signal.get("comparison", ">=")
    threshold = signal.get("threshold_celsius")
    threshold_upper = signal.get("threshold_celsius_upper")

    if comparison == ">=" and threshold is not None:
        return "YES" if actual_max >= threshold else "NO"
    elif comparison == "<=" and threshold is not None:
        return "YES" if actual_max <= threshold else "NO"
    elif comparison == "range" and threshold is not None and threshold_upper is not None:
        return "YES" if threshold <= actual_max < threshold_upper else "NO"
    else:
        # Default: treat as >= if we have a threshold
        if threshold is not None:
            return "YES" if actual_max >= threshold else "NO"
        return "NO"


def calculate_pnl(signal: dict, outcome: str) -> dict:
    """Calculate PnL for a resolved signal.

    BUY YES at market_price. If outcome is YES, profit = size / price * (1 - price).
    If outcome is NO, loss = -size.

    Args:
        signal: Signal dict with market_price and recommended_size_usd.
        outcome: "YES" or "NO".

    Returns:
        Dict with pnl and return_pct.
    """
    price = signal.get("market_price", 0)
    size = signal.get("recommended_size_usd", 0)

    if price <= 0 or size <= 0:
        return {"pnl": 0.0, "return_pct": 0.0}

    if outcome == "YES":
        # Bought YES shares at `price`, each pays $1 on win
        # Number of shares = size / price
        # Profit = shares * (1 - price) = size / price * (1 - price)
        pnl = size / price * (1 - price)
        return_pct = (1 - price) / price
    else:
        # Lost entire stake
        pnl = -size
        return_pct = -1.0

    return {"pnl": round(pnl, 4), "return_pct": round(return_pct, 4)}


def resolve_signals() -> dict:
    """Resolve all past signals and compute outcomes.

    Returns:
        Dict with resolved_signals, pending_signals, summary, and newly_resolved.
    """
    signals = load_history()
    already_resolved = load_existing_outcomes()
    collector = WeatherHistoryCollector()

    today = date.today()
    resolved_signals = []
    pending_signals = []
    newly_resolved = []

    # Keep previously resolved signals
    for sig in already_resolved.values():
        resolved_signals.append(sig)

    seen_keys = set(already_resolved.keys())

    for signal in signals:
        key = _signal_key(signal)
        if key in seen_keys:
            continue

        target_date_str = signal.get("target_date")
        if not target_date_str:
            continue

        target_date = date.fromisoformat(target_date_str)

        if target_date >= today:
            pending_signals.append(signal)
            continue

        city = signal.get("city")
        if not city:
            continue

        actual = collector.get_actual_weather(city, target_date)
        if actual is None:
            logger.warning(f"No actual weather for {city} on {target_date}")
            pending_signals.append(signal)
            continue

        outcome = determine_outcome(actual.actual_max_temp, signal)
        pnl_info = calculate_pnl(signal, outcome)

        resolved = {
            **signal,
            "actual_temp": actual.actual_max_temp,
            "outcome": outcome,
            "pnl": pnl_info["pnl"],
            "return_pct": pnl_info["return_pct"],
        }
        resolved_signals.append(resolved)
        newly_resolved.append(resolved)
        seen_keys.add(key)

        logger.info(
            f"Resolved: {signal.get('title', 'Unknown')} → {outcome} "
            f"(actual={actual.actual_max_temp}°C, pnl=${pnl_info['pnl']:+.2f})"
        )

    # Build summary
    wins = [s for s in resolved_signals if s.get("outcome") == "YES"]
    losses = [s for s in resolved_signals if s.get("outcome") == "NO"]
    total_resolved = len(resolved_signals)
    total_pnl = sum(s.get("pnl", 0) for s in resolved_signals)
    total_risked = sum(s.get("recommended_size_usd", 0) for s in resolved_signals)

    summary = {
        "total_resolved": total_resolved,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / total_resolved if total_resolved > 0 else 0,
        "total_pnl": round(total_pnl, 4),
        "total_risked": round(total_risked, 4),
        "pending": len(pending_signals),
    }

    return {
        "resolved_signals": resolved_signals,
        "pending_signals": pending_signals,
        "summary": summary,
        "newly_resolved": newly_resolved,
    }


def save_outcomes(result: dict) -> None:
    """Save outcomes to data/signals/outcomes.json."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "resolved_signals": result["resolved_signals"],
        "pending_signals": result["pending_signals"],
        "summary": result["summary"],
        "last_updated": date.today().isoformat(),
    }

    with open(OUTCOMES_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved outcomes to {OUTCOMES_PATH}")


def print_summary(result: dict) -> None:
    """Print a console summary of resolved signals."""
    summary = result["summary"]

    print("\n" + "=" * 60)
    print("SIGNAL OUTCOME RESOLUTION")
    print("=" * 60)
    print(f"Total Resolved:  {summary['total_resolved']}")
    print(f"Wins:            {summary['wins']}")
    print(f"Losses:          {summary['losses']}")
    print(f"Win Rate:        {summary['win_rate']:.1%}")
    print(f"Total PnL:       ${summary['total_pnl']:+.2f}")
    print(f"Total Risked:    ${summary['total_risked']:.2f}")
    print(f"Pending:         {summary['pending']}")

    newly = result.get("newly_resolved", [])
    if newly:
        print("\n" + "-" * 60)
        print(f"NEWLY RESOLVED ({len(newly)})")
        print("-" * 60)
        for sig in newly:
            print(f"  {sig.get('title', 'Unknown')}")
            print(f"    Outcome: {sig['outcome']} | Actual: {sig['actual_temp']}°C | PnL: ${sig['pnl']:+.2f}")
    else:
        print("\nNo new resolutions this run.")

    print("=" * 60)


def send_discord_summary(result: dict) -> None:
    """Send a Discord notification with newly resolved signals."""
    import requests

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        return

    newly = result.get("newly_resolved", [])
    if not newly:
        return

    summary = result["summary"]

    lines = []
    for sig in newly:
        outcome_emoji = "W" if sig["outcome"] == "YES" else "L"
        lines.append(
            f"**[{outcome_emoji}] {sig.get('title', 'Unknown')}**\n"
            f"Actual: {sig['actual_temp']}°C | PnL: ${sig['pnl']:+.2f}"
        )

    description = "\n\n".join(lines)

    embed = {
        "title": f"Signal Outcomes: {len(newly)} Resolved",
        "description": description,
        "color": 0x00CC66 if summary["total_pnl"] >= 0 else 0xCC3300,
        "fields": [
            {"name": "Record", "value": f"{summary['wins']}W-{summary['losses']}L", "inline": True},
            {"name": "Win Rate", "value": f"{summary['win_rate']:.1%}", "inline": True},
            {"name": "Total PnL", "value": f"${summary['total_pnl']:+.2f}", "inline": True},
        ],
        "footer": {"text": f"Pending: {summary['pending']} signals"},
    }

    try:
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        resp.raise_for_status()
        logger.info("Discord outcome summary sent")
    except Exception:
        logger.warning("Failed to send Discord outcome summary", exc_info=True)


def main():
    logger.info("Resolving past signals...")

    result = resolve_signals()
    save_outcomes(result)
    print_summary(result)
    send_discord_summary(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
