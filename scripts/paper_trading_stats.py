#!/usr/bin/env python3
"""Display paper trading statistics.

Recovers all signals from git history (since history.json gets overwritten),
fetches actual weather data for resolved markets, and calculates PnL.

Usage:
    python scripts/paper_trading_stats.py
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.weather_history import WeatherHistoryCollector


@dataclass
class Signal:
    """A trading signal."""

    market_id: str
    title: str
    city: str
    target_date: str
    market_price: float
    p_model: float
    edge: float
    recommended_size_usd: float
    scan_time: str
    threshold_celsius: float | None = None
    threshold_celsius_upper: float | None = None
    comparison: str = ">="

    @property
    def shares(self) -> float:
        """Number of shares purchased."""
        if self.market_price <= 0:
            return 0
        return self.recommended_size_usd / self.market_price


def extract_signals_from_git() -> list[Signal]:
    """Extract all unique tradeable signals from git history.

    Returns:
        List of Signal objects, deduplicated by (market_id, target_date).
    """
    # Get commits that touched signal files
    result = subprocess.run(
        ["git", "log", "--oneline", "--all", "--", "data/signals/*.json"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    if result.returncode != 0:
        print("Warning: Could not read git history")
        return []

    commits = [line.split()[0] for line in result.stdout.strip().split("\n") if line]

    all_signals: list[Signal] = []
    seen_ids: set[tuple[str, str]] = set()

    for commit in commits:
        try:
            show_result = subprocess.run(
                ["git", "show", f"{commit}:data/signals/history.json"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            if show_result.returncode != 0:
                continue

            data = json.loads(show_result.stdout)
            for scan in data.get("scans", []):
                scan_time = scan.get("scan_time", "")
                for sig in scan.get("signals", []):
                    if not sig.get("should_trade"):
                        continue

                    sig_id = (sig.get("market_id"), sig.get("target_date"))
                    if sig_id in seen_ids:
                        continue
                    seen_ids.add(sig_id)

                    # Parse threshold from title for range markets
                    import re

                    title = sig.get("title", "")
                    threshold = sig.get("threshold_celsius")
                    threshold_upper = None
                    comparison = ">="

                    # Detect market type from title and extract threshold if needed
                    if "or below" in title.lower():
                        comparison = "<="
                        # Extract threshold from title like "15°F or below"
                        if threshold is None:
                            match = re.search(r"(\d+)°F or below", title)
                            if match:
                                threshold = (int(match.group(1)) - 32) * 5 / 9
                    elif "or higher" in title.lower():
                        comparison = ">="
                        # Extract threshold from title like "26°F or higher"
                        if threshold is None:
                            match = re.search(r"(\d+)°F or higher", title)
                            if match:
                                threshold = (int(match.group(1)) - 32) * 5 / 9
                    elif "-" in title and "°F" in title:
                        # Range market like "18-19°F"
                        comparison = "range"
                        # Extract range from title
                        match = re.search(r"(\d+)-(\d+)°F", title)
                        if match:
                            low_f = int(match.group(1))
                            high_f = int(match.group(2)) + 1  # Upper bound exclusive
                            threshold = (low_f - 32) * 5 / 9
                            threshold_upper = (high_f - 32) * 5 / 9

                    signal = Signal(
                        market_id=sig.get("market_id", ""),
                        title=title,
                        city=sig.get("city", ""),
                        target_date=sig.get("target_date", ""),
                        market_price=sig.get("market_price", 0),
                        p_model=sig.get("p_model", 0),
                        edge=sig.get("edge", 0),
                        recommended_size_usd=sig.get("recommended_size_usd", 0),
                        scan_time=scan_time,
                        threshold_celsius=threshold,
                        threshold_celsius_upper=threshold_upper,
                        comparison=comparison,
                    )
                    all_signals.append(signal)
        except (json.JSONDecodeError, KeyError):
            continue

    return all_signals


def determine_outcome(actual_temp: float, signal: Signal) -> bool:
    """Determine if signal won (YES outcome).

    Args:
        actual_temp: Actual max temperature in Celsius.
        signal: The signal to evaluate.

    Returns:
        True if YES won, False if NO won.
    """
    if signal.comparison == "<=":
        return actual_temp <= (signal.threshold_celsius or 0)
    elif signal.comparison == "range":
        low = signal.threshold_celsius or 0
        high = signal.threshold_celsius_upper or 0
        return low <= actual_temp < high
    else:  # >= or default
        return actual_temp >= (signal.threshold_celsius or 0)


def calculate_pnl(signal: Signal, won: bool) -> float:
    """Calculate PnL for a signal.

    Args:
        signal: The signal.
        won: Whether YES won.

    Returns:
        PnL in dollars.
    """
    if won:
        # Each YES share pays $1, we paid market_price per share
        return signal.shares * (1 - signal.market_price)
    else:
        # Lost entire stake
        return -signal.recommended_size_usd


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def print_stats(signals: list[Signal], collector: WeatherHistoryCollector) -> None:
    """Print paper trading statistics."""
    today = date.today()

    print("=" * 70)
    print("PAPER TRADING STATS - Polymarket Weather Bot")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    resolved: list[tuple[Signal, float, bool, float]] = []  # signal, actual, won, pnl
    pending: list[Signal] = []

    for signal in signals:
        target = date.fromisoformat(signal.target_date)

        if target >= today:
            pending.append(signal)
            continue

        actual = collector.get_actual_weather(signal.city, target)
        if actual is None:
            pending.append(signal)
            continue

        actual_temp = actual.actual_max_temp
        won = determine_outcome(actual_temp, signal)
        pnl = calculate_pnl(signal, won)
        resolved.append((signal, actual_temp, won, pnl))

    # Print resolved trades
    if resolved:
        print("RESOLVED TRADES:")
        print("-" * 70)
        for signal, actual_temp, won, pnl in resolved:
            actual_f = celsius_to_fahrenheit(actual_temp)
            status = "WIN" if won else "LOSS"
            print(f"{signal.title}")
            print(f"  Target: {signal.target_date} | Price: ${signal.market_price:.4f} | "
                  f"Size: ${signal.recommended_size_usd:.2f} | Shares: {signal.shares:.1f}")
            print(f"  Actual: {actual_temp:.1f}°C ({actual_f:.1f}°F)")
            print(f"  Result: {status} | P&L: ${pnl:+.2f}")
            print()

    # Print pending trades
    if pending:
        print("PENDING TRADES:")
        print("-" * 70)
        for signal in pending:
            print(f"{signal.title}")
            print(f"  Target: {signal.target_date} | Price: ${signal.market_price:.4f} | "
                  f"Size: ${signal.recommended_size_usd:.2f}")
            print()

    # Summary stats
    wins = [r for r in resolved if r[2]]
    losses = [r for r in resolved if not r[2]]
    total_pnl = sum(r[3] for r in resolved)
    total_invested = sum(r[0].recommended_size_usd for r in resolved)
    pending_invested = sum(s.recommended_size_usd for s in pending)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Signals:    {len(signals)}")
    print(f"Resolved:         {len(resolved)}")
    print(f"Pending:          {len(pending)}")
    print()
    print(f"Wins:             {len(wins)}")
    print(f"Losses:           {len(losses)}")
    if resolved:
        print(f"Win Rate:         {len(wins) / len(resolved) * 100:.1f}%")
    print()
    print(f"Total Invested:   ${total_invested:.2f}")
    print(f"Pending Capital:  ${pending_invested:.2f}")
    print(f"Realized P&L:     ${total_pnl:+.2f}")
    if total_invested > 0:
        print(f"Return:           {total_pnl / total_invested * 100:+.1f}%")

    # P&L by city
    pnl_by_city: dict[str, float] = {}
    trades_by_city: dict[str, int] = {}
    for signal, _, _, pnl in resolved:
        pnl_by_city[signal.city] = pnl_by_city.get(signal.city, 0) + pnl
        trades_by_city[signal.city] = trades_by_city.get(signal.city, 0) + 1

    if pnl_by_city:
        print()
        print("P&L BY CITY:")
        print("-" * 40)
        for city in sorted(pnl_by_city.keys()):
            print(f"  {city:20s} ${pnl_by_city[city]:+8.2f}  ({trades_by_city[city]} trades)")

    print("=" * 70)


def main() -> int:
    """Main entry point."""
    print("Extracting signals from git history...")
    signals = extract_signals_from_git()

    if not signals:
        print("No tradeable signals found in git history.")
        print("Make sure you're in a git repository with signal data.")
        return 1

    print(f"Found {len(signals)} unique tradeable signals.")
    print()

    collector = WeatherHistoryCollector()
    print_stats(signals, collector)

    return 0


if __name__ == "__main__":
    sys.exit(main())
