#!/usr/bin/env python3
"""Display paper trading statistics.

Recovers all signals from git history (since history.json gets overwritten),
fetches actual weather data from NWS station observations (matching
Polymarket's resolution source), and calculates PnL.

Usage:
    python scripts/paper_trading_stats.py
"""

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# NWS station IDs used by Polymarket for each city
CITY_NWS_STATIONS: dict[str, str] = {
    "New York City": "KLGA",
}

# Timezone offsets (standard time) for local-day filtering
CITY_UTC_OFFSETS: dict[str, int] = {
    "New York City": -5,  # EST
}


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


def get_nws_daily_max(station_id: str, target_date: date, utc_offset: int) -> Optional[float]:
    """Fetch the daily max temperature from NWS station observations.

    Queries the NWS API for all observations during the local calendar day,
    and returns the highest temperature in Fahrenheit (rounded to nearest
    integer, matching Polymarket's resolution).

    Args:
        station_id: NWS station identifier (e.g. "KLGA").
        target_date: The local calendar date.
        utc_offset: UTC offset in hours for the station's timezone (e.g. -5 for EST).

    Returns:
        Max temperature in Fahrenheit as integer, or None if unavailable.
    """
    # Convert local day boundaries to UTC
    offset = timedelta(hours=utc_offset)
    start_utc = datetime(target_date.year, target_date.month, target_date.day,
                         tzinfo=timezone.utc) - offset
    end_utc = start_utc + timedelta(days=1)

    url = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    headers = {
        "User-Agent": "polymarket-weather-bot",
        "Accept": "application/geo+json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"NWS API request failed for {station_id}: {e}")
        return None

    max_temp_f: Optional[float] = None
    for obs in data.get("features", []):
        props = obs.get("properties", {})
        temp = props.get("temperature", {})
        val = temp.get("value")
        if val is not None:
            temp_f = val * 9 / 5 + 32
            if max_temp_f is None or temp_f > max_temp_f:
                max_temp_f = temp_f

    if max_temp_f is not None:
        return round(max_temp_f)
    return None


def extract_signals_from_git() -> list[Signal]:
    """Extract all unique tradeable signals from git history.

    Returns:
        List of Signal objects, deduplicated by (market_id, target_date).
    """
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

                    title = sig.get("title", "")
                    threshold = sig.get("threshold_celsius")
                    threshold_upper = None
                    comparison = ">="

                    if "or below" in title.lower():
                        comparison = "<="
                        if threshold is None:
                            match = re.search(r"(\d+)°F or below", title)
                            if match:
                                threshold = (int(match.group(1)) - 32) * 5 / 9
                    elif "or higher" in title.lower():
                        comparison = ">="
                        if threshold is None:
                            match = re.search(r"(\d+)°F or higher", title)
                            if match:
                                threshold = (int(match.group(1)) - 32) * 5 / 9
                    elif "-" in title and "°F" in title:
                        comparison = "range"
                        match = re.search(r"(\d+)-(\d+)°F", title)
                        if match:
                            low_f = int(match.group(1))
                            high_f = int(match.group(2)) + 1
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


def determine_outcome_f(actual_temp_f: int, signal: Signal) -> bool:
    """Determine if signal won based on actual temp in Fahrenheit.

    Uses integer Fahrenheit to match Polymarket's resolution.

    Args:
        actual_temp_f: Actual max temperature in whole-degree Fahrenheit.
        signal: The signal to evaluate.

    Returns:
        True if YES won, False if NO won.
    """
    title = signal.title

    if signal.comparison == "<=":
        # "15°F or below" means actual <= 15
        match = re.search(r"(\d+)°F or below", title)
        if match:
            return actual_temp_f <= int(match.group(1))
        return False
    elif signal.comparison == ">=":
        # "26°F or higher" means actual >= 26
        match = re.search(r"(\d+)°F or higher", title)
        if match:
            return actual_temp_f >= int(match.group(1))
        return False
    elif signal.comparison == "range":
        # "18-19°F" means actual is 18 or 19
        match = re.search(r"(\d+)-(\d+)°F", title)
        if match:
            low_f = int(match.group(1))
            high_f = int(match.group(2))
            return low_f <= actual_temp_f <= high_f
        return False
    return False


def calculate_pnl(signal: Signal, won: bool) -> float:
    """Calculate PnL for a signal.

    Args:
        signal: The signal.
        won: Whether YES won.

    Returns:
        PnL in dollars.
    """
    if won:
        return signal.shares * (1 - signal.market_price)
    else:
        return -signal.recommended_size_usd


def print_stats(signals: list[Signal]) -> None:
    """Print paper trading statistics."""
    today = date.today()

    print("=" * 70)
    print("PAPER TRADING STATS - Polymarket Weather Bot")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Weather source: NWS station observations (KLGA for NYC)")
    print()

    resolved: list[tuple[Signal, int, bool, float]] = []
    pending: list[Signal] = []
    temp_cache: dict[tuple[str, str], Optional[int]] = {}

    for signal in signals:
        target = date.fromisoformat(signal.target_date)

        if target >= today:
            pending.append(signal)
            continue

        cache_key = (signal.city, signal.target_date)
        if cache_key not in temp_cache:
            station = CITY_NWS_STATIONS.get(signal.city)
            utc_offset = CITY_UTC_OFFSETS.get(signal.city)
            if station and utc_offset is not None:
                temp_cache[cache_key] = get_nws_daily_max(station, target, utc_offset)
            else:
                logger.warning(f"No NWS station configured for {signal.city}")
                temp_cache[cache_key] = None

        actual_f = temp_cache[cache_key]
        if actual_f is None:
            pending.append(signal)
            continue

        won = determine_outcome_f(actual_f, signal)
        pnl = calculate_pnl(signal, won)
        resolved.append((signal, actual_f, won, pnl))

    # Print resolved trades
    if resolved:
        print("RESOLVED TRADES:")
        print("-" * 70)
        for signal, actual_f, won, pnl in resolved:
            status = "WIN" if won else "LOSS"
            print(f"{signal.title}")
            print(f"  Target: {signal.target_date} | Price: ${signal.market_price:.4f} | "
                  f"Size: ${signal.recommended_size_usd:.2f} | Shares: {signal.shares:.1f}")
            print(f"  Actual: {actual_f}°F (KLGA)")
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

    print_stats(signals)

    return 0


if __name__ == "__main__":
    sys.exit(main())
