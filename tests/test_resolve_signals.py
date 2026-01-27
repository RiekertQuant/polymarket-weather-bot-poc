"""Tests for scripts/resolve_signals.py outcome resolution logic."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.resolve_signals import (
    determine_outcome,
    calculate_pnl,
    resolve_signals,
    _signal_key,
    load_history,
    load_existing_outcomes,
    HISTORY_PATH,
    OUTCOMES_PATH,
)
from src.backtest.weather_history import ActualWeather


def _make_signal(
    market_id="m1",
    city="New York",
    threshold=30.0,
    threshold_upper=None,
    comparison=">=",
    target_date=None,
    market_price=0.60,
    size=10.0,
):
    """Helper to build a signal dict."""
    if target_date is None:
        target_date = (date.today() - timedelta(days=2)).isoformat()
    return {
        "market_id": market_id,
        "title": f"Test market {market_id}",
        "city": city,
        "threshold_celsius": threshold,
        "threshold_celsius_upper": threshold_upper,
        "comparison": comparison,
        "target_date": target_date,
        "market_price": market_price,
        "recommended_size_usd": size,
    }


# ── determine_outcome tests ──────────────────────────────────────────


class TestDetermineOutcome:
    def test_resolve_ge_win(self):
        """actual >= threshold → YES"""
        signal = _make_signal(threshold=30.0, comparison=">=")
        assert determine_outcome(32.0, signal) == "YES"

    def test_resolve_ge_loss(self):
        """actual < threshold → NO"""
        signal = _make_signal(threshold=30.0, comparison=">=")
        assert determine_outcome(28.0, signal) == "NO"

    def test_resolve_ge_exact(self):
        """actual == threshold → YES for >="""
        signal = _make_signal(threshold=30.0, comparison=">=")
        assert determine_outcome(30.0, signal) == "YES"

    def test_resolve_le_win(self):
        """actual <= threshold → YES"""
        signal = _make_signal(threshold=30.0, comparison="<=")
        assert determine_outcome(28.0, signal) == "YES"

    def test_resolve_le_exact(self):
        """actual == threshold → YES for <="""
        signal = _make_signal(threshold=30.0, comparison="<=")
        assert determine_outcome(30.0, signal) == "YES"

    def test_resolve_le_loss(self):
        """actual > threshold → NO for <="""
        signal = _make_signal(threshold=30.0, comparison="<=")
        assert determine_outcome(32.0, signal) == "NO"

    def test_resolve_range_win(self):
        """actual in [lower, upper) → YES"""
        signal = _make_signal(threshold=25.0, threshold_upper=30.0, comparison="range")
        assert determine_outcome(27.0, signal) == "YES"

    def test_resolve_range_lower_bound(self):
        """actual == lower bound → YES (inclusive)"""
        signal = _make_signal(threshold=25.0, threshold_upper=30.0, comparison="range")
        assert determine_outcome(25.0, signal) == "YES"

    def test_resolve_range_upper_bound(self):
        """actual == upper bound → NO (exclusive)"""
        signal = _make_signal(threshold=25.0, threshold_upper=30.0, comparison="range")
        assert determine_outcome(30.0, signal) == "NO"

    def test_resolve_range_loss(self):
        """actual outside range → NO"""
        signal = _make_signal(threshold=25.0, threshold_upper=30.0, comparison="range")
        assert determine_outcome(35.0, signal) == "NO"


# ── calculate_pnl tests ──────────────────────────────────────────────


class TestCalculatePnl:
    def test_win_pnl(self):
        """WIN: pnl = size / price * (1 - price)"""
        signal = _make_signal(market_price=0.60, size=10.0)
        result = calculate_pnl(signal, "YES")
        # 10 / 0.60 * 0.40 = 6.6667
        assert result["pnl"] == pytest.approx(6.6667, abs=0.01)
        assert result["return_pct"] == pytest.approx(0.6667, abs=0.01)

    def test_loss_pnl(self):
        """LOSS: pnl = -size"""
        signal = _make_signal(market_price=0.60, size=10.0)
        result = calculate_pnl(signal, "NO")
        assert result["pnl"] == -10.0
        assert result["return_pct"] == -1.0

    def test_zero_price(self):
        """Edge case: zero price returns zero PnL."""
        signal = _make_signal(market_price=0.0, size=10.0)
        result = calculate_pnl(signal, "YES")
        assert result["pnl"] == 0.0

    def test_zero_size(self):
        """Edge case: zero size returns zero PnL."""
        signal = _make_signal(market_price=0.60, size=0.0)
        result = calculate_pnl(signal, "YES")
        assert result["pnl"] == 0.0


# ── resolve_signals integration tests ────────────────────────────────


class TestResolveSignals:
    def _mock_history(self, signals):
        """Create a mock history dict with given signals."""
        return {
            "scans": [
                {
                    "date": "2026-01-20",
                    "scan_time": "2026-01-20T12:00:00Z",
                    "data_source": "mock",
                    "markets_scanned": len(signals),
                    "signals_generated": len(signals),
                    "signals": signals,
                }
            ],
            "total_signals": len(signals),
            "signals_by_city": {},
        }

    @patch("scripts.resolve_signals.OUTCOMES_PATH")
    @patch("scripts.resolve_signals.HISTORY_PATH")
    @patch("scripts.resolve_signals.WeatherHistoryCollector")
    def test_skips_future_signals(self, MockCollector, mock_hist_path, mock_out_path):
        """Signals with target_date >= today should not be resolved."""
        future_date = (date.today() + timedelta(days=5)).isoformat()
        signal = _make_signal(target_date=future_date)

        mock_hist_path.exists.return_value = True
        mock_hist_path.__str__ = lambda self: "history.json"
        mock_hist_path.open = MagicMock()

        mock_out_path.exists.return_value = False

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.read = MagicMock(
                return_value=json.dumps(self._mock_history([signal]))
            )

            with patch("scripts.resolve_signals.load_history", return_value=[signal]):
                with patch("scripts.resolve_signals.load_existing_outcomes", return_value={}):
                    result = resolve_signals()

        assert len(result["resolved_signals"]) == 0
        assert len(result["pending_signals"]) == 1

    @patch("scripts.resolve_signals.WeatherHistoryCollector")
    def test_skips_already_resolved(self, MockCollector):
        """Previously resolved signals should not be re-fetched."""
        past_date = (date.today() - timedelta(days=3)).isoformat()
        signal = _make_signal(target_date=past_date)

        already = {
            _signal_key(signal): {
                **signal,
                "actual_temp": 32.0,
                "outcome": "YES",
                "pnl": 6.67,
                "return_pct": 0.67,
            }
        }

        with patch("scripts.resolve_signals.load_history", return_value=[signal]):
            with patch("scripts.resolve_signals.load_existing_outcomes", return_value=already):
                result = resolve_signals()

        # The collector should never be called
        MockCollector.return_value.get_actual_weather.assert_not_called()
        assert len(result["resolved_signals"]) == 1
        assert len(result["newly_resolved"]) == 0

    @patch("scripts.resolve_signals.WeatherHistoryCollector")
    def test_resolves_past_signal(self, MockCollector):
        """A past signal with actual weather data should be resolved."""
        past_date = (date.today() - timedelta(days=3)).isoformat()
        signal = _make_signal(
            threshold=30.0, comparison=">=", target_date=past_date,
            market_price=0.60, size=10.0,
        )

        mock_actual = ActualWeather(
            date=date.fromisoformat(past_date),
            city="New York",
            actual_max_temp=32.0,
            actual_min_temp=20.0,
        )
        MockCollector.return_value.get_actual_weather.return_value = mock_actual

        with patch("scripts.resolve_signals.load_history", return_value=[signal]):
            with patch("scripts.resolve_signals.load_existing_outcomes", return_value={}):
                result = resolve_signals()

        assert len(result["resolved_signals"]) == 1
        assert len(result["newly_resolved"]) == 1
        resolved = result["resolved_signals"][0]
        assert resolved["outcome"] == "YES"
        assert resolved["actual_temp"] == 32.0
        assert resolved["pnl"] > 0
