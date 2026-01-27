"""Tests for src/backtest/report.py — BacktestReport formatting (no network)."""

import json
from datetime import date

import pytest

from src.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    DailySnapshot,
)
from src.backtest.report import BacktestReport


def _trade(
    city: str = "New York City",
    pnl: float = 5.0,
    resolved: bool = True,
) -> BacktestTrade:
    return BacktestTrade(
        market_id="m1",
        market_title=f"Will {city} hit 30°C?",
        city=city,
        threshold_celsius=30.0,
        target_date=date(2026, 2, 1),
        entry_date=date(2026, 1, 31),
        entry_price=0.05,
        shares=100.0,
        cost_basis=5.0,
        p_model=0.90,
        edge=0.85,
        forecast_temp=33.0,
        outcome=True if pnl > 0 else False,
        actual_temp=31.0 if pnl > 0 else 28.0,
        pnl=pnl,
        return_pct=(pnl / 5.0) * 100,
        resolved=resolved,
        resolution_date=date(2026, 2, 1) if resolved else None,
    )


def _snapshot(day_offset: int = 0, cum_pnl: float = 0.0) -> DailySnapshot:
    return DailySnapshot(
        date=date(2026, 1, 31) + __import__("datetime").timedelta(days=day_offset),
        balance=1000.0 + cum_pnl,
        open_positions=0,
        daily_pnl=cum_pnl if day_offset == 0 else 0.0,
        cumulative_pnl=cum_pnl,
        trades_made=1 if day_offset == 0 else 0,
        win_rate=1.0,
    )


def _result(
    trades: list[BacktestTrade] | None = None,
    snapshots: list[DailySnapshot] | None = None,
) -> BacktestResult:
    if trades is None:
        trades = [_trade(pnl=5.0), _trade(city="London", pnl=-3.0)]
    if snapshots is None:
        snapshots = [_snapshot(0, 0.0), _snapshot(1, 5.0), _snapshot(2, 2.0)]

    return BacktestResult(
        config=BacktestConfig(),
        start_date=date(2026, 1, 31),
        end_date=date(2026, 2, 2),
        trading_days=3,
        total_trades=2,
        winning_trades=1,
        losing_trades=1,
        win_rate=0.5,
        total_pnl=2.0,
        total_return_pct=0.2,
        avg_trade_pnl=1.0,
        max_win=5.0,
        max_loss=-3.0,
        max_drawdown=3.0,
        max_drawdown_pct=0.3,
        sharpe_ratio=0.5,
        trades=trades,
        daily_snapshots=snapshots,
        pnl_by_city={"New York City": 5.0, "London": -3.0},
        trades_by_city={"New York City": 1, "London": 1},
    )


# ── TestBacktestReport ───────────────────────────────────────────────


class TestBacktestReport:
    def test_summary_contains_key_strings(self):
        report = BacktestReport(_result())
        text = report.summary()

        assert "BACKTEST REPORT" in text
        assert "$2.00" in text or "2.00" in text  # total P&L
        assert "50.0%" in text  # win rate

    def test_trade_log_contains_headers_and_entries(self):
        report = BacktestReport(_result())
        text = report.trade_log()

        assert "TRADE LOG" in text
        assert "Date" in text
        assert "City" in text
        assert "New York City" in text

    def test_equity_curve_empty_snapshots(self):
        res = _result(snapshots=[])
        report = BacktestReport(res)
        text = report.equity_curve_ascii()

        assert "No daily data" in text

    def test_save_json_writes_valid_json(self, tmp_path):
        report = BacktestReport(_result())
        json_path = tmp_path / "test_output.json"
        report.save_json(json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert "summary" in data
        assert "trades" in data
        assert data["summary"]["total_trades"] == 2
