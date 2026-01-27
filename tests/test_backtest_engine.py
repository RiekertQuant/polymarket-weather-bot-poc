"""Tests for src/backtest/engine.py — BacktestEngine internals (no network)."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.backtest.data_collector import HistoricalMarket, PricePoint
from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestTrade,
)
from src.backtest.weather_history import (
    WeatherHistoryCollector,
    HistoricalForecast,
    ActualWeather,
)


def _config(**overrides) -> BacktestConfig:
    defaults = dict(
        initial_balance=1000.0,
        min_bet_usd=2.0,
        max_bet_usd=5.0,
        max_daily_risk_usd=50.0,
        max_risk_per_city_usd=20.0,
        max_trades_per_day=10,
        forecast_sigma=2.0,
        slippage=0.0,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _market(
    target_date: date = date(2026, 2, 1),
    price_history: list[PricePoint] | None = None,
    city: str = "New York City",
    threshold: float = 30.0,
) -> HistoricalMarket:
    if price_history is None:
        price_history = [
            PricePoint(datetime(2026, 1, 30, 12, 0), 0.05),
            PricePoint(datetime(2026, 1, 31, 12, 0), 0.06),
        ]
    return HistoricalMarket(
        condition_id="cond-1",
        token_id="tok-1",
        question=f"Will {city} hit {threshold}°C?",
        description="",
        end_date=target_date,
        outcome=True,
        city=city,
        threshold_celsius=threshold,
        target_date=target_date,
        comparison=">=",
        price_history=price_history,
        volume=1000.0,
    )


def _mock_weather(
    forecast_temp: float = 33.0,
    actual_temp: float = 31.0,
) -> WeatherHistoryCollector:
    collector = MagicMock(spec=WeatherHistoryCollector)
    collector.get_forecast_at_decision_time.return_value = HistoricalForecast(
        forecast_date=date(2026, 1, 31),
        target_date=date(2026, 2, 1),
        city="New York City",
        forecast_max_temp=forecast_temp,
        forecast_min_temp=20.0,
        lead_days=1,
    )
    collector.get_actual_weather.return_value = ActualWeather(
        date=date(2026, 2, 1),
        city="New York City",
        actual_max_temp=actual_temp,
        actual_min_temp=18.0,
    )
    return collector


# ── TestGetPriceAtTime ───────────────────────────────────────────────


class TestGetPriceAtTime:
    def test_returns_latest_before_target(self):
        engine = BacktestEngine(_config())
        mkt = _market()
        target = datetime(2026, 1, 31, 23, 59, 59)

        price = engine._get_price_at_time(mkt, target)
        assert price == pytest.approx(0.06)

    def test_returns_none_when_no_history(self):
        engine = BacktestEngine(_config())
        mkt = _market(price_history=[])

        price = engine._get_price_at_time(mkt, datetime(2026, 1, 31))
        assert price is None

    def test_handles_exact_timestamp(self):
        engine = BacktestEngine(_config())
        mkt = _market()
        exact = datetime(2026, 1, 30, 12, 0)

        price = engine._get_price_at_time(mkt, exact)
        assert price == pytest.approx(0.05)


# ── TestCalculateProbability ─────────────────────────────────────────


class TestCalculateProbability:
    def test_ge_threshold_below_forecast(self):
        engine = BacktestEngine(_config(forecast_sigma=2.0))
        # threshold=28 well below forecast=33 → high prob
        p = engine._calculate_probability(33.0, 28.0, ">=")
        assert p > 0.90

    def test_ge_threshold_above_forecast(self):
        engine = BacktestEngine(_config(forecast_sigma=2.0))
        # threshold=38 well above forecast=33 → low prob
        p = engine._calculate_probability(33.0, 38.0, ">=")
        assert p < 0.01

    def test_lt_comparison_reversed(self):
        engine = BacktestEngine(_config(forecast_sigma=2.0))
        # threshold=38 well above forecast=33 → high prob for "<"
        p = engine._calculate_probability(33.0, 38.0, "<")
        assert p > 0.99


# ── TestResolveTrade ─────────────────────────────────────────────────


class TestResolveTrade:
    def _trade(self, threshold: float = 30.0) -> BacktestTrade:
        return BacktestTrade(
            market_id="m1",
            market_title="test",
            city="NYC",
            threshold_celsius=threshold,
            target_date=date(2026, 2, 1),
            entry_date=date(2026, 1, 31),
            entry_price=0.05,
            shares=100.0,
            cost_basis=5.0,
            p_model=0.90,
            edge=0.85,
            forecast_temp=33.0,
        )

    def test_yes_outcome_positive_pnl(self):
        engine = BacktestEngine(_config())
        trade = self._trade(threshold=30.0)
        engine._resolve_trade(trade, actual_temp=31.0)

        assert trade.outcome is True
        assert trade.pnl > 0
        assert trade.pnl == pytest.approx(100.0 * (1.0 - 0.05))

    def test_no_outcome_negative_pnl(self):
        engine = BacktestEngine(_config())
        trade = self._trade(threshold=30.0)
        engine._resolve_trade(trade, actual_temp=29.0)

        assert trade.outcome is False
        assert trade.pnl == pytest.approx(-5.0)

    def test_return_pct(self):
        engine = BacktestEngine(_config())
        trade = self._trade(threshold=30.0)
        engine._resolve_trade(trade, actual_temp=31.0)

        expected_pct = (trade.pnl / trade.cost_basis) * 100
        assert trade.return_pct == pytest.approx(expected_pct)


# ── TestCalculateBetSize ─────────────────────────────────────────────


class TestCalculateBetSize:
    def test_returns_valid_amount(self):
        engine = BacktestEngine(_config())
        amount = engine._calculate_bet_size(edge=0.50, price=0.05, current_city_risk=0, current_daily_risk=0, trades_today=0)

        assert amount is not None
        assert amount >= 2.0
        assert amount <= 5.0

    def test_returns_none_when_trades_exceeded(self):
        engine = BacktestEngine(_config(max_trades_per_day=2))
        amount = engine._calculate_bet_size(edge=0.50, price=0.05, current_city_risk=0, current_daily_risk=0, trades_today=2)

        assert amount is None

    def test_returns_none_when_daily_risk_exhausted(self):
        engine = BacktestEngine(_config(max_daily_risk_usd=10.0))
        amount = engine._calculate_bet_size(edge=0.50, price=0.05, current_city_risk=0, current_daily_risk=10.0, trades_today=0)

        assert amount is None


# ── TestRunBacktest ──────────────────────────────────────────────────


class TestRunBacktest:
    def test_produces_result_with_synthetic_market(self):
        weather = _mock_weather(forecast_temp=33.0, actual_temp=31.0)
        cfg = _config(
            slippage=0.0,
            price_min=0.001,
            price_max=0.10,
            min_edge_absolute=0.15,
            min_edge_relative=0.10,
        )
        engine = BacktestEngine(cfg, weather)
        mkt = _market(target_date=date(2026, 2, 1))

        result = engine.run(
            [mkt],
            start_date=date(2026, 1, 31),
            end_date=date(2026, 2, 1),
        )

        assert result is not None
        assert result.trading_days == 2

    def test_zero_markets_gives_zero_trades(self):
        weather = _mock_weather()
        engine = BacktestEngine(_config(), weather)

        result = engine.run(
            [],
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
        )

        assert result.total_trades == 0
        assert result.total_pnl == 0.0
