"""Tests for src/strategy/sizing.py — PositionSizer and kelly_criterion."""

import pytest

from src.config import Settings
from src.strategy.sizing import PositionSizer, BetSize


def _settings(**overrides) -> Settings:
    defaults = dict(
        min_bet_usd=2.0,
        max_bet_usd=5.0,
        max_daily_risk_usd=50.0,
        max_risk_per_city_usd=20.0,
        max_trades_per_run=10,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _sizer(**overrides) -> PositionSizer:
    return PositionSizer(_settings(**overrides))


# ── TestCalculateBetSize ─────────────────────────────────────────────


class TestCalculateBetSize:
    def test_basic_bet(self):
        sizer = _sizer()
        result = sizer.calculate_bet_size(edge=0.25, price=0.10)

        assert result is not None
        assert result.amount_usd >= 2.0
        assert result.shares == pytest.approx(result.amount_usd / 0.10)
        assert result.potential_profit == pytest.approx(result.shares * 0.90)
        assert result.potential_loss == pytest.approx(result.amount_usd)

    def test_higher_edge_gives_larger_bet(self):
        sizer = _sizer()
        low = sizer.calculate_bet_size(edge=0.10, price=0.10)
        high = sizer.calculate_bet_size(edge=0.40, price=0.10)

        assert low is not None and high is not None
        assert high.amount_usd > low.amount_usd

    def test_capped_at_max_bet(self):
        sizer = _sizer(max_bet_usd=5.0)
        result = sizer.calculate_bet_size(edge=1.0, price=0.05)

        assert result is not None
        assert result.amount_usd <= 5.0

    def test_capped_at_remaining_daily_risk(self):
        sizer = _sizer(max_daily_risk_usd=50.0, max_bet_usd=10.0)
        result = sizer.calculate_bet_size(
            edge=0.50, price=0.10, current_daily_risk=47.0,
        )

        assert result is not None
        assert result.amount_usd <= 3.0

    def test_capped_at_remaining_city_risk(self):
        sizer = _sizer(max_risk_per_city_usd=20.0, max_bet_usd=10.0)
        result = sizer.calculate_bet_size(
            edge=0.50, price=0.10, current_city_risk=17.0,
        )

        assert result is not None
        assert result.amount_usd <= 3.0

    def test_returns_none_when_trades_exceeded(self):
        sizer = _sizer(max_trades_per_run=3)
        result = sizer.calculate_bet_size(edge=0.50, price=0.10, trades_today=3)

        assert result is None

    def test_returns_none_when_daily_risk_exhausted(self):
        sizer = _sizer(max_daily_risk_usd=10.0)
        result = sizer.calculate_bet_size(
            edge=0.50, price=0.10, current_daily_risk=10.0,
        )

        assert result is None

    def test_returns_none_when_city_risk_exhausted(self):
        sizer = _sizer(max_risk_per_city_usd=5.0)
        result = sizer.calculate_bet_size(
            edge=0.50, price=0.10, current_city_risk=5.0,
        )

        assert result is None

    def test_returns_none_when_amount_below_min(self):
        sizer = _sizer(min_bet_usd=2.0, max_daily_risk_usd=50.0, max_risk_per_city_usd=20.0)
        # remaining daily = 50 - 49 = 1 < min_bet 2
        result = sizer.calculate_bet_size(
            edge=0.50, price=0.10, current_daily_risk=49.0,
        )

        assert result is None


# ── TestBetSizeFields ────────────────────────────────────────────────


class TestBetSizeFields:
    def test_shares_calculation(self):
        sizer = _sizer()
        result = sizer.calculate_bet_size(edge=0.25, price=0.10)

        assert result is not None
        assert result.shares == pytest.approx(result.amount_usd / result.price)

    def test_profit_and_loss(self):
        sizer = _sizer()
        result = sizer.calculate_bet_size(edge=0.25, price=0.10)

        assert result is not None
        assert result.potential_profit == pytest.approx(result.shares * (1.0 - result.price))
        assert result.potential_loss == pytest.approx(result.amount_usd)


# ── TestKellyCriterion ───────────────────────────────────────────────


class TestKellyCriterion:
    def test_fair_bet_returns_zero(self):
        sizer = _sizer()
        assert sizer.kelly_criterion(p_model=0.5, price=0.5) == pytest.approx(0.0)

    def test_favorable_bet_positive(self):
        sizer = _sizer()
        result = sizer.kelly_criterion(p_model=0.7, price=0.5, fraction=0.25)

        assert result > 0.0

    def test_unfavorable_bet_clamped_to_zero(self):
        sizer = _sizer()
        result = sizer.kelly_criterion(p_model=0.3, price=0.5)

        assert result == 0.0

    def test_edge_case_price_zero(self):
        sizer = _sizer()
        assert sizer.kelly_criterion(p_model=0.5, price=0.0) == 0.0

    def test_edge_case_price_one(self):
        sizer = _sizer()
        assert sizer.kelly_criterion(p_model=0.5, price=1.0) == 0.0
