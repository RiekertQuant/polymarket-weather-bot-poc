"""Tests for src/strategy/decision.py — DecisionEngine."""

from datetime import date

import pytest

from src.config import Settings
from src.polymarket.client_base import Market
from src.strategy.decision import DecisionEngine, TradingDecision
from src.strategy.filters import MarketFilter
from src.strategy.sizing import PositionSizer
from src.weather.probability import ProbabilityResult


def _make_market(
    market_id: str = "m1",
    yes_price: float = 0.05,
    city: str = "New York City",
    threshold: float = 30.0,
    active: bool = True,
) -> Market:
    return Market(
        id=market_id,
        title=f"Will {city} hit {threshold}°C?",
        description="",
        end_date=date(2026, 2, 1),
        yes_price=yes_price,
        volume=1000.0,
        active=active,
        city=city,
        threshold_celsius=threshold,
        target_date=date(2026, 2, 1),
        comparison=">=",
    )


def _make_prob(p_calibrated: float = 0.85, forecast_temp: float = 32.0) -> ProbabilityResult:
    return ProbabilityResult(
        p_raw=p_calibrated,
        p_calibrated=p_calibrated,
        forecast_temp=forecast_temp,
        threshold=30.0,
        sigma=2.0,
        comparison=">=",
    )


def _settings() -> Settings:
    return Settings(
        min_bet_usd=2.0,
        max_bet_usd=5.0,
        max_daily_risk_usd=50.0,
        max_risk_per_city_usd=20.0,
        max_trades_per_run=10,
        price_min=0.001,
        price_max=0.25,
        skip_price_min=0.40,
        skip_price_max=0.60,
        min_edge_absolute=0.15,
        min_edge_relative=0.10,
    )


def _engine(settings: Settings | None = None) -> DecisionEngine:
    s = settings or _settings()
    return DecisionEngine(
        market_filter=MarketFilter(s),
        position_sizer=PositionSizer(s),
    )


# ── TestEvaluateMarket ───────────────────────────────────────────────


class TestEvaluateMarket:
    def test_trade_when_filters_pass_and_edge_sufficient(self):
        engine = _engine()
        market = _make_market(yes_price=0.05)
        prob = _make_prob(p_calibrated=0.85)

        decision = engine.evaluate_market(market, prob)

        assert decision.should_trade is True
        assert decision.side == "YES"
        assert decision.bet_size is not None
        assert decision.bet_size.amount_usd > 0

    def test_no_trade_when_price_in_5050_zone(self):
        engine = _engine()
        market = _make_market(yes_price=0.50)
        prob = _make_prob(p_calibrated=0.85)

        decision = engine.evaluate_market(market, prob)

        assert decision.should_trade is False
        assert decision.side == "NONE"

    def test_no_trade_when_sizer_returns_none(self):
        """Daily risk exhausted → sizer returns None → no trade."""
        engine = _engine()
        market = _make_market(yes_price=0.05)
        prob = _make_prob(p_calibrated=0.85)

        decision = engine.evaluate_market(
            market, prob, current_daily_risk=50.0,
        )

        assert decision.should_trade is False

    def test_edge_calculation(self):
        engine = _engine()
        market = _make_market(yes_price=0.05)
        prob = _make_prob(p_calibrated=0.85)

        decision = engine.evaluate_market(market, prob)

        assert decision.edge == pytest.approx(0.80)

    def test_threshold_city_date_propagated(self):
        engine = _engine()
        market = _make_market(
            city="London", threshold=12.0, yes_price=0.05,
        )
        prob = _make_prob(p_calibrated=0.85)

        decision = engine.evaluate_market(market, prob)

        assert decision.city == "London"
        assert decision.threshold_celsius == 12.0
        assert decision.target_date == date(2026, 2, 1)

    def test_no_trade_when_edge_insufficient(self):
        engine = _engine()
        market = _make_market(yes_price=0.05)
        prob = _make_prob(p_calibrated=0.10)  # edge = 0.05, too low

        decision = engine.evaluate_market(market, prob)

        assert decision.should_trade is False


# ── TestEvaluateMarkets ──────────────────────────────────────────────


class TestEvaluateMarkets:
    def test_sorted_by_edge_descending(self):
        engine = _engine()
        m1 = _make_market(market_id="m1", yes_price=0.05)
        m2 = _make_market(market_id="m2", yes_price=0.08)
        prob1 = _make_prob(p_calibrated=0.85)  # edge 0.80
        prob2 = _make_prob(p_calibrated=0.90)  # edge 0.82

        decisions = engine.evaluate_markets(
            [m1, m2],
            {"m1": prob1, "m2": prob2},
        )

        assert len(decisions) == 2
        assert decisions[0].edge >= decisions[1].edge

    def test_running_risk_accumulation(self):
        """First trade passes; later one blocked by city limit."""
        s = _settings()
        s = Settings(
            min_bet_usd=2.0,
            max_bet_usd=5.0,
            max_daily_risk_usd=50.0,
            max_risk_per_city_usd=5.0,  # tight city limit
            max_trades_per_run=10,
            price_min=0.001,
            price_max=0.25,
            skip_price_min=0.40,
            skip_price_max=0.60,
            min_edge_absolute=0.15,
            min_edge_relative=0.10,
        )
        engine = _engine(s)

        m1 = _make_market(market_id="m1", yes_price=0.05, city="Seoul")
        m2 = _make_market(market_id="m2", yes_price=0.05, city="Seoul")
        prob = _make_prob(p_calibrated=0.85)

        decisions = engine.evaluate_markets(
            [m1, m2],
            {"m1": prob, "m2": prob},
        )

        traded = [d for d in decisions if d.should_trade]
        # With city limit of $5 and max bet $5, only 1 trade fits
        assert len(traded) == 1

    def test_markets_missing_from_prob_results_skipped(self):
        engine = _engine()
        m1 = _make_market(market_id="m1")
        m2 = _make_market(market_id="m2")
        prob1 = _make_prob()

        decisions = engine.evaluate_markets(
            [m1, m2],
            {"m1": prob1},  # m2 missing
        )

        assert len(decisions) == 1
        assert decisions[0].market_id == "m1"
