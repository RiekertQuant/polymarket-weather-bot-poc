"""Tests for src/backtest/data_collector.py — HistoricalMarket serialization (no network)."""

from datetime import date, datetime

import pytest

from src.backtest.data_collector import (
    HistoricalMarket,
    PolymarketHistoricalCollector,
    PricePoint,
)


def _market() -> HistoricalMarket:
    return HistoricalMarket(
        condition_id="cond-abc",
        token_id="tok-xyz",
        question="Will NYC hit 30°C on Feb 1?",
        description="Temperature market",
        end_date=date(2026, 2, 1),
        outcome=True,
        city="New York City",
        threshold_celsius=30.0,
        target_date=date(2026, 2, 1),
        comparison=">=",
        price_history=[
            PricePoint(datetime(2026, 1, 30, 12, 0, 0), 0.05),
            PricePoint(datetime(2026, 1, 31, 18, 30, 0), 0.08),
        ],
        volume=5000.0,
    )


# ── TestHistoricalMarketSerialization ────────────────────────────────


class TestHistoricalMarketSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        original = _market()
        data = original.to_dict()
        restored = HistoricalMarket.from_dict(data)

        assert restored.condition_id == original.condition_id
        assert restored.token_id == original.token_id
        assert restored.question == original.question
        assert restored.city == original.city
        assert restored.threshold_celsius == original.threshold_celsius
        assert restored.target_date == original.target_date
        assert restored.comparison == original.comparison
        assert restored.outcome == original.outcome
        assert restored.volume == pytest.approx(original.volume)
        assert len(restored.price_history) == 2

    def test_from_dict_missing_optional_fields_uses_defaults(self):
        minimal = {
            "condition_id": "c1",
            "token_id": "t1",
            "question": "test",
            "end_date": "2026-02-01",
        }
        m = HistoricalMarket.from_dict(minimal)

        assert m.description == ""
        assert m.city is None
        assert m.threshold_celsius is None
        assert m.target_date is None
        assert m.comparison == ">="
        assert m.volume == 0.0
        assert m.price_history == []

    def test_price_history_datetime_roundtrip(self):
        original = _market()
        data = original.to_dict()
        restored = HistoricalMarket.from_dict(data)

        assert restored.price_history[0].timestamp == original.price_history[0].timestamp
        assert restored.price_history[1].price == pytest.approx(0.08)


# ── TestPolymarketHistoricalCollectorCache ───────────────────────────


class TestPolymarketHistoricalCollectorCache:
    def test_save_and_load_cache_roundtrip(self, tmp_path):
        collector = PolymarketHistoricalCollector(cache_dir=tmp_path)
        markets = [_market()]

        collector.save_to_cache(markets, "test_cache.json")
        loaded = collector.load_from_cache("test_cache.json")

        assert len(loaded) == 1
        assert loaded[0].condition_id == "cond-abc"
        assert loaded[0].city == "New York City"

    def test_load_missing_file_returns_empty(self, tmp_path):
        collector = PolymarketHistoricalCollector(cache_dir=tmp_path)
        loaded = collector.load_from_cache("nonexistent.json")

        assert loaded == []
