"""Tests for src/storage/db.py — Database, TradeRepository, PredictionRepository."""

from datetime import datetime
from pathlib import Path

import pytest

from src.execution.broker_base import Position, PositionStatus
from src.storage.db import Database, TradeRepository, PredictionRepository
from src.weather.probability import ProbabilityResult


def _make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.initialize()
    return db


def _make_position(
    position_id: str = "pos-1",
    city: str = "London",
    status: PositionStatus = PositionStatus.OPEN,
    cost_basis: float = 5.0,
) -> Position:
    return Position(
        id=position_id,
        market_id="mkt-1",
        market_title="Will London hit 12°C?",
        city=city,
        side="YES",
        shares=100.0,
        entry_price=0.05,
        cost_basis=cost_basis,
        status=status,
        created_at=datetime(2026, 1, 15, 12, 0, 0),
    )


def _make_prob_result() -> ProbabilityResult:
    return ProbabilityResult(
        p_raw=0.80,
        p_calibrated=0.82,
        forecast_temp=14.0,
        threshold=12.0,
        sigma=2.0,
        comparison=">=",
    )


# ── TestDatabase ─────────────────────────────────────────────────────


class TestDatabase:
    def test_initialize_creates_tables(self, tmp_path):
        db = _make_db(tmp_path)
        conn = db.connect()

        # Verify tables exist by querying sqlite_master
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}

        assert "trades" in table_names
        assert "predictions" in table_names

    def test_connect_returns_same_connection(self, tmp_path):
        db = _make_db(tmp_path)
        conn1 = db.connect()
        conn2 = db.connect()

        assert conn1 is conn2

    def test_close_then_connect_creates_fresh(self, tmp_path):
        db = _make_db(tmp_path)
        conn1 = db.connect()
        db.close()
        conn2 = db.connect()

        assert conn1 is not conn2


# ── TestTradeRepository ──────────────────────────────────────────────


class TestTradeRepository:
    def test_save_and_get_roundtrip(self, tmp_path):
        db = _make_db(tmp_path)
        repo = TradeRepository(db)
        pos = _make_position()

        repo.save_position(pos)
        loaded = repo.get_position("pos-1")

        assert loaded is not None
        assert loaded.id == "pos-1"
        assert loaded.city == "London"
        assert loaded.cost_basis == 5.0

    def test_get_open_positions(self, tmp_path):
        db = _make_db(tmp_path)
        repo = TradeRepository(db)

        repo.save_position(_make_position("p1", status=PositionStatus.OPEN))
        repo.save_position(_make_position("p2", status=PositionStatus.SETTLED_WIN))
        repo.save_position(_make_position("p3", status=PositionStatus.OPEN))

        open_pos = repo.get_open_positions()
        assert len(open_pos) == 2

    def test_get_positions_by_city(self, tmp_path):
        db = _make_db(tmp_path)
        repo = TradeRepository(db)

        repo.save_position(_make_position("p1", city="London"))
        repo.save_position(_make_position("p2", city="Seoul"))
        repo.save_position(_make_position("p3", city="London"))

        london = repo.get_positions_by_city("London")
        assert len(london) == 2

    def test_get_total_city_risk(self, tmp_path):
        db = _make_db(tmp_path)
        repo = TradeRepository(db)

        repo.save_position(_make_position("p1", city="Seoul", cost_basis=3.0, status=PositionStatus.OPEN))
        repo.save_position(_make_position("p2", city="Seoul", cost_basis=7.0, status=PositionStatus.OPEN))
        repo.save_position(_make_position("p3", city="Seoul", cost_basis=10.0, status=PositionStatus.SETTLED_WIN))

        total = repo.get_total_city_risk("Seoul")
        assert total == pytest.approx(10.0)  # only OPEN positions

    def test_upsert_same_id(self, tmp_path):
        db = _make_db(tmp_path)
        repo = TradeRepository(db)

        repo.save_position(_make_position("p1", cost_basis=5.0))
        repo.save_position(_make_position("p1", cost_basis=8.0))

        loaded = repo.get_position("p1")
        assert loaded is not None
        assert loaded.cost_basis == 8.0


# ── TestPredictionRepository ─────────────────────────────────────────


class TestPredictionRepository:
    def test_save_prediction_returns_id_and_increments_count(self, tmp_path):
        db = _make_db(tmp_path)
        repo = PredictionRepository(db)
        prob = _make_prob_result()

        assert repo.get_predictions_count() == 0

        row_id = repo.save_prediction("mkt-1", "London", "2026-02-01", prob, 0.05)
        assert row_id is not None
        assert repo.get_predictions_count() == 1

    def test_update_outcome(self, tmp_path):
        db = _make_db(tmp_path)
        repo = PredictionRepository(db)
        prob = _make_prob_result()

        row_id = repo.save_prediction("mkt-1", "London", "2026-02-01", prob)
        repo.update_outcome(row_id, actual_temp=13.5, outcome=True)

        conn = db.connect()
        row = conn.execute("SELECT actual_temp, outcome FROM predictions WHERE id = ?", (row_id,)).fetchone()
        assert row["actual_temp"] == pytest.approx(13.5)
        assert row["outcome"] == 1

    def test_get_training_data_returns_none_when_insufficient(self, tmp_path):
        db = _make_db(tmp_path)
        repo = PredictionRepository(db)
        prob = _make_prob_result()

        # Save 3 predictions with outcomes (less than min_samples=10)
        for i in range(3):
            row_id = repo.save_prediction(f"m{i}", "London", "2026-02-01", prob)
            repo.update_outcome(row_id, actual_temp=13.0, outcome=True)

        result = repo.get_training_data(min_samples=10)
        assert result is None

    def test_get_training_data_returns_data_when_enough(self, tmp_path):
        db = _make_db(tmp_path)
        repo = PredictionRepository(db)
        prob = _make_prob_result()

        for i in range(15):
            row_id = repo.save_prediction(f"m{i}", "London", "2026-02-01", prob)
            repo.update_outcome(row_id, actual_temp=13.0, outcome=(i % 2 == 0))

        result = repo.get_training_data(min_samples=10)
        assert result is not None
        p_raw, outcomes = result
        assert len(p_raw) == 15
        assert len(outcomes) == 15
