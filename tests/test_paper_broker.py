"""Tests for paper broker."""

import pytest

from src.execution.paper_broker import PaperBroker
from src.execution.broker_base import PositionStatus


class TestPaperBrokerBasics:
    """Tests for basic paper broker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.broker = PaperBroker(initial_balance=30.0)

    def test_initial_balance(self):
        """Should start with correct initial balance."""
        assert self.broker.get_balance() == 30.0
        assert self.broker.get_initial_balance() == 30.0

    def test_buy_yes_reduces_balance(self):
        """Buying should reduce balance."""
        result = self.broker.buy_yes(
            market_id="test-market",
            market_title="Test Market",
            city="London",
            amount_usd=5.0,
            price=0.05,
        )

        assert result.success
        assert self.broker.get_balance() == 25.0

    def test_buy_yes_creates_position(self):
        """Buying should create a position."""
        result = self.broker.buy_yes(
            market_id="test-market",
            market_title="Test Market",
            city="London",
            amount_usd=5.0,
            price=0.05,
        )

        assert result.success
        assert result.position_id is not None

        position = self.broker.get_position(result.position_id)
        assert position is not None
        assert position.market_id == "test-market"
        assert position.city == "London"
        assert position.side == "YES"
        assert position.entry_price == 0.05
        assert position.cost_basis == 5.0
        assert position.shares == 100.0  # $5 / $0.05 = 100 shares
        assert position.status == PositionStatus.OPEN

    def test_buy_yes_insufficient_balance(self):
        """Should fail if insufficient balance."""
        result = self.broker.buy_yes(
            market_id="test-market",
            market_title="Test Market",
            city="London",
            amount_usd=50.0,  # More than $30 balance
            price=0.05,
        )

        assert not result.success
        assert "Insufficient balance" in result.error_message
        assert self.broker.get_balance() == 30.0

    def test_buy_yes_invalid_price(self):
        """Should fail with invalid price."""
        result = self.broker.buy_yes(
            market_id="test-market",
            market_title="Test Market",
            city="London",
            amount_usd=5.0,
            price=0.0,
        )

        assert not result.success
        assert "Invalid price" in result.error_message

    def test_multiple_positions(self):
        """Should track multiple positions."""
        self.broker.buy_yes("m1", "Market 1", "London", 5.0, 0.05)
        self.broker.buy_yes("m2", "Market 2", "NYC", 5.0, 0.08)

        positions = self.broker.get_positions()
        assert len(positions) == 2
        assert self.broker.get_balance() == 20.0


class TestPaperBrokerSettlement:
    """Tests for position settlement."""

    def setup_method(self):
        """Set up test fixtures."""
        self.broker = PaperBroker(initial_balance=30.0)

    def test_settle_win(self):
        """Settlement with YES winning should profit."""
        result = self.broker.buy_yes(
            market_id="test-market",
            market_title="Test Market",
            city="London",
            amount_usd=5.0,
            price=0.05,  # 100 shares
        )

        # Settle with YES winning
        position = self.broker.settle_position(result.position_id, outcome=True)

        assert position is not None
        assert position.status == PositionStatus.SETTLED_WIN
        # Payout: 100 shares * $1 = $100
        # PnL: $100 - $5 = $95
        assert position.pnl == 95.0
        # Balance: $25 (after buy) + $100 (payout) = $125
        assert self.broker.get_balance() == 125.0

    def test_settle_loss(self):
        """Settlement with NO winning should lose."""
        result = self.broker.buy_yes(
            market_id="test-market",
            market_title="Test Market",
            city="London",
            amount_usd=5.0,
            price=0.05,
        )

        # Settle with NO winning (YES loses)
        position = self.broker.settle_position(result.position_id, outcome=False)

        assert position is not None
        assert position.status == PositionStatus.SETTLED_LOSS
        # Payout: $0 (YES shares worthless)
        # PnL: $0 - $5 = -$5
        assert position.pnl == -5.0
        # Balance: $25 (after buy) + $0 (payout) = $25
        assert self.broker.get_balance() == 25.0

    def test_settle_nonexistent_position(self):
        """Settling nonexistent position returns None."""
        position = self.broker.settle_position("nonexistent", outcome=True)
        assert position is None

    def test_settle_already_settled(self):
        """Settling already settled position returns unchanged."""
        result = self.broker.buy_yes(
            market_id="test-market",
            market_title="Test Market",
            city="London",
            amount_usd=5.0,
            price=0.05,
        )

        # Settle once
        self.broker.settle_position(result.position_id, outcome=True)
        balance_after_first = self.broker.get_balance()

        # Try to settle again
        position = self.broker.settle_position(result.position_id, outcome=False)

        # Should return position but not change balance
        assert position is not None
        assert position.status == PositionStatus.SETTLED_WIN
        assert self.broker.get_balance() == balance_after_first


class TestPaperBrokerTracking:
    """Tests for position and trade tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.broker = PaperBroker(initial_balance=30.0)

    def test_get_open_positions(self):
        """Should return only open positions."""
        r1 = self.broker.buy_yes("m1", "Market 1", "London", 5.0, 0.05)
        r2 = self.broker.buy_yes("m2", "Market 2", "NYC", 5.0, 0.08)

        # Settle one
        self.broker.settle_position(r1.position_id, outcome=True)

        open_positions = self.broker.get_open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].market_id == "m2"

    def test_trade_history(self):
        """Should track trade history."""
        self.broker.buy_yes("m1", "Market 1", "London", 5.0, 0.05)
        self.broker.buy_yes("m2", "Market 2", "NYC", 3.0, 0.08)

        history = self.broker.get_trade_history()
        assert len(history) == 2
        assert history[0]["type"] == "BUY_YES"
        assert history[0]["amount_usd"] == 5.0
        assert history[1]["amount_usd"] == 3.0

    def test_total_pnl(self):
        """Should track total PnL."""
        r1 = self.broker.buy_yes("m1", "Market 1", "London", 5.0, 0.05)
        r2 = self.broker.buy_yes("m2", "Market 2", "NYC", 5.0, 0.10)

        # Win one, lose one
        self.broker.settle_position(r1.position_id, outcome=True)  # +$95
        self.broker.settle_position(r2.position_id, outcome=False)  # -$5

        total_pnl = self.broker.get_total_pnl()
        assert total_pnl == 90.0  # $95 - $5

    def test_reset(self):
        """Reset should restore initial state."""
        self.broker.buy_yes("m1", "Market 1", "London", 5.0, 0.05)

        self.broker.reset()

        assert self.broker.get_balance() == 30.0
        assert len(self.broker.get_positions()) == 0
        assert len(self.broker.get_trade_history()) == 0
