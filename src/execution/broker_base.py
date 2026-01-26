"""Base classes for trade execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(Enum):
    """Position status."""

    OPEN = "OPEN"
    SETTLED_WIN = "SETTLED_WIN"
    SETTLED_LOSS = "SETTLED_LOSS"
    CLOSED = "CLOSED"


@dataclass
class Position:
    """Represents a trading position."""

    id: str
    market_id: str
    market_title: str
    city: str
    side: str  # "YES" or "NO"
    shares: float
    entry_price: float
    cost_basis: float  # Total amount paid
    status: PositionStatus
    created_at: datetime
    settled_at: Optional[datetime] = None
    pnl: Optional[float] = None  # Profit/loss after settlement


@dataclass
class TradeResult:
    """Result of a trade execution."""

    success: bool
    position_id: Optional[str] = None
    error_message: Optional[str] = None
    shares_filled: float = 0.0
    average_price: float = 0.0


class BrokerBase(ABC):
    """Abstract base class for brokers."""

    @abstractmethod
    def get_balance(self) -> float:
        """Get current account balance in USD.

        Returns:
            Account balance.
        """
        pass

    @abstractmethod
    def buy_yes(
        self,
        market_id: str,
        market_title: str,
        city: str,
        amount_usd: float,
        price: float,
    ) -> TradeResult:
        """Buy YES shares.

        Args:
            market_id: Market identifier.
            market_title: Market title.
            city: City name.
            amount_usd: Amount to spend.
            price: Price per share.

        Returns:
            TradeResult with execution details.
        """
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of open positions.
        """
        pass

    @abstractmethod
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position.

        Args:
            position_id: Position identifier.

        Returns:
            Position if found.
        """
        pass

    @abstractmethod
    def settle_position(
        self,
        position_id: str,
        outcome: bool,
    ) -> Optional[Position]:
        """Settle a position with outcome.

        Args:
            position_id: Position identifier.
            outcome: True if YES wins, False if NO wins.

        Returns:
            Updated position with PnL.
        """
        pass


class LiveBroker(BrokerBase):
    """Live broker for real trading.

    WARNING: NOT IMPLEMENTED. This is a stub only.
    DO NOT use for real trading.
    """

    def __init__(self):
        """Initialize live broker.

        Raises:
            NotImplementedError: Always - live trading not implemented.
        """
        # TODO: Implement live broker with Polymarket API integration
        # This would require:
        # - Private key management
        # - CLOB API authentication
        # - Order signing
        # - Position tracking
        raise NotImplementedError(
            "LiveBroker is not implemented. Use PaperBroker for testing. "
            "Live trading is intentionally disabled for safety."
        )

    def get_balance(self) -> float:
        raise NotImplementedError("Live trading not implemented")

    def buy_yes(
        self,
        market_id: str,
        market_title: str,
        city: str,
        amount_usd: float,
        price: float,
    ) -> TradeResult:
        raise NotImplementedError("Live trading not implemented")

    def get_positions(self) -> list[Position]:
        raise NotImplementedError("Live trading not implemented")

    def get_position(self, position_id: str) -> Optional[Position]:
        raise NotImplementedError("Live trading not implemented")

    def settle_position(self, position_id: str, outcome: bool) -> Optional[Position]:
        raise NotImplementedError("Live trading not implemented")
