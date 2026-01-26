"""Paper trading broker implementation."""

import logging
import uuid
from datetime import datetime
from typing import Optional

from src.execution.broker_base import (
    BrokerBase,
    Position,
    PositionStatus,
    TradeResult,
)

logger = logging.getLogger(__name__)


class PaperBroker(BrokerBase):
    """Paper trading broker for simulated trading.

    All trades are simulated - no real money is used.
    """

    def __init__(self, initial_balance: float = 30.0):
        """Initialize paper broker.

        Args:
            initial_balance: Starting balance in USD (default $30).
        """
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._positions: dict[str, Position] = {}
        self._trade_history: list[dict] = []
        logger.info(f"Paper broker initialized with ${initial_balance:.2f}")

    def get_balance(self) -> float:
        """Get current account balance.

        Returns:
            Current balance in USD.
        """
        return self._balance

    def get_initial_balance(self) -> float:
        """Get initial account balance.

        Returns:
            Initial balance in USD.
        """
        return self._initial_balance

    def get_total_pnl(self) -> float:
        """Get total profit/loss.

        Returns:
            Total PnL from settled positions.
        """
        total_pnl = 0.0
        for pos in self._positions.values():
            if pos.pnl is not None:
                total_pnl += pos.pnl
        return total_pnl

    def buy_yes(
        self,
        market_id: str,
        market_title: str,
        city: str,
        amount_usd: float,
        price: float,
    ) -> TradeResult:
        """Buy YES shares (paper trade).

        Args:
            market_id: Market identifier.
            market_title: Market title.
            city: City name.
            amount_usd: Amount to spend.
            price: Price per share.

        Returns:
            TradeResult with execution details.
        """
        # Validate inputs
        if amount_usd <= 0:
            return TradeResult(
                success=False,
                error_message="Amount must be positive",
            )

        if price <= 0 or price >= 1:
            return TradeResult(
                success=False,
                error_message=f"Invalid price: {price}",
            )

        if amount_usd > self._balance:
            return TradeResult(
                success=False,
                error_message=f"Insufficient balance: ${self._balance:.2f} < ${amount_usd:.2f}",
            )

        # Execute paper trade
        shares = amount_usd / price
        position_id = str(uuid.uuid4())[:8]

        position = Position(
            id=position_id,
            market_id=market_id,
            market_title=market_title,
            city=city,
            side="YES",
            shares=shares,
            entry_price=price,
            cost_basis=amount_usd,
            status=PositionStatus.OPEN,
            created_at=datetime.now(),
        )

        # Update state
        self._positions[position_id] = position
        self._balance -= amount_usd

        # Log trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "type": "BUY_YES",
            "market_id": market_id,
            "position_id": position_id,
            "shares": shares,
            "price": price,
            "amount_usd": amount_usd,
            "balance_after": self._balance,
        }
        self._trade_history.append(trade_record)

        logger.info(
            f"PAPER TRADE: Bought {shares:.2f} YES shares @ ${price:.3f} "
            f"for ${amount_usd:.2f} | Balance: ${self._balance:.2f}"
        )

        return TradeResult(
            success=True,
            position_id=position_id,
            shares_filled=shares,
            average_price=price,
        )

    def get_positions(self) -> list[Position]:
        """Get all positions.

        Returns:
            List of all positions.
        """
        return list(self._positions.values())

    def get_open_positions(self) -> list[Position]:
        """Get only open positions.

        Returns:
            List of open positions.
        """
        return [p for p in self._positions.values() if p.status == PositionStatus.OPEN]

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position.

        Args:
            position_id: Position identifier.

        Returns:
            Position if found.
        """
        return self._positions.get(position_id)

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
        position = self._positions.get(position_id)
        if position is None:
            logger.warning(f"Position not found: {position_id}")
            return None

        if position.status != PositionStatus.OPEN:
            logger.warning(f"Position already settled: {position_id}")
            return position

        # Calculate PnL
        if outcome:  # YES wins
            # Each share pays $1
            payout = position.shares * 1.0
            pnl = payout - position.cost_basis
            position.status = PositionStatus.SETTLED_WIN
        else:  # NO wins
            # YES shares worth $0
            payout = 0.0
            pnl = -position.cost_basis
            position.status = PositionStatus.SETTLED_LOSS

        position.pnl = pnl
        position.settled_at = datetime.now()

        # Update balance with payout
        self._balance += payout

        logger.info(
            f"SETTLEMENT: {position_id} | Outcome: {'YES' if outcome else 'NO'} | "
            f"PnL: ${pnl:+.2f} | Balance: ${self._balance:.2f}"
        )

        return position

    def get_trade_history(self) -> list[dict]:
        """Get trade history.

        Returns:
            List of trade records.
        """
        return self._trade_history.copy()

    def reset(self) -> None:
        """Reset broker to initial state."""
        self._balance = self._initial_balance
        self._positions.clear()
        self._trade_history.clear()
        logger.info("Paper broker reset")
