"""Position sizing logic."""

from dataclasses import dataclass
from typing import Optional

from src.config import Settings


@dataclass
class BetSize:
    """Calculated bet size."""

    amount_usd: float  # Amount to bet in USD
    shares: float  # Number of shares to buy
    price: float  # Price per share
    potential_profit: float  # If outcome is YES (pays $1 per share)
    potential_loss: float  # Amount risked


class PositionSizer:
    """Calculates position sizes for trades."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize sizer with settings.

        Args:
            settings: Trading settings. Uses defaults if not provided.
        """
        self.settings = settings or Settings()

    def calculate_bet_size(
        self,
        edge: float,
        price: float,
        current_city_risk: float = 0.0,
        current_daily_risk: float = 0.0,
        trades_today: int = 0,
    ) -> Optional[BetSize]:
        """Calculate bet size based on edge and risk limits.

        Args:
            edge: Expected edge (p_model - price).
            price: Current market price.
            current_city_risk: Risk already taken for this city.
            current_daily_risk: Total risk taken today.
            trades_today: Number of trades already made.

        Returns:
            BetSize if trade is valid, None otherwise.
        """
        # Check trade count limit
        if trades_today >= self.settings.max_trades_per_run:
            return None

        # Check daily risk limit
        remaining_daily = self.settings.max_daily_risk_usd - current_daily_risk
        if remaining_daily <= 0:
            return None

        # Check city risk limit
        remaining_city = self.settings.max_risk_per_city_usd - current_city_risk
        if remaining_city <= 0:
            return None

        # Calculate base bet size
        # Scale between min and max based on edge
        # Higher edge -> closer to max bet
        edge_factor = min(edge / 0.50, 1.0)  # Cap at 50% edge
        base_bet = self.settings.min_bet_usd + (
            edge_factor * (self.settings.max_bet_usd - self.settings.min_bet_usd)
        )

        # Apply limits
        bet_amount = min(
            base_bet,
            remaining_daily,
            remaining_city,
            self.settings.max_bet_usd,
        )

        # Ensure minimum bet
        if bet_amount < self.settings.min_bet_usd:
            return None

        # Calculate shares
        shares = bet_amount / price

        # Calculate potential outcomes
        # If YES wins: we get $1 per share, paid $price per share
        potential_profit = shares * (1.0 - price)
        potential_loss = bet_amount

        return BetSize(
            amount_usd=bet_amount,
            shares=shares,
            price=price,
            potential_profit=potential_profit,
            potential_loss=potential_loss,
        )

    def kelly_criterion(
        self,
        p_model: float,
        price: float,
        fraction: float = 0.25,
    ) -> float:
        """Calculate Kelly criterion bet size.

        Uses fractional Kelly for more conservative sizing.

        Args:
            p_model: Model probability.
            price: Market price.
            fraction: Kelly fraction (0.25 = quarter Kelly).

        Returns:
            Optimal bet fraction of bankroll.
        """
        if price <= 0 or price >= 1:
            return 0.0

        # Kelly formula for binary outcome
        # f* = (p * b - q) / b
        # where p = probability of win, q = 1-p, b = odds received
        b = (1.0 - price) / price  # Odds
        q = 1.0 - p_model

        kelly = (p_model * b - q) / b

        # Apply fraction and ensure non-negative
        return max(0.0, kelly * fraction)
