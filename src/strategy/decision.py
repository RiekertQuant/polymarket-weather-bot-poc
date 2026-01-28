"""Trading decision logic."""

from dataclasses import dataclass
from datetime import date
from typing import Optional

from src.polymarket.client_base import Market
from src.strategy.filters import MarketFilter, FilterResult, FilterReason
from src.strategy.sizing import PositionSizer, BetSize
from src.weather.probability import ProbabilityResult


@dataclass
class TradingDecision:
    """A trading decision for a market."""

    market_id: str
    market_title: str
    city: str
    threshold_celsius: float
    target_date: date

    # Prices
    market_price: float  # Current YES price
    p_model: float  # Model probability
    edge: float  # p_model - market_price

    # Decision
    should_trade: bool
    side: str  # "YES" or "NO" or "NONE"
    bet_size: Optional[BetSize] = None

    # Filter info
    filter_result: Optional[FilterResult] = None

    # Probability details
    probability_result: Optional[ProbabilityResult] = None


class DecisionEngine:
    """Engine for making trading decisions."""

    def __init__(
        self,
        market_filter: Optional[MarketFilter] = None,
        position_sizer: Optional[PositionSizer] = None,
    ):
        """Initialize decision engine.

        Args:
            market_filter: Market filter for screening.
            position_sizer: Position sizer for bet sizing.
        """
        self.market_filter = market_filter or MarketFilter()
        self.position_sizer = position_sizer or PositionSizer()

    def evaluate_market(
        self,
        market: Market,
        prob_result: ProbabilityResult,
        current_city_risk: float = 0.0,
        current_daily_risk: float = 0.0,
        trades_today: int = 0,
    ) -> TradingDecision:
        """Evaluate a market and make trading decision.

        Args:
            market: Market to evaluate.
            prob_result: Probability calculation result.
            current_city_risk: Current risk for this city.
            current_daily_risk: Current daily risk total.
            trades_today: Number of trades already made today.

        Returns:
            TradingDecision with all details.
        """
        price = market.yes_price
        p_model = prob_result.p_calibrated

        # Apply filters
        filter_result = self.market_filter.apply_all_filters(
            price=price,
            p_model=p_model,
            city=market.city,
            active=market.active,
        )

        # Base decision
        decision = TradingDecision(
            market_id=market.id,
            market_title=market.title,
            city=market.city or "Unknown",
            threshold_celsius=market.threshold_celsius or 0.0,
            target_date=market.target_date or date.today(),
            market_price=price,
            p_model=p_model,
            edge=p_model - price,
            should_trade=False,
            side="NONE",
            filter_result=filter_result,
            probability_result=prob_result,
        )

        if not filter_result.passed:
            return decision

        # Calculate position size
        bet_size = self.position_sizer.calculate_bet_size(
            edge=decision.edge,
            price=price,
            current_city_risk=current_city_risk,
            current_daily_risk=current_daily_risk,
            trades_today=trades_today,
        )

        if bet_size is None or bet_size.amount_usd <= 0:
            return decision

        # We have a valid trade
        decision.should_trade = True
        decision.side = "YES"  # We're buying YES at cheap prices
        decision.bet_size = bet_size

        return decision

    def evaluate_markets(
        self,
        markets: list[Market],
        prob_results: dict[str, ProbabilityResult],
        city_risk: Optional[dict[str, float]] = None,
        daily_risk: float = 0.0,
        trades_today: int = 0,
        max_bets_per_city_date: int = 1,
    ) -> list[TradingDecision]:
        """Evaluate multiple markets.

        Args:
            markets: List of markets to evaluate.
            prob_results: Probability results keyed by market ID.
            city_risk: Current risk per city.
            daily_risk: Current daily risk total.
            trades_today: Number of trades already made today.
            max_bets_per_city_date: Maximum bets allowed per city/date combo.
                Prevents correlated bets on same day (e.g., betting both
                17F and 18-19F bands which are highly correlated).

        Returns:
            List of TradingDecisions, sorted by edge descending.
        """
        city_risk = city_risk or {}

        # First pass: evaluate all markets and collect decisions
        all_decisions = []
        for market in markets:
            prob_result = prob_results.get(market.id)
            if prob_result is None:
                continue

            city = market.city or "Unknown"
            decision = self.evaluate_market(
                market=market,
                prob_result=prob_result,
                current_city_risk=city_risk.get(city, 0.0),
                current_daily_risk=daily_risk,
                trades_today=trades_today,
            )
            all_decisions.append(decision)

        # Sort by edge (best opportunities first)
        all_decisions.sort(key=lambda d: d.edge, reverse=True)

        # Second pass: apply correlated bets filter
        # Only allow max_bets_per_city_date trades per city/date combination
        city_date_counts: dict[tuple[str, date], int] = {}
        final_decisions = []

        for decision in all_decisions:
            key = (decision.city, decision.target_date)

            if decision.should_trade and decision.bet_size:
                current_count = city_date_counts.get(key, 0)

                if current_count >= max_bets_per_city_date:
                    # Skip this trade - already have enough bets for this city/date
                    decision.should_trade = False
                    decision.side = "NONE"
                    decision.bet_size = None
                    if decision.filter_result:
                        decision.filter_result = FilterResult(
                            passed=False,
                            reason=FilterReason.CORRELATED_BET,
                            details=f"Already have {current_count} bet(s) for {decision.city} on {decision.target_date}",
                        )
                else:
                    # Allow this trade
                    city_date_counts[key] = current_count + 1
                    city_risk[decision.city] = city_risk.get(decision.city, 0.0) + decision.bet_size.amount_usd
                    daily_risk += decision.bet_size.amount_usd
                    trades_today += 1

            final_decisions.append(decision)

        return final_decisions
