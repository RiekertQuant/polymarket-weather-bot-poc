"""Backtesting engine for historical strategy evaluation."""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from src.backtest.data_collector import HistoricalMarket, PricePoint
from src.backtest.weather_history import WeatherHistoryCollector, HistoricalForecast
from src.config import Settings
from src.strategy.filters import MarketFilter, FilterResult, FilterReason
from src.weather.probability import ProbabilityResult

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A simulated trade in the backtest."""

    # Market info
    market_id: str
    market_title: str
    city: str
    threshold_celsius: float
    target_date: date

    # Trade details
    entry_date: date  # When we entered
    entry_price: float  # Price we bought at
    shares: float
    cost_basis: float  # Amount risked

    # Model info
    p_model: float
    edge: float
    forecast_temp: float

    # Outcome
    outcome: Optional[bool] = None  # True=YES won, False=NO won
    actual_temp: Optional[float] = None
    pnl: Optional[float] = None
    return_pct: Optional[float] = None

    # Resolution
    resolved: bool = False
    resolution_date: Optional[date] = None


@dataclass
class DailySnapshot:
    """State at end of each simulated day."""

    date: date
    balance: float
    open_positions: int
    daily_pnl: float
    cumulative_pnl: float
    trades_made: int
    win_rate: float


@dataclass
class BacktestConfig:
    """Configuration for backtest."""

    # Starting capital
    initial_balance: float = 1000.0

    # Strategy settings (mirrors src.config.Settings)
    min_bet_usd: float = 2.0
    max_bet_usd: float = 5.0
    max_daily_risk_usd: float = 50.0
    max_risk_per_city_usd: float = 20.0
    max_trades_per_day: int = 10

    # Filter settings
    price_min: float = 0.001
    price_max: float = 0.10
    skip_price_min: float = 0.40
    skip_price_max: float = 0.60
    min_edge_absolute: float = 0.60
    min_edge_relative: float = 0.30

    # Weather model
    forecast_sigma: float = 2.0

    # Backtest settings
    days_before_to_trade: int = 1  # Trade N days before resolution
    slippage: float = 0.01  # 1% slippage on entry


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    # Config used
    config: BacktestConfig

    # Date range
    start_date: date
    end_date: date
    trading_days: int

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L
    total_pnl: float
    total_return_pct: float
    avg_trade_pnl: float
    max_win: float
    max_loss: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]

    # Detailed data
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_snapshots: list[DailySnapshot] = field(default_factory=list)

    # By city breakdown
    pnl_by_city: dict[str, float] = field(default_factory=dict)
    trades_by_city: dict[str, int] = field(default_factory=dict)


class BacktestEngine:
    """Engine for backtesting the trading strategy."""

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        weather_collector: Optional[WeatherHistoryCollector] = None,
    ):
        """Initialize backtest engine.

        Args:
            config: Backtest configuration.
            weather_collector: Weather history collector.
        """
        self.config = config or BacktestConfig()
        self.weather_collector = weather_collector or WeatherHistoryCollector()

        # Create settings object for filters
        self._settings = Settings(
            min_bet_usd=self.config.min_bet_usd,
            max_bet_usd=self.config.max_bet_usd,
            max_daily_risk_usd=self.config.max_daily_risk_usd,
            max_risk_per_city_usd=self.config.max_risk_per_city_usd,
            price_min=self.config.price_min,
            price_max=self.config.price_max,
            skip_price_min=self.config.skip_price_min,
            skip_price_max=self.config.skip_price_max,
            min_edge_absolute=self.config.min_edge_absolute,
            min_edge_relative=self.config.min_edge_relative,
            forecast_sigma=self.config.forecast_sigma,
        )

        self.market_filter = MarketFilter(self._settings)

        # State
        self.balance = self.config.initial_balance
        self.trades: list[BacktestTrade] = []
        self.open_positions: list[BacktestTrade] = []
        self.daily_snapshots: list[DailySnapshot] = []

    def _get_price_at_time(
        self,
        market: HistoricalMarket,
        target_datetime: datetime,
    ) -> Optional[float]:
        """Get market price at or before a specific time.

        Args:
            market: Historical market with price history.
            target_datetime: Time to get price for.

        Returns:
            Price at that time, or None if not available.
        """
        if not market.price_history:
            return None

        # Find the latest price before or at target time
        best_price = None
        best_time = None

        for point in market.price_history:
            if point.timestamp <= target_datetime:
                if best_time is None or point.timestamp > best_time:
                    best_time = point.timestamp
                    best_price = point.price

        return best_price

    def _calculate_probability(
        self,
        forecast_temp: float,
        threshold: float,
        comparison: str = ">=",
    ) -> float:
        """Calculate probability using same model as live trading.

        Args:
            forecast_temp: Forecasted max temperature.
            threshold: Temperature threshold.
            comparison: ">=" or "<".

        Returns:
            Model probability.
        """
        from scipy import stats
        import numpy as np

        sigma = self.config.forecast_sigma
        z_score = (threshold - forecast_temp) / sigma

        if comparison == ">=":
            p = 1.0 - stats.norm.cdf(z_score)
        else:
            p = stats.norm.cdf(z_score)

        return float(np.clip(p, 0.0, 1.0))

    def _apply_filters(
        self,
        price: float,
        p_model: float,
        city: str,
    ) -> FilterResult:
        """Apply strategy filters.

        Args:
            price: Market price.
            p_model: Model probability.
            city: City name.

        Returns:
            FilterResult.
        """
        return self.market_filter.apply_all_filters(
            price=price,
            p_model=p_model,
            city=city,
            active=True,
        )

    def _calculate_bet_size(
        self,
        edge: float,
        price: float,
        current_city_risk: float,
        current_daily_risk: float,
        trades_today: int,
    ) -> Optional[float]:
        """Calculate bet size in USD.

        Args:
            edge: Expected edge.
            price: Market price.
            current_city_risk: Current city risk.
            current_daily_risk: Current daily risk.
            trades_today: Trades made today.

        Returns:
            Bet amount in USD, or None if no trade.
        """
        if trades_today >= self.config.max_trades_per_day:
            return None

        remaining_daily = self.config.max_daily_risk_usd - current_daily_risk
        if remaining_daily <= 0:
            return None

        remaining_city = self.config.max_risk_per_city_usd - current_city_risk
        if remaining_city <= 0:
            return None

        # Scale bet by edge
        edge_factor = min(edge / 0.50, 1.0)
        base_bet = self.config.min_bet_usd + (
            edge_factor * (self.config.max_bet_usd - self.config.min_bet_usd)
        )

        bet_amount = min(base_bet, remaining_daily, remaining_city, self.config.max_bet_usd)

        if bet_amount < self.config.min_bet_usd:
            return None

        return bet_amount

    def _resolve_trade(
        self,
        trade: BacktestTrade,
        actual_temp: float,
    ) -> None:
        """Resolve a trade with actual outcome.

        Args:
            trade: Trade to resolve.
            actual_temp: Actual observed temperature.
        """
        trade.actual_temp = actual_temp

        # Determine outcome
        if trade.threshold_celsius is not None:
            # For >= comparison (most markets)
            trade.outcome = actual_temp >= trade.threshold_celsius

        if trade.outcome is None:
            return

        trade.resolved = True
        trade.resolution_date = trade.target_date

        # Calculate P&L
        if trade.outcome:  # YES won
            # We bought YES shares, they pay $1 each
            trade.pnl = trade.shares * (1.0 - trade.entry_price)
        else:  # NO won
            # Our YES shares are worthless
            trade.pnl = -trade.cost_basis

        trade.return_pct = (trade.pnl / trade.cost_basis) * 100 if trade.cost_basis > 0 else 0

    def evaluate_market(
        self,
        market: HistoricalMarket,
        decision_date: date,
        current_city_risk: float = 0.0,
        current_daily_risk: float = 0.0,
        trades_today: int = 0,
    ) -> Optional[BacktestTrade]:
        """Evaluate a market for trading on a specific date.

        Args:
            market: Historical market.
            decision_date: Date of trading decision.
            current_city_risk: Current city risk.
            current_daily_risk: Current daily risk.
            trades_today: Trades already made today.

        Returns:
            BacktestTrade if we should trade, None otherwise.
        """
        if market.city is None or market.threshold_celsius is None or market.target_date is None:
            return None

        # Get price at decision time (end of day)
        decision_datetime = datetime.combine(decision_date, datetime.max.time().replace(microsecond=0))
        price = self._get_price_at_time(market, decision_datetime)

        if price is None:
            logger.debug(f"No price for {market.question[:30]} on {decision_date}")
            return None

        # Get historical forecast (what would we have known?)
        forecast = self.weather_collector.get_forecast_at_decision_time(
            city=market.city,
            target_date=market.target_date,
            days_before=self.config.days_before_to_trade,
        )

        if forecast is None:
            logger.debug(f"No forecast for {market.city} on {market.target_date}")
            return None

        # Calculate model probability
        p_model = self._calculate_probability(
            forecast_temp=forecast.forecast_max_temp,
            threshold=market.threshold_celsius,
            comparison=market.comparison,
        )

        # Apply filters
        filter_result = self._apply_filters(price, p_model, market.city)

        if not filter_result.passed:
            logger.debug(f"Filtered: {market.question[:30]} - {filter_result.reason.value}")
            return None

        edge = p_model - price

        # Calculate bet size
        bet_amount = self._calculate_bet_size(
            edge=edge,
            price=price,
            current_city_risk=current_city_risk,
            current_daily_risk=current_daily_risk,
            trades_today=trades_today,
        )

        if bet_amount is None:
            return None

        # Apply slippage
        execution_price = price * (1 + self.config.slippage)
        shares = bet_amount / execution_price

        return BacktestTrade(
            market_id=market.condition_id,
            market_title=market.question,
            city=market.city,
            threshold_celsius=market.threshold_celsius,
            target_date=market.target_date,
            entry_date=decision_date,
            entry_price=execution_price,
            shares=shares,
            cost_basis=bet_amount,
            p_model=p_model,
            edge=edge,
            forecast_temp=forecast.forecast_max_temp,
        )

    def run(
        self,
        markets: list[HistoricalMarket],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """Run backtest on historical markets.

        Args:
            markets: List of historical markets with price data.
            start_date: Start of backtest period.
            end_date: End of backtest period.

        Returns:
            BacktestResult with all metrics.
        """
        # Reset state
        self.balance = self.config.initial_balance
        self.trades = []
        self.open_positions = []
        self.daily_snapshots = []

        # Determine date range
        if start_date is None:
            # Find earliest market
            dates = [m.target_date for m in markets if m.target_date]
            start_date = min(dates) - timedelta(days=7) if dates else date.today() - timedelta(days=90)

        if end_date is None:
            dates = [m.target_date for m in markets if m.target_date]
            end_date = max(dates) if dates else date.today()

        logger.info(f"Running backtest from {start_date} to {end_date}")
        logger.info(f"Markets: {len(markets)}, Initial balance: ${self.config.initial_balance:.2f}")

        # Track running totals
        cumulative_pnl = 0.0
        peak_balance = self.config.initial_balance
        max_drawdown = 0.0

        # Iterate through each day
        current_date = start_date
        trading_days = 0

        while current_date <= end_date:
            trading_days += 1
            daily_pnl = 0.0
            trades_today = 0
            daily_risk = 0.0
            city_risk: dict[str, float] = {}

            # 1. Resolve any positions that expire today
            positions_to_remove = []
            for position in self.open_positions:
                if position.target_date == current_date:
                    # Get actual weather
                    actual = self.weather_collector.get_actual_weather(
                        position.city,
                        position.target_date,
                    )

                    if actual:
                        self._resolve_trade(position, actual.actual_max_temp)
                        if position.pnl is not None:
                            daily_pnl += position.pnl
                            self.balance += position.pnl

                    positions_to_remove.append(position)

            for pos in positions_to_remove:
                self.open_positions.remove(pos)

            # 2. Look for new trading opportunities
            # Find markets resolving tomorrow (or N days from now)
            target_resolution = current_date + timedelta(days=self.config.days_before_to_trade)

            for market in markets:
                if market.target_date != target_resolution:
                    continue

                # Track city risk
                city = market.city or "Unknown"
                current_city_risk = city_risk.get(city, 0.0)

                trade = self.evaluate_market(
                    market=market,
                    decision_date=current_date,
                    current_city_risk=current_city_risk,
                    current_daily_risk=daily_risk,
                    trades_today=trades_today,
                )

                if trade is not None:
                    # Check if we have enough balance
                    if trade.cost_basis <= self.balance:
                        self.balance -= trade.cost_basis
                        self.trades.append(trade)
                        self.open_positions.append(trade)

                        city_risk[city] = current_city_risk + trade.cost_basis
                        daily_risk += trade.cost_basis
                        trades_today += 1

                        logger.debug(
                            f"{current_date} TRADE: {trade.city} {trade.threshold_celsius}C "
                            f"@ ${trade.entry_price:.3f} | p={trade.p_model:.2f} | "
                            f"edge={trade.edge:.2f} | ${trade.cost_basis:.2f}"
                        )

            # 3. Record daily snapshot
            cumulative_pnl += daily_pnl
            total_balance = self.balance + sum(p.cost_basis for p in self.open_positions)

            # Track drawdown
            if total_balance > peak_balance:
                peak_balance = total_balance
            drawdown = peak_balance - total_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            # Calculate win rate
            resolved_trades = [t for t in self.trades if t.resolved]
            wins = sum(1 for t in resolved_trades if t.pnl and t.pnl > 0)
            win_rate = wins / len(resolved_trades) if resolved_trades else 0.0

            snapshot = DailySnapshot(
                date=current_date,
                balance=self.balance,
                open_positions=len(self.open_positions),
                daily_pnl=daily_pnl,
                cumulative_pnl=cumulative_pnl,
                trades_made=trades_today,
                win_rate=win_rate,
            )
            self.daily_snapshots.append(snapshot)

            current_date += timedelta(days=1)

        # Calculate final metrics
        return self._calculate_results(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            max_drawdown=max_drawdown,
        )

    def _calculate_results(
        self,
        start_date: date,
        end_date: date,
        trading_days: int,
        max_drawdown: float,
    ) -> BacktestResult:
        """Calculate final backtest results.

        Args:
            start_date: Backtest start.
            end_date: Backtest end.
            trading_days: Number of days.
            max_drawdown: Maximum drawdown observed.

        Returns:
            BacktestResult with all metrics.
        """
        resolved_trades = [t for t in self.trades if t.resolved]

        total_trades = len(resolved_trades)
        winning_trades = sum(1 for t in resolved_trades if t.pnl and t.pnl > 0)
        losing_trades = sum(1 for t in resolved_trades if t.pnl and t.pnl < 0)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        total_pnl = sum(t.pnl or 0 for t in resolved_trades)
        total_return_pct = (total_pnl / self.config.initial_balance) * 100

        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        pnls = [t.pnl or 0 for t in resolved_trades]
        max_win = max(pnls) if pnls else 0.0
        max_loss = min(pnls) if pnls else 0.0

        max_drawdown_pct = (max_drawdown / self.config.initial_balance) * 100

        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = None
        if len(pnls) > 1:
            import numpy as np
            returns = np.array(pnls)
            if returns.std() > 0:
                sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))

        # P&L by city
        pnl_by_city: dict[str, float] = {}
        trades_by_city: dict[str, int] = {}

        for trade in resolved_trades:
            city = trade.city
            pnl_by_city[city] = pnl_by_city.get(city, 0.0) + (trade.pnl or 0)
            trades_by_city[city] = trades_by_city.get(city, 0) + 1

        return BacktestResult(
            config=self.config,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            avg_trade_pnl=avg_trade_pnl,
            max_win=max_win,
            max_loss=max_loss,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            daily_snapshots=self.daily_snapshots,
            pnl_by_city=pnl_by_city,
            trades_by_city=trades_by_city,
        )
