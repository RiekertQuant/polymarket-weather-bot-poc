"""Backtest report generation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.backtest.engine import BacktestResult, BacktestTrade, DailySnapshot

logger = logging.getLogger(__name__)


class BacktestReport:
    """Generates reports from backtest results."""

    def __init__(self, result: BacktestResult):
        """Initialize report generator.

        Args:
            result: Backtest result to report on.
        """
        self.result = result

    def summary(self) -> str:
        """Generate text summary of backtest results.

        Returns:
            Formatted text summary.
        """
        r = self.result

        lines = [
            "=" * 70,
            "BACKTEST REPORT",
            "=" * 70,
            "",
            f"Period: {r.start_date} to {r.end_date} ({r.trading_days} days)",
            f"Initial Balance: ${r.config.initial_balance:,.2f}",
            "",
            "-" * 70,
            "PERFORMANCE SUMMARY",
            "-" * 70,
            "",
            f"  Total P&L:        ${r.total_pnl:+,.2f}",
            f"  Total Return:     {r.total_return_pct:+.2f}%",
            f"  Final Balance:    ${r.config.initial_balance + r.total_pnl:,.2f}",
            "",
            f"  Total Trades:     {r.total_trades}",
            f"  Winning Trades:   {r.winning_trades}",
            f"  Losing Trades:    {r.losing_trades}",
            f"  Win Rate:         {r.win_rate * 100:.1f}%",
            "",
            f"  Avg Trade P&L:    ${r.avg_trade_pnl:+.2f}",
            f"  Max Win:          ${r.max_win:+.2f}",
            f"  Max Loss:         ${r.max_loss:+.2f}",
            "",
            "-" * 70,
            "RISK METRICS",
            "-" * 70,
            "",
            f"  Max Drawdown:     ${r.max_drawdown:.2f} ({r.max_drawdown_pct:.2f}%)",
            f"  Sharpe Ratio:     {r.sharpe_ratio:.2f}" if r.sharpe_ratio else "  Sharpe Ratio:     N/A",
            "",
        ]

        # Add city breakdown if available
        if r.pnl_by_city:
            lines.extend([
                "-" * 70,
                "P&L BY CITY",
                "-" * 70,
                "",
            ])

            for city in sorted(r.pnl_by_city.keys()):
                pnl = r.pnl_by_city[city]
                trades = r.trades_by_city.get(city, 0)
                lines.append(f"  {city:20s}  ${pnl:+8.2f}  ({trades} trades)")

            lines.append("")

        # Add strategy settings
        lines.extend([
            "-" * 70,
            "STRATEGY SETTINGS",
            "-" * 70,
            "",
            f"  Price Range:      [{r.config.price_min:.3f}, {r.config.price_max:.3f}]",
            f"  Skip 50/50:       [{r.config.skip_price_min:.2f}, {r.config.skip_price_max:.2f}]",
            f"  Min Edge Abs:     {r.config.min_edge_absolute:.2f}",
            f"  Min Edge Rel:     {r.config.min_edge_relative:.2f}",
            f"  Bet Range:        [${r.config.min_bet_usd:.2f}, ${r.config.max_bet_usd:.2f}]",
            f"  Forecast Sigma:   {r.config.forecast_sigma:.1f}C",
            f"  Slippage:         {r.config.slippage * 100:.1f}%",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def trade_log(self, max_trades: Optional[int] = None) -> str:
        """Generate trade log.

        Args:
            max_trades: Maximum trades to show.

        Returns:
            Formatted trade log.
        """
        trades = self.result.trades
        if max_trades:
            trades = trades[:max_trades]

        lines = [
            "-" * 100,
            "TRADE LOG",
            "-" * 100,
            "",
            f"{'Date':<12} {'City':<15} {'Threshold':<10} {'Price':<8} {'P(model)':<10} {'Edge':<8} {'Cost':<10} {'P&L':<10} {'Result':<8}",
            "-" * 100,
        ]

        for trade in trades:
            result_str = ""
            pnl_str = ""

            if trade.resolved:
                result_str = "WIN" if (trade.pnl and trade.pnl > 0) else "LOSS"
                pnl_str = f"${trade.pnl:+.2f}" if trade.pnl else ""
            else:
                result_str = "OPEN"

            lines.append(
                f"{trade.entry_date!s:<12} "
                f"{trade.city:<15} "
                f"{trade.threshold_celsius:>6.1f}C   "
                f"${trade.entry_price:<6.3f} "
                f"{trade.p_model:<10.3f} "
                f"{trade.edge:+.3f}   "
                f"${trade.cost_basis:<8.2f} "
                f"{pnl_str:<10} "
                f"{result_str:<8}"
            )

        lines.extend(["", "-" * 100])

        return "\n".join(lines)

    def equity_curve_ascii(self, width: int = 60, height: int = 15) -> str:
        """Generate ASCII equity curve.

        Args:
            width: Chart width in characters.
            height: Chart height in rows.

        Returns:
            ASCII art equity curve.
        """
        snapshots = self.result.daily_snapshots

        if not snapshots:
            return "No daily data available for equity curve."

        # Get cumulative P&L values
        values = [s.cumulative_pnl for s in snapshots]

        if len(values) < 2:
            return "Insufficient data for equity curve."

        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val

        if val_range == 0:
            val_range = 1  # Avoid division by zero

        lines = [
            "",
            "-" * (width + 10),
            "EQUITY CURVE (Cumulative P&L)",
            "-" * (width + 10),
            "",
        ]

        # Build the chart
        chart = [[" " for _ in range(width)] for _ in range(height)]

        # Sample values to fit width
        step = max(1, len(values) // width)
        sampled = values[::step][:width]

        for x, val in enumerate(sampled):
            # Normalize to chart height
            y = int((val - min_val) / val_range * (height - 1))
            y = height - 1 - y  # Invert for display
            y = max(0, min(height - 1, y))
            chart[y][x] = "*"

        # Add y-axis labels
        for i, row in enumerate(chart):
            if i == 0:
                label = f"${max_val:>+8.2f} |"
            elif i == height - 1:
                label = f"${min_val:>+8.2f} |"
            elif i == height // 2:
                mid = (max_val + min_val) / 2
                label = f"${mid:>+8.2f} |"
            else:
                label = "           |"

            lines.append(label + "".join(row))

        # X-axis
        lines.append("           +" + "-" * width)
        lines.append(f"           {snapshots[0].date}{'':>{width-22}}{snapshots[-1].date}")
        lines.append("")

        return "\n".join(lines)

    def full_report(self) -> str:
        """Generate full report with all sections.

        Returns:
            Complete formatted report.
        """
        sections = [
            self.summary(),
            "",
            self.equity_curve_ascii(),
            "",
            self.trade_log(max_trades=50),
        ]

        return "\n".join(sections)

    def save_report(self, output_dir: Path = Path("data/backtest_results")) -> Path:
        """Save report to file.

        Args:
            output_dir: Directory for output files.

        Returns:
            Path to saved report.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save text report
        report_path = output_dir / f"backtest_report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(self.full_report())

        logger.info(f"Saved text report to {report_path}")

        # Save JSON data
        json_path = output_dir / f"backtest_data_{timestamp}.json"
        self.save_json(json_path)

        return report_path

    def save_json(self, path: Path) -> None:
        """Save results as JSON for further analysis.

        Args:
            path: Output file path.
        """
        r = self.result

        data = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "start_date": r.start_date.isoformat(),
                "end_date": r.end_date.isoformat(),
                "trading_days": r.trading_days,
                "initial_balance": r.config.initial_balance,
                "final_balance": r.config.initial_balance + r.total_pnl,
                "total_pnl": r.total_pnl,
                "total_return_pct": r.total_return_pct,
                "total_trades": r.total_trades,
                "winning_trades": r.winning_trades,
                "losing_trades": r.losing_trades,
                "win_rate": r.win_rate,
                "avg_trade_pnl": r.avg_trade_pnl,
                "max_win": r.max_win,
                "max_loss": r.max_loss,
                "max_drawdown": r.max_drawdown,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
            },
            "config": {
                "initial_balance": r.config.initial_balance,
                "min_bet_usd": r.config.min_bet_usd,
                "max_bet_usd": r.config.max_bet_usd,
                "max_daily_risk_usd": r.config.max_daily_risk_usd,
                "max_risk_per_city_usd": r.config.max_risk_per_city_usd,
                "price_min": r.config.price_min,
                "price_max": r.config.price_max,
                "skip_price_min": r.config.skip_price_min,
                "skip_price_max": r.config.skip_price_max,
                "min_edge_absolute": r.config.min_edge_absolute,
                "min_edge_relative": r.config.min_edge_relative,
                "forecast_sigma": r.config.forecast_sigma,
                "slippage": r.config.slippage,
            },
            "pnl_by_city": r.pnl_by_city,
            "trades_by_city": r.trades_by_city,
            "trades": [
                {
                    "market_id": t.market_id,
                    "city": t.city,
                    "threshold_celsius": t.threshold_celsius,
                    "target_date": t.target_date.isoformat(),
                    "entry_date": t.entry_date.isoformat(),
                    "entry_price": t.entry_price,
                    "shares": t.shares,
                    "cost_basis": t.cost_basis,
                    "p_model": t.p_model,
                    "edge": t.edge,
                    "forecast_temp": t.forecast_temp,
                    "actual_temp": t.actual_temp,
                    "outcome": t.outcome,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "resolved": t.resolved,
                }
                for t in r.trades
            ],
            "daily_snapshots": [
                {
                    "date": s.date.isoformat(),
                    "balance": s.balance,
                    "open_positions": s.open_positions,
                    "daily_pnl": s.daily_pnl,
                    "cumulative_pnl": s.cumulative_pnl,
                    "trades_made": s.trades_made,
                    "win_rate": s.win_rate,
                }
                for s in r.daily_snapshots
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved JSON data to {path}")


def print_quick_summary(result: BacktestResult) -> None:
    """Print a quick one-line summary.

    Args:
        result: Backtest result.
    """
    print(
        f"Backtest: {result.total_trades} trades | "
        f"Win Rate: {result.win_rate*100:.1f}% | "
        f"P&L: ${result.total_pnl:+.2f} ({result.total_return_pct:+.1f}%) | "
        f"Sharpe: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "N/A"
    )
