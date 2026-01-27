# Polymarket Weather Trading Bot (POC)

A paper trading bot for Polymarket weather temperature markets. This is a **Proof of Concept** that demonstrates automated trading logic for temperature-based prediction markets.

## ⚠️ Important Disclaimers

- **PAPER TRADING ONLY**: This bot does NOT place real trades. All trading is simulated.
- **NO FINANCIAL ADVICE**: This is for educational and research purposes only.
- **LIVE TRADING DISABLED**: The LiveBroker is intentionally not implemented.

## Features

- **Market Discovery**: Scans Polymarket for weather temperature markets
- **Weather Probability Engine**: Uses Open-Meteo forecasts to estimate true probabilities
- **Asymmetric Trading Strategy**: Only trades cheap mispriced shares with high edge
- **Risk Management**: Configurable position limits per city and daily caps
- **ML Calibration**: Optional calibration module for probability refinement
- **Paper Trading**: Full simulation with PnL tracking

## Supported Cities

- New York City
- London
- Seoul

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Unix/Mac)
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Running the Bot

```bash
# Run single trading cycle (paper mode)
python -m src.runner

# Run tests
pytest -q

# Train calibrator (requires historical data)
python scripts/train_calibrator.py
```

## Backtesting

The bot includes a full backtesting system to evaluate strategy performance on historical data.

```bash
# Run backtest with synthetic data (for testing)
python scripts/run_backtest.py --synthetic

# Run backtest and try to collect real Polymarket data
python scripts/run_backtest.py --collect --max-markets 50

# Run backtest and save report to file
python scripts/run_backtest.py --synthetic --save

# Customize initial balance
python scripts/run_backtest.py --synthetic --initial-balance 5000
```

### Backtest Data Sources

| Data Type | Source | Notes |
|-----------|--------|-------|
| Market prices | Polymarket Gamma/CLOB API | Auto-collected if weather markets exist |
| Historical forecasts | Open-Meteo Historical Forecast API | What was predicted at decision time |
| Actual temperatures | Open-Meteo Historical Weather API | For determining outcomes |

### Backtest Output

The backtester generates:
- **Performance summary**: P&L, win rate, Sharpe ratio, max drawdown
- **Trade log**: Every trade with entry price, model probability, edge, outcome
- **Equity curve**: ASCII visualization of cumulative P&L
- **JSON export**: Full data for further analysis

Reports are saved to `data/backtest_results/`.

## Automated Signal Tracking (GitHub Actions)

Track the bot's theoretical performance over time using GitHub Actions.

### Setup

1. Push this repo to GitHub
2. Go to Settings > Actions > General
3. Enable "Read and write permissions" under Workflow permissions
4. The bot will automatically run daily at 14:00 UTC

### How It Works

- **Daily scan**: GitHub Actions runs `scripts/daily_signal_scan.py` daily
- **Signal logging**: Trade signals are saved to `data/signals/` and committed to the repo
- **Outcome tracking**: Run `python scripts/review_signals.py` to check how signals performed

### Manual Run

```bash
# Run signal scan locally
python scripts/daily_signal_scan.py

# Review past signals and their outcomes
python scripts/review_signals.py
```

### Files Generated

| File | Description |
|------|-------------|
| `data/signals/signals_YYYY-MM-DD.json` | Daily signal scan results |
| `data/signals/latest.json` | Most recent scan |
| `data/signals/history.json` | Running summary of all scans |
| `data/signals/outcomes.json` | Signals with actual outcomes (after review) |

### Viewing Results

- Check the **Actions** tab in GitHub for daily run summaries
- Each run shows markets scanned, signals generated
- Signal files are committed to the repo for permanent tracking
- Run `review_signals.py` to see win/loss rates against actual weather data

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:
- `TRADING_MODE=paper` - Always use paper mode
- `MIN_BET_USD=2.0` - Minimum bet size
- `MAX_BET_USD=5.0` - Maximum bet size
- `MAX_DAILY_RISK_USD=50.0` - Daily risk cap
- `FORECAST_SIGMA=2.0` - Weather forecast uncertainty

## Trading Rules

1. **Skip 50/50**: Ignore markets priced [0.40, 0.60]
2. **Cheap Shares Only**: Price must be in [0.001, 0.10]
3. **Strong Edge Required**: P(model) >= max(0.60, price + 0.30)
4. **Position Limits**: Per-city and daily caps enforced

## Project Structure

```
├── src/
│   ├── config.py           # Pydantic settings
│   ├── runner.py           # Main entry point
│   ├── polymarket/         # Polymarket client modules
│   │   ├── client_base.py  # Base classes
│   │   ├── mock_client.py  # Fixtures-based client
│   │   ├── real_client.py  # API client (best-effort)
│   │   └── parsing.py      # Market title parsing
│   ├── weather/            # Weather forecast modules
│   │   ├── open_meteo.py   # Open-Meteo API client
│   │   └── probability.py  # Probability calculations
│   ├── ml/                 # Machine learning modules
│   │   ├── calibrator.py   # Probability calibration
│   │   └── features.py     # Feature engineering
│   ├── strategy/           # Trading strategy
│   │   ├── filters.py      # Market filtering
│   │   ├── decision.py     # Trading decisions
│   │   └── sizing.py       # Position sizing
│   ├── execution/          # Trade execution
│   │   ├── broker_base.py  # Broker interface
│   │   └── paper_broker.py # Paper trading
│   ├── storage/            # Database
│   │   ├── db.py           # SQLite operations
│   │   └── schema.sql      # Database schema
│   ├── backtest/           # Backtesting system
│   │   ├── data_collector.py  # Polymarket historical data
│   │   ├── weather_history.py # Historical forecasts/actuals
│   │   ├── engine.py       # Backtest simulation engine
│   │   └── report.py       # Report generation
│   └── vision/             # OCR module (stub)
│       └── market_ocr.py   # Image processing
├── tests/
│   ├── fixtures/           # Test data
│   │   ├── markets.json    # Sample markets
│   │   └── orderbooks.json # Sample orderbooks
│   ├── test_market_parsing.py
│   ├── test_probability.py
│   ├── test_strategy_filters.py
│   └── test_paper_broker.py
├── scripts/
│   ├── train_calibrator.py # ML training script
│   └── run_backtest.py     # Backtest runner
└── data/
    └── backtest_cache/     # Cached market/weather data
    └── backtest_results/   # Generated reports
```

## Simulating New Markets

Edit `tests/fixtures/markets.json` to add new market scenarios:

```json
{
  "id": "new-market-id",
  "title": "Will London hit 12°C tomorrow?",
  "description": "Market description...",
  "end_date": "2025-02-01",
  "yes_price": 0.06,
  "volume": 1000.0,
  "active": true
}
```

## Known Limitations / TODOs

- [ ] **Live trading intentionally disabled** - LiveBroker not implemented
- [ ] Real Polymarket API integration is best-effort (may need API key)
- [ ] OCR module is a stub (not implemented)
- [ ] Calibration requires manual outcome labeling
- [ ] No scheduled/continuous running (single run only)
- [x] ~~Weather history collection not automated~~ - Now handled by backtester
- [x] ~~No backtesting capability~~ - Full backtester implemented

### Backtest Limitations
- Polymarket may not always have weather/temperature markets available
- Synthetic data mode available when real market data unavailable
- Historical forecast API data available from Jan 2024 onwards

## License

MIT License - For educational purposes only.
