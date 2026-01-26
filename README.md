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
└── scripts/
    └── train_calibrator.py # ML training script
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
- [ ] Weather history collection not automated

## License

MIT License - For educational purposes only.
