-- Database schema for Polymarket Weather Bot

-- Trades table: records all paper trades
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    market_title TEXT NOT NULL,
    city TEXT NOT NULL,
    side TEXT NOT NULL,
    shares REAL NOT NULL,
    entry_price REAL NOT NULL,
    cost_basis REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    created_at TEXT NOT NULL,
    settled_at TEXT,
    pnl REAL
);

-- Predictions table: stores all probability predictions for ML training
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    city TEXT NOT NULL,
    target_date TEXT NOT NULL,
    threshold_celsius REAL NOT NULL,
    forecast_temp REAL NOT NULL,
    p_raw REAL NOT NULL,
    p_calibrated REAL NOT NULL,
    sigma REAL NOT NULL,
    comparison TEXT NOT NULL DEFAULT '>=',
    market_price REAL,
    created_at TEXT NOT NULL,
    -- Outcome (filled in after settlement)
    actual_temp REAL,
    outcome INTEGER  -- 1 if condition was met, 0 if not, NULL if unknown
);

-- Weather history: stores actual weather outcomes for calibration
CREATE TABLE IF NOT EXISTS weather_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    date TEXT NOT NULL,
    actual_max_temp REAL NOT NULL,
    actual_min_temp REAL,
    forecast_max_temp REAL,
    forecast_error REAL,  -- actual - forecast
    created_at TEXT NOT NULL,
    UNIQUE(city, date)
);

-- Run log: tracks each bot run
CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    markets_scanned INTEGER DEFAULT 0,
    trades_made INTEGER DEFAULT 0,
    total_risk REAL DEFAULT 0,
    status TEXT DEFAULT 'RUNNING',
    error_message TEXT
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trades_city ON trades(city);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_predictions_city ON predictions(city);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_weather_city_date ON weather_history(city, date);
