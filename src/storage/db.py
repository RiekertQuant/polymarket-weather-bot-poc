"""SQLite database management."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.execution.broker_base import Position, PositionStatus
from src.weather.probability import ProbabilityResult

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Path):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create directory for database file if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            SQLite connection.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def initialize(self) -> None:
        """Initialize database schema."""
        schema_path = Path(__file__).parent / "schema.sql"
        if not schema_path.exists():
            logger.warning("Schema file not found, using embedded schema")
            self._create_schema_embedded()
            return

        with open(schema_path) as f:
            schema = f.read()

        conn = self.connect()
        conn.executescript(schema)
        conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def _create_schema_embedded(self) -> None:
        """Create schema without external file."""
        conn = self.connect()
        conn.executescript("""
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
                actual_temp REAL,
                outcome INTEGER
            );

            CREATE TABLE IF NOT EXISTS weather_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                date TEXT NOT NULL,
                actual_max_temp REAL NOT NULL,
                actual_min_temp REAL,
                forecast_max_temp REAL,
                forecast_error REAL,
                created_at TEXT NOT NULL,
                UNIQUE(city, date)
            );

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
        """)
        conn.commit()


class TradeRepository:
    """Repository for trade records."""

    def __init__(self, db: Database):
        """Initialize repository.

        Args:
            db: Database instance.
        """
        self.db = db

    def save_position(self, position: Position) -> None:
        """Save a position to database.

        Args:
            position: Position to save.
        """
        conn = self.db.connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO trades
            (id, market_id, market_title, city, side, shares, entry_price,
             cost_basis, status, created_at, settled_at, pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.id,
                position.market_id,
                position.market_title,
                position.city,
                position.side,
                position.shares,
                position.entry_price,
                position.cost_basis,
                position.status.value,
                position.created_at.isoformat(),
                position.settled_at.isoformat() if position.settled_at else None,
                position.pnl,
            ),
        )
        conn.commit()

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID.

        Args:
            position_id: Position identifier.

        Returns:
            Position if found.
        """
        conn = self.db.connect()
        row = conn.execute(
            "SELECT * FROM trades WHERE id = ?",
            (position_id,),
        ).fetchone()

        if row is None:
            return None

        return self._row_to_position(row)

    def get_open_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of open positions.
        """
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'OPEN'",
        ).fetchall()
        return [self._row_to_position(row) for row in rows]

    def get_positions_by_city(self, city: str) -> list[Position]:
        """Get positions for a city.

        Args:
            city: City name.

        Returns:
            List of positions.
        """
        conn = self.db.connect()
        rows = conn.execute(
            "SELECT * FROM trades WHERE city = ?",
            (city,),
        ).fetchall()
        return [self._row_to_position(row) for row in rows]

    def get_total_city_risk(self, city: str) -> float:
        """Get total open risk for a city.

        Args:
            city: City name.

        Returns:
            Total cost basis of open positions.
        """
        conn = self.db.connect()
        result = conn.execute(
            "SELECT COALESCE(SUM(cost_basis), 0) FROM trades WHERE city = ? AND status = 'OPEN'",
            (city,),
        ).fetchone()
        return result[0] if result else 0.0

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        """Convert database row to Position.

        Args:
            row: Database row.

        Returns:
            Position object.
        """
        return Position(
            id=row["id"],
            market_id=row["market_id"],
            market_title=row["market_title"],
            city=row["city"],
            side=row["side"],
            shares=row["shares"],
            entry_price=row["entry_price"],
            cost_basis=row["cost_basis"],
            status=PositionStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            settled_at=datetime.fromisoformat(row["settled_at"]) if row["settled_at"] else None,
            pnl=row["pnl"],
        )


class PredictionRepository:
    """Repository for prediction records."""

    def __init__(self, db: Database):
        """Initialize repository.

        Args:
            db: Database instance.
        """
        self.db = db

    def save_prediction(
        self,
        market_id: str,
        city: str,
        target_date: str,
        prob_result: ProbabilityResult,
        market_price: Optional[float] = None,
    ) -> int:
        """Save a prediction to database.

        Args:
            market_id: Market identifier.
            city: City name.
            target_date: Target date as ISO string.
            prob_result: Probability calculation result.
            market_price: Current market price.

        Returns:
            Inserted row ID.
        """
        conn = self.db.connect()
        cursor = conn.execute(
            """
            INSERT INTO predictions
            (market_id, city, target_date, threshold_celsius, forecast_temp,
             p_raw, p_calibrated, sigma, comparison, market_price, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                market_id,
                city,
                target_date,
                prob_result.threshold,
                prob_result.forecast_temp,
                prob_result.p_raw,
                prob_result.p_calibrated,
                prob_result.sigma,
                prob_result.comparison,
                market_price,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def update_outcome(
        self,
        prediction_id: int,
        actual_temp: float,
        outcome: bool,
    ) -> None:
        """Update prediction with actual outcome.

        Args:
            prediction_id: Prediction ID.
            actual_temp: Actual temperature observed.
            outcome: Whether condition was met.
        """
        conn = self.db.connect()
        conn.execute(
            """
            UPDATE predictions
            SET actual_temp = ?, outcome = ?
            WHERE id = ?
            """,
            (actual_temp, 1 if outcome else 0, prediction_id),
        )
        conn.commit()

    def get_training_data(self, min_samples: int = 10) -> Optional[tuple]:
        """Get data for training calibrator.

        Args:
            min_samples: Minimum samples required.

        Returns:
            Tuple of (p_raw array, outcome array) if enough data, else None.
        """
        conn = self.db.connect()
        rows = conn.execute(
            """
            SELECT p_raw, outcome FROM predictions
            WHERE outcome IS NOT NULL
            """
        ).fetchall()

        if len(rows) < min_samples:
            return None

        p_raw = [row["p_raw"] for row in rows]
        outcomes = [row["outcome"] for row in rows]
        return p_raw, outcomes

    def get_predictions_count(self) -> int:
        """Get total number of predictions.

        Returns:
            Count of predictions.
        """
        conn = self.db.connect()
        result = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return result[0] if result else 0
