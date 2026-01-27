"""Discord webhook notifications for trade signals."""

import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Sends trade signal alerts to a Discord webhook."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def send_signal_alert(self, result: dict, outcomes_summary: dict | None = None) -> None:
        """Post a signal alert to Discord if there are signals.

        Args:
            result: Scan result dict from scan_markets().
            outcomes_summary: Optional outcomes summary from outcomes.json.
        """
        if not self.webhook_url:
            logger.debug("Discord webhook URL not configured, skipping notification")
            return

        if result.get("signals_generated", 0) == 0:
            logger.debug("No signals to report, skipping Discord notification")
            return

        try:
            payload = self._build_payload(result, outcomes_summary)
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Discord alert sent successfully")
        except Exception:
            logger.warning("Failed to send Discord alert", exc_info=True)

    def _build_payload(self, result: dict, outcomes_summary: dict | None = None) -> dict:
        """Build a Discord webhook payload with embeds.

        Args:
            result: Scan result dict.
            outcomes_summary: Optional outcomes summary with PnL stats.

        Returns:
            JSON-serialisable dict for the Discord webhook API.
        """
        signals = result.get("signals", [])
        signal_lines = []
        for sig in signals:
            price = sig.get("market_price", 0)
            p_model = sig.get("p_model", 0)
            edge = sig.get("edge", 0)
            size = sig.get("recommended_size_usd", 0)
            title = sig.get("title", "Unknown")
            signal_lines.append(
                f"**{title}**\n"
                f"Price: ${price:.3f} | Model: {p_model:.1%} | "
                f"Edge: {edge:+.1%} | Size: ${size:.2f}"
            )

        description = "\n\n".join(signal_lines)

        embed = {
            "title": f"🔔 {result['signals_generated']} Trade Signal(s) Found",
            "description": description,
            "color": 0x00CC66,
            "footer": {
                "text": (
                    f"Scanned {result.get('markets_scanned', '?')} markets "
                    f"• Source: {result.get('data_source', 'unknown')}"
                ),
            },
            "timestamp": result.get(
                "scan_time",
                datetime.now(timezone.utc).isoformat(),
            ),
        }

        if outcomes_summary and outcomes_summary.get("total_resolved", 0) > 0:
            total_pnl = outcomes_summary.get("total_pnl", 0)
            wins = outcomes_summary.get("wins", 0)
            losses = outcomes_summary.get("losses", 0)
            win_rate = outcomes_summary.get("win_rate", 0)
            pending = outcomes_summary.get("pending", 0)

            pnl_sign = "+" if total_pnl >= 0 else ""
            embed["fields"] = [
                {"name": "Total PnL", "value": f"${pnl_sign}{total_pnl:.2f}", "inline": True},
                {"name": "Record", "value": f"{wins}W-{losses}L ({win_rate:.0%})", "inline": True},
                {"name": "Pending", "value": str(pending), "inline": True},
            ]

        return {"embeds": [embed]}
