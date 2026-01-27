"""Tests for src/notifications/discord.py — DiscordNotifier."""

from unittest.mock import patch, MagicMock

import pytest
import requests

from src.notifications.discord import DiscordNotifier


def _make_result(signals_generated: int = 1) -> dict:
    """Build a minimal scan result dict."""
    signals = []
    if signals_generated > 0:
        signals.append(
            {
                "title": "Will NYC hit 30°C?",
                "market_price": 0.05,
                "p_model": 0.85,
                "edge": 0.80,
                "recommended_size_usd": 4.50,
            }
        )
    return {
        "scan_time": "2026-01-27T14:00:00Z",
        "scan_date": "2026-01-27",
        "data_source": "mock_fixtures",
        "markets_scanned": 5,
        "signals_generated": signals_generated,
        "signals": signals,
    }


class TestDiscordNotifier:
    @patch("src.notifications.discord.requests.post")
    def test_send_alert_posts_to_webhook(self, mock_post: MagicMock):
        mock_post.return_value = MagicMock(status_code=204)

        notifier = DiscordNotifier("https://discord.com/api/webhooks/test")
        notifier.send_signal_alert(_make_result(signals_generated=1))

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.args[0] == "https://discord.com/api/webhooks/test"
        payload = call_kwargs.kwargs["json"]
        assert "embeds" in payload
        embed = payload["embeds"][0]
        assert "1 Trade Signal" in embed["title"]
        assert "NYC" in embed["description"]

    @patch("src.notifications.discord.requests.post")
    def test_send_alert_skips_when_no_signals(self, mock_post: MagicMock):
        notifier = DiscordNotifier("https://discord.com/api/webhooks/test")
        notifier.send_signal_alert(_make_result(signals_generated=0))

        mock_post.assert_not_called()

    @patch("src.notifications.discord.requests.post")
    def test_send_alert_handles_request_error(self, mock_post: MagicMock):
        mock_post.side_effect = requests.ConnectionError("network down")

        notifier = DiscordNotifier("https://discord.com/api/webhooks/test")
        # Should not raise
        notifier.send_signal_alert(_make_result(signals_generated=1))

    @patch("src.notifications.discord.requests.post")
    def test_disabled_when_no_url(self, mock_post: MagicMock):
        notifier = DiscordNotifier("")
        notifier.send_signal_alert(_make_result(signals_generated=1))

        mock_post.assert_not_called()
