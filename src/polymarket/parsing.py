"""Parse Polymarket market titles to extract weather information."""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional


@dataclass
class ParsedMarket:
    """Parsed information from a market title."""

    city: Optional[str] = None
    threshold_celsius: Optional[float] = None
    target_date: Optional[date] = None
    comparison: str = ">="  # ">=" for "hit X°C or higher", "<" for "below"


# City name variations and their canonical names
CITY_ALIASES = {
    # New York
    "new york": "New York City",
    "new york city": "New York City",
    "nyc": "New York City",
    "ny": "New York City",
    # London
    "london": "London",
    # Seoul
    "seoul": "Seoul",
    # Add more cities as needed
}

# Regex patterns for parsing
TEMP_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*[cC](?:elsius)?",
    re.IGNORECASE,
)

DATE_PATTERNS = [
    (re.compile(r"\btomorrow\b", re.IGNORECASE), lambda: date.today() + timedelta(days=1)),
    (re.compile(r"\btoday\b", re.IGNORECASE), lambda: date.today()),
    (
        re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b"),
        None,  # Will be handled specially
    ),
]


def normalize_city(text: str) -> Optional[str]:
    """Normalize city name to canonical form.

    Args:
        text: Text that may contain a city name.

    Returns:
        Canonical city name if found, None otherwise.
    """
    text_lower = text.lower()
    for alias, canonical in CITY_ALIASES.items():
        if alias in text_lower:
            return canonical
    return None


def extract_temperature(text: str) -> Optional[float]:
    """Extract temperature threshold from text.

    Args:
        text: Text containing temperature information.

    Returns:
        Temperature in Celsius if found, None otherwise.
    """
    match = TEMP_PATTERN.search(text)
    if match:
        return float(match.group(1))
    return None


def extract_date(text: str) -> Optional[date]:
    """Extract target date from text.

    Args:
        text: Text containing date information.

    Returns:
        Target date if found, None otherwise.
    """
    # Check for relative dates
    if re.search(r"\btomorrow\b", text, re.IGNORECASE):
        return date.today() + timedelta(days=1)
    if re.search(r"\btoday\b", text, re.IGNORECASE):
        return date.today()

    # Check for explicit dates (MM/DD/YYYY or DD/MM/YYYY)
    date_match = re.search(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b", text)
    if date_match:
        m, d, y = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
        if y < 100:
            y += 2000
        try:
            return date(y, m, d)
        except ValueError:
            # Try DD/MM/YYYY format
            try:
                return date(y, d, m)
            except ValueError:
                pass

    return None


def detect_comparison(text: str) -> str:
    """Detect comparison type from text.

    Args:
        text: Market title text.

    Returns:
        ">=" for "hit/reach/above" or "<" for "below/under".
    """
    below_patterns = [r"\bbelow\b", r"\bunder\b", r"\bless than\b", r"\blower than\b"]
    for pattern in below_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "<"
    return ">="


def parse_market_title(title: str, description: str = "") -> ParsedMarket:
    """Parse a market title to extract weather trading information.

    Args:
        title: The market title (e.g., "Will London hit 9°C tomorrow?")
        description: Optional market description for additional context.

    Returns:
        ParsedMarket with extracted information.
    """
    combined_text = f"{title} {description}"

    return ParsedMarket(
        city=normalize_city(combined_text),
        threshold_celsius=extract_temperature(combined_text),
        target_date=extract_date(combined_text),
        comparison=detect_comparison(combined_text),
    )


def is_valid_weather_market(parsed: ParsedMarket) -> bool:
    """Check if parsed market has all required fields.

    Args:
        parsed: Parsed market information.

    Returns:
        True if market has city, threshold, and date.
    """
    return (
        parsed.city is not None
        and parsed.threshold_celsius is not None
        and parsed.target_date is not None
    )
