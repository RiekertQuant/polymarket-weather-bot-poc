"""OCR module for extracting market information from images.

This module is a STUB for future implementation.
It defines interfaces for extracting market data from screenshots.

TODO:
- Integrate with an OCR library (Tesseract, EasyOCR, or cloud API)
- Add image preprocessing for better accuracy
- Implement market title extraction
- Implement price extraction from orderbook screenshots
- Add confidence scores for extracted data
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OCRResult:
    """Result of OCR extraction."""

    text: str  # Raw extracted text
    market_title: Optional[str] = None  # Parsed market title
    yes_price: Optional[float] = None  # Extracted YES price
    no_price: Optional[float] = None  # Extracted NO price
    confidence: float = 0.0  # Confidence score (0-1)
    error: Optional[str] = None  # Error message if failed


class MarketOCR:
    """OCR interface for market screenshots.

    This is a STUB implementation that returns placeholder data.
    Actual OCR functionality is not yet implemented.
    """

    def __init__(self, ocr_engine: str = "stub"):
        """Initialize OCR engine.

        Args:
            ocr_engine: OCR engine to use. Currently only "stub" is supported.

        TODO:
            - Support "tesseract" engine
            - Support "easyocr" engine
            - Support cloud OCR APIs (Google Vision, AWS Textract)
        """
        self.engine = ocr_engine

    def extract_from_image(self, image_path: Path) -> OCRResult:
        """Extract market information from an image.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted data.

        TODO:
            - Implement actual OCR extraction
            - Add preprocessing (resize, contrast, etc.)
            - Parse market title from extracted text
            - Extract prices from orderbook display
        """
        if not image_path.exists():
            return OCRResult(
                text="",
                error=f"Image file not found: {image_path}",
            )

        # STUB: Return placeholder indicating OCR is not implemented
        return OCRResult(
            text="[OCR NOT IMPLEMENTED]",
            error="OCR extraction not yet implemented. This is a stub.",
            confidence=0.0,
        )

    def extract_market_title(self, image_path: Path) -> Optional[str]:
        """Extract market title from screenshot.

        Args:
            image_path: Path to screenshot.

        Returns:
            Market title if found, None otherwise.

        TODO:
            - Implement title region detection
            - Use OCR to extract text
            - Clean and validate extracted title
        """
        result = self.extract_from_image(image_path)
        return result.market_title

    def extract_prices(self, image_path: Path) -> tuple[Optional[float], Optional[float]]:
        """Extract YES and NO prices from orderbook screenshot.

        Args:
            image_path: Path to screenshot.

        Returns:
            Tuple of (yes_price, no_price), either may be None.

        TODO:
            - Implement orderbook region detection
            - Extract price values
            - Validate prices sum to ~1.0
        """
        result = self.extract_from_image(image_path)
        return result.yes_price, result.no_price
