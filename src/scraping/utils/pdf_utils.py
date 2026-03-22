import io
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def download_file(
    url: str, timeout: int = 60, headers: Optional[dict] = None
) -> Optional[bytes]:
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


def extract_text_from_pdf(pdf_content: bytes) -> Optional[str]:
    if not pdf_content:
        return None

    if len(pdf_content) < 100:
        logger.warning("PDF content too small, skipping")
        return None

    try:
        from pdfminer.high_level import extract_text

        pdf_file = io.BytesIO(pdf_content)
        text = extract_text(pdf_file)
        if text and text.strip():
            return clean_pdf_text(text)
    except Exception as e:
        logger.info(f"pdfminer extraction failed: {e}")

    try:
        import pypdf

        pdf_file = io.BytesIO(pdf_content)
        reader = pypdf.PdfReader(pdf_file)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        result = "\n".join(texts)
        if result.strip():
            return clean_pdf_text(result)
    except Exception as e:
        logger.info(f"pypdf extraction failed: {e}")

    return None


def clean_pdf_text(text: str) -> str:
    # Fix encoding issues first - detect and convert mojibake
    text = fix_encoding_issues(text)

    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove page indicators like "Page X of Y"
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)

    # Remove standalone page numbers (lines that are only numbers)
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)

    # Remove common PDF artifacts: bullet points, checkmarks
    text = re.sub(r"[•·▪▫]", "", text)
    text = re.sub(r"[✓✔✗✘]", "", text)

    # Remove lines that are only bullets or symbols
    text = re.sub(r"^[\s•·▪▫✓✔✗✘]+$", "", text, flags=re.MULTILINE)

    # Clean up hyphenation at line breaks (word wrap artifacts)
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"(\w)\n(\w)", r"\1\2", text)

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    text = re.sub(r"[ \t]+", " ", text)

    lines = []
    prev_empty = False
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                lines.append("")
                prev_empty = True
        else:
            lines.append(stripped)
            prev_empty = False

    text = "\n".join(lines)

    return text.strip()


def fix_encoding_issues(text: str) -> str:
    """
    Fix common encoding issues in extracted PDF text.
    Many PDFs have Armenian text encoded in Windows-1252 or similar
    that gets misinterpreted as UTF-8, resulting in mojibake.
    """
    # Common mojibake patterns for Armenian characters
    # These are Windows-1252 bytes interpreted as UTF-8
    encoding_fixes = {
        # These patterns are indicative of encoding issues
        # Try to detect and fix common Armenian mojibake
        "Ï³ÛùÇ": "պաշտոնական",  # common header
        "ÙÇçáóáí": "բանկի",
        "ì³ñÏ³ÛÇÝ": "գրավոր",
        "å³ÛÙ³Ý³·ñ»ñÇ": "դիմումի",
        "´³ÝÏÇ": "հաշվի",
        "Ð³×³Ëáñ¹Ç": "տոկոսի",
        "ï³ñ³Ó³ÛÝáõÃÛáõÝÝ»ñÁ": "տրամադրվող",
    }

    for broken, fixed in encoding_fixes.items():
        text = text.replace(broken, fixed)

    # Try to decode from common encodings if text contains mojibake indicators
    if any(
        ord(c) > 0xFFFF or (ord(c) > 0xFF and c in "ÏùÇÙì³ñå´ÐïÝ") for c in text[:1000]
    ):
        # Try Windows-1252
        try:
            # First encode to bytes treating the string as latin-1, then decode properly
            test_decode = text.encode("latin-1").decode("windows-1252")
            # Check if result has valid Armenian (U+0530 to U+058F)
            if any(0x0530 <= ord(c) <= 0x058F for c in test_decode):
                text = test_decode
        except Exception:
            pass

    return text
