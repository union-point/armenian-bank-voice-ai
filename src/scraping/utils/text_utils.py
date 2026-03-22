import re


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    return text


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def remove_html_tags(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)

    return normalize_whitespace(normalize_text(text))


def extract_text_from_elements(elements) -> str:
    texts = []
    for elem in elements:
        text = elem.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)

    combined = " ".join(texts)
    return normalize_text(combined)


def clean_for_rag(text: str) -> str:
    text = remove_html_tags(text)
    text = normalize_whitespace(text)
    text = normalize_text(text)

    # Additional RAG-specific cleaning
    text = remove_rag_noise(text)

    return text


def remove_rag_noise(text: str) -> str:
    """Remove noise that degrades RAG quality."""
    if not text:
        return ""

    # Remove bullet points and symbols
    text = re.sub(r"[•·▪▫●○]", "", text)

    # Remove checkmarks and crosses
    text = re.sub(r"[✓✔✗✘☑☒]", "", text)

    # Remove numbered list prefixes at line start (1., 2., etc.)
    text = re.sub(r"^\d+\.\s*", "", text, flags=re.MULTILINE)

    # Remove lettered list prefixes (a., b., etc.)
    text = re.sub(r"^[a-zA-Z]\.\s*", "", text, flags=re.MULTILINE)

    # Remove Roman numerals at line start
    text = re.sub(r"^[IVXLC]+\.?\s*", "", text, flags=re.MULTILINE)

    # Clean up repeated special characters
    text = re.sub(r"[-=]{3,}", "", text)

    # Remove standalone numbers that are likely page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    return text.strip()


def clean_for_rag_aggressive(text: str) -> str:
    """
    More aggressive cleaning for messy extracted text.
    Use this as a fallback when clean_for_rag isn't enough.
    """
    if not text:
        return ""

    # First pass: remove lines that are mostly noise
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip lines that are only numbers, bullets, or symbols
        if re.match(r"^[\d\s•·▪▫✓✔✗✘]+$", stripped):
            continue

        # Skip very short lines (likely noise)
        if len(stripped) < 3:
            continue

        lines.append(stripped)

    text = "\n".join(lines)

    # Now apply regular cleaning
    text = clean_for_rag(text)

    # Fix common encoding issues
    text = fix_armenian_mojibake(text)

    return text.strip()


def fix_armenian_mojibake(text: str) -> str:
    """
    Fix common mojibake patterns in Armenian text.
    This handles cases where Armenian PDF text was extracted with wrong encoding.
    """
    if not text:
        return text

    # Common mojibake patterns (Windows-1252 misinterpreted as UTF-8)
    mojibake_map = {
        "Ï³ÛùÇ": "պաշտոնական",
        "ÙÇçáóáí": "բանկի",
        "ì³ñÏ³ÛÇÝ": "գրավոր",
        "å³ÛÙ³Ý³·ñ»ñÇ": "դիմումի",
        "´³ÝÏÇ": "հաշվի",
        "Ð³×³Ëáñ¹Ç": "տոկոսի",
        "ï³ñ³Ó³ÛÝáõÃÛáõÝÝ»ñÁ": "տրամադրվող",
        "²ñµÇïñ³Å³ÛÇÝ": "ֆինանսական",
        "¹³ï³ñ³Ý": "պայման",
        "í»×»ñÇ": "վարկի",
        "ÉáõÍíáõÙ": "տրամադրում",
    }

    for broken, fixed in mojibake_map.items():
        text = text.replace(broken, fixed)

    return text


def is_scrapable_text(text: str) -> bool:
    """
    Check if extracted text is actually scrapable (not just noise).
    Returns False if the text appears to be from a scanned image or garbled.
    """
    if not text or len(text.strip()) < 100:
        return False

    # Count meaningful characters vs noise
    noise_chars = set("•·▪▫✓✔✗✘0123456789 \t\n.,:")

    # Check if more than 80% of first 500 chars are noise
    sample = text[:500]
    if not sample:
        return False

    noise_count = sum(1 for c in sample if c in noise_chars)

    if noise_count / len(sample) > 0.8:
        return False

    # Check for valid Armenian characters (U+0530 to U+058F)
    armenian_chars = sum(1 for c in sample if 0x0530 <= ord(c) <= 0x058F)

    # Or check for other readable text (Latin, Cyrillic)
    readable = sum(1 for c in sample if c.isalpha() and ord(c) < 128)
    readable += sum(1 for c in sample if 0x0400 <= ord(c) <= 0x04FF)  # Cyrillic

    if armenian_chars + readable < 10:
        return False

    return True
