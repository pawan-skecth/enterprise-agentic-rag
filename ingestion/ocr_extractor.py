from pathlib import Path
from io import BytesIO
import pytesseract
from PIL import Image

# Hardcode Tesseract path (Windows fallback — avoids PATH issues)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_image(image_input, min_chars: int = 5) -> str:
    """
    Run OCR on an image and return extracted text.

    Args:
        image_input: file path (str/Path) OR image bytes (bytes)
        min_chars:   ignore results shorter than this (likely noise)

    Returns:
        Extracted text (empty string if nothing meaningful found)
    """
    # Load image
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(BytesIO(image_input))
    else:
        raise TypeError("image_input must be a path or bytes")

    # Convert to RGB (Tesseract works best with RGB / L mode)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Run OCR
    text = pytesseract.image_to_string(img).strip()

    return text if len(text) >= min_chars else ""


def has_meaningful_text(text: str, min_words: int = 3) -> bool:
    """
    Simple heuristic: does this OCR output look like real text?
    Used to decide whether to keep OCR results or send image to VLM instead.
    """
    if not text:
        return False
    words = [w for w in text.split() if len(w) > 1]
    return len(words) >= min_words