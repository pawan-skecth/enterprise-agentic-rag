from pathlib import Path


def load_text(file_path: str) -> list[dict]:
    """
    Extract text from a plain-text file (.txt, .md, .csv).

    Args:
        file_path: path to the text file

    Returns:
        List with one dict: [{"text": str, "page": None, "source": str}]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    # Try UTF-8 first, fall back to latin-1 for older files
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    text = text.strip()
    if not text:
        return []

    return [{"text": text, "page": None, "source": path.name}]