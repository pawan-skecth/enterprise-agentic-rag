from pathlib import Path
from bs4 import BeautifulSoup


def load_html(file_path: str) -> list[dict]:
    """
    Extract text from an HTML file, stripping tags/scripts/styles.

    Args:
        file_path: path to the .html / .htm file

    Returns:
        List with one dict: [{"text": str, "page": None, "source": str}]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML not found: {file_path}")

    # Try utf-8 first, then latin-1
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1")

    soup = BeautifulSoup(raw, "lxml")  # 'lxml' parser = faster + more forgiving

    # Drop script/style/nav/footer junk
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()

    # Extract readable text (collapse blank lines)
    lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
    text = "\n".join(line for line in lines if line)

    if not text:
        return []

    return [{"text": text, "page": None, "source": path.name}]