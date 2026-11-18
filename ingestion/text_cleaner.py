import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalize and clean extracted text before chunking.

    - Fix unicode quirks (smart quotes, dashes, NBSP)
    - Drop control characters
    - Remove page-number / footer noise
    - Collapse excessive blank lines + trailing whitespace
    """
    if not text:
        return ""

    # 1) Unicode normalize (NFKC merges look-alikes)
    text = unicodedata.normalize("NFKC", text)

    # 2) Replace common "smart" characters with plain ones
    replacements = {
        "\u2018": "'", "\u2019": "'",   # ‘ ’
        "\u201c": '"', "\u201d": '"',   # “ ”
        "\u2013": "-", "\u2014": "-",   # – —
        "\u00a0": " ",                   # non-breaking space
        "\u2022": "-",                   # bullet
        "\u2026": "...",                 # ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 3) Remove control chars (keep \n and \t)
    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C"
    )

    # 4) Remove standalone page-number lines (e.g. "Page 5 of 68", "- 12 -", "5")
    lines = []
    for line in text.splitlines():
        s = line.strip()

        # Skip typical footer / page number patterns
        if re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", s, re.IGNORECASE):
            continue
        if re.fullmatch(r"[-–—\s]*\d+[-–—\s]*", s):
            continue

        lines.append(line.rstrip())  # strip trailing spaces only

    # 5) Collapse 3+ blank lines into exactly 2
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6) Collapse runs of spaces / tabs inside lines
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def clean_blocks(blocks: list[dict]) -> list[dict]:
    """
    Apply clean_text() to each block returned by a loader.
    Keeps all other metadata fields intact.
    """
    cleaned = []
    for b in blocks:
        new_text = clean_text(b.get("text", ""))
        if not new_text:
            continue  # drop blocks that became empty after cleaning
        out = dict(b)   # copy metadata
        out["text"] = new_text
        cleaned.append(out)
    return cleaned