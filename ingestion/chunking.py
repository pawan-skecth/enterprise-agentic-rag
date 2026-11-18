from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_recursive(
    blocks: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Split text blocks into smaller chunks using recursive character splitting.

    Args:
        blocks:        output from a loader (or cleaner)
        chunk_size:    max characters per chunk (roughly 125 words)
        chunk_overlap: characters overlapping between chunks (preserves context)

    Returns:
        List of chunk dicts:
            {
              "text": str,
              "chunk_id": int,
              "source": str,
              "page": int | None,
              "sheet": str | None,          # if Excel
              "images_processed": int,
            }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Order matters — try paragraph, then sentence, then word, then char
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_id = 0

    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue

        # Split this block's text
        pieces = splitter.split_text(text)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            chunks.append({
                "text": piece,
                "chunk_id": chunk_id,
                "source": block.get("source", ""),
                "page": block.get("page"),
                "sheet": block.get("sheet"),              # may be None
                "images_processed": block.get("images_processed", 0),
            })
            chunk_id += 1

    return chunks