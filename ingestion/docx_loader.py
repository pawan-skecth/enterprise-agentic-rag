from pathlib import Path
from docx import Document

from ingestion.image_router import process_image


def load_docx(file_path: str, extract_images: bool = True) -> list[dict]:
    """
    Extract text + embedded-image descriptions from a DOCX file.

    Args:
        file_path:      path to the .docx file
        extract_images: if True, extract embedded images and run OCR/VLM

    Returns:
        List with one dict: [{
            "text": str, "page": None, "source": str,
            "images_processed": int
        }]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {file_path}")

    doc = Document(file_path)
    parts = []

    # 1) Paragraphs
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            parts.append(t)

    # 2) Tables
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    # 3) Embedded images
    images_processed = 0
    if extract_images:
        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue
            try:
                img_bytes = rel.target_part.blob
                result = process_image(img_bytes)
                if result["combined_text"]:
                    parts.append(f"[DOCX Image]\n{result['combined_text']}")
                    images_processed += 1
            except Exception as e:
                print(f"[docx_loader] image failed: {e}")

    text = "\n\n".join(parts).strip()
    if not text:
        return []

    return [{
        "text": text,
        "page": None,
        "source": path.name,
        "images_processed": images_processed,
    }]