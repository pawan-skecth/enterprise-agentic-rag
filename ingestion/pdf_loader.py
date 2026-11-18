from pathlib import Path
import fitz  # PyMuPDF

from ingestion.image_router import process_image


def load_pdf(file_path: str, extract_images: bool = True) -> list[dict]:
    """
    Extract text + embedded-image descriptions from a PDF, page by page.

    Args:
        file_path:      path to the PDF file
        extract_images: if True, extract embedded images and run OCR/VLM on them

    Returns:
        List of dicts: [{"text": str, "page": int, "source": str, "images_processed": int}, ...]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc, start=1):
            page_parts = []

            # 1) Text content
            text = page.get_text().strip()
            if text:
                page_parts.append(text)

            # 2) Embedded images → OCR / VLM
            images_processed = 0
            if extract_images:
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    try:
                        base = doc.extract_image(xref)
                        img_bytes = base["image"]
                        result = process_image(img_bytes)
                        if result["combined_text"]:
                            page_parts.append(
                                f"[Page {i} Image]\n{result['combined_text']}"
                            )
                            images_processed += 1
                    except Exception as e:
                        print(f"[pdf_loader] image xref {xref} failed: {e}")

            combined = "\n\n".join(page_parts).strip()
            if combined:
                pages.append({
                    "text": combined,
                    "page": i,
                    "source": path.name,
                    "images_processed": images_processed,
                })

    return pages