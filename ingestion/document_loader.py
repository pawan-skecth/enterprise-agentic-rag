from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.docx_loader import load_docx
from ingestion.excel_loader import load_excel
from ingestion.text_loader import load_text
from ingestion.html_loader import load_html


# Extension → loader function map
_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".xlsx": load_excel,
    ".xls": load_excel,
    ".txt": load_text,
    ".md": load_text,
    ".csv": load_text,
    ".html": load_html,
    ".htm": load_html,
}


def load_document(file_path: str) -> list[dict]:
    """
    Universal entry point. Dispatches to the correct loader
    based on file extension.

    Returns:
        List of dicts: [{"text": str, "page": int | None, "source": str}, ...]
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)

    if loader is None:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(_LOADERS.keys()))}"
        )

    return loader(str(path))