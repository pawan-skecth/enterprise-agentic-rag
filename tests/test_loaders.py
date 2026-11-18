"""
Quick verification that all document loaders work through
the unified document_loader.load_document() entry point.
"""
from pathlib import Path
from ingestion.document_loader import load_document

RAW = Path("data/raw")

# Pick files that actually exist in data/raw
TEST_FILES = [
    "sample.pdf",
    # "CDM-ICAM_Reference_Architecture.pdf",
    "monitor_job.docx",
    "Trade Journal.xlsx",
    "Trade Journal - september.csv",
    "job_management.html",
]


def run():
    print(f"{'File':<50} {'Blocks':>8} {'Chars':>10} {'Images':>8}")
    print("-" * 80)
    for name in TEST_FILES:
        path = RAW / name
        if not path.exists():
            print(f"{name:<50} {'SKIP (not found)':>28}")
            continue
        try:
            docs = load_document(str(path))
            total_chars = sum(len(d["text"]) for d in docs)
            total_images = sum(d.get("images_processed", 0) for d in docs)
            print(f"{name:<50} {len(docs):>8} {total_chars:>10} {total_images:>8}")
        except Exception as e:
            print(f"{name:<50} ERROR: {e}")


if __name__ == "__main__":
    run()