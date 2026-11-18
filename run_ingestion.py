"""
End-to-end ingestion pipeline (serial, simple).

Usage:
    python run_ingestion.py                  # incremental
    python run_ingestion.py --recreate       # wipe collection first
    python run_ingestion.py --raw-dir data/raw
"""
import sys
import time
from pathlib import Path

from ingestion.document_loader import load_document
from ingestion.text_cleaner import clean_blocks
from ingestion.chunking import chunk_recursive
from ingestion.embedding import embed_documents
from retrieval.qdrant_upsert import ensure_collection, upsert_chunks
import config


SUPPORTED_EXT = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md", ".csv", ".html", ".htm"}


def ingest(raw_dir: Path, recreate: bool = False) -> None:
    start = time.time()

    if not raw_dir.exists():
        print(f"❌ Raw dir not found: {raw_dir}")
        return

    files = sorted(
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    )
    if not files:
        print(f"❌ No supported files in {raw_dir}")
        return

    print(f"Found {len(files)} file(s) in {raw_dir}\n")

    # 1. Prepare Qdrant
    ensure_collection(recreate=recreate)

    total_chunks = 0
    total_files_ok = 0
    total_files_fail = 0
    summary = []
    global_offset = 0   # ← NEW: global counter so IDs never collide across files

    # 2. Process each file (serial — one at a time)
    for i, f in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] 📄 {f.name}")
        try:
            blocks = load_document(str(f))
            if not blocks:
                print(f"   ⏭️  no content extracted, skipping")
                summary.append((f.name, 0, 0, "empty"))
                continue

            blocks = clean_blocks(blocks)

            chunks = chunk_recursive(blocks, chunk_size=500, chunk_overlap=100)
            if not chunks:
                print(f"   ⏭️  no chunks after cleaning, skipping")
                summary.append((f.name, 0, 0, "no_chunks"))
                continue
            print(f"   ✂️  {len(chunks)} chunks")

            vectors = embed_documents([c["text"] for c in chunks])
            print(f"   🔢 {len(vectors)} vectors")

            uploaded = upsert_chunks(chunks, vectors, start_id=global_offset)
            global_offset += uploaded
            print(f"   ⬆️  {uploaded} points uploaded")

            total_chunks += uploaded
            total_files_ok += 1
            summary.append((f.name, len(chunks), uploaded, "ok"))

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            total_files_fail += 1
            summary.append((f.name, 0, 0, "error"))

    # 3. Summary
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    for name, n_chunks, n_uploaded, status in summary:
        print(f"  {name:<45} {n_chunks:>4} chunks → {n_uploaded:>4} uploaded  [{status}]")
    print("-" * 70)
    print(f"  Files OK:      {total_files_ok}")
    print(f"  Files failed:  {total_files_fail}")
    print(f"  Total chunks:  {total_chunks}")
    print(f"  Time:          {elapsed:.1f}s")
    print("=" * 70)


def _parse_args():
    recreate = "--recreate" in sys.argv
    raw_dir = config.RAW_DATA_DIR
    if "--raw-dir" in sys.argv:
        i = sys.argv.index("--raw-dir")
        raw_dir = Path(sys.argv[i + 1])
    return raw_dir, recreate


if __name__ == "__main__":
    raw_dir, recreate = _parse_args()
    if recreate:
        print("⚠️  --recreate flag set → wiping collection first\n")
    ingest(raw_dir, recreate=recreate)