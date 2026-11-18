import statistics
from pathlib import Path
from collections import Counter

from ingestion.document_loader import load_document
from ingestion.text_cleaner import clean_blocks
from ingestion.chunking import chunk_recursive


def analyze(file_path: str, chunk_size: int = 500, overlap: int = 100):
    blocks = clean_blocks(load_document(file_path))
    chunks = chunk_recursive(blocks, chunk_size=chunk_size, chunk_overlap=overlap)

    sizes = [len(c["text"]) for c in chunks]
    if not sizes:
        print(f"{file_path}: no chunks produced")
        return

    print(f"\n=== {Path(file_path).name} ===")
    print(f"Chunks: {len(chunks)}")
    print(f"Min:    {min(sizes)} chars")
    print(f"Max:    {max(sizes)} chars")
    print(f"Mean:   {statistics.mean(sizes):.0f} chars")
    print(f"Median: {statistics.median(sizes):.0f} chars")
    print(f"Stdev:  {statistics.stdev(sizes):.0f} chars" if len(sizes) > 1 else "")

    # Bucket distribution
    buckets = Counter()
    for s in sizes:
        if s < 100:
            buckets["<100"] += 1
        elif s < 200:
            buckets["100-200"] += 1
        elif s < 300:
            buckets["200-300"] += 1
        elif s < 400:
            buckets["300-400"] += 1
        elif s < 500:
            buckets["400-500"] += 1
        else:
            buckets["500+"] += 1

    print("Distribution:")
    for label in ["<100", "100-200", "200-300", "300-400", "400-500", "500+"]:
        n = buckets.get(label, 0)
        bar = "█" * n
        print(f"  {label:>8}: {n:>3}  {bar}")


if __name__ == "__main__":
    analyze("data/raw/sample.pdf")
    analyze("data/raw/monitor_job.docx")
    analyze("data/raw/Trade Journal - september.csv")
    analyze("data/raw/job_management.html")