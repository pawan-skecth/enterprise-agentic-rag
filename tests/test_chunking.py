from ingestion.pdf_loader import load_pdf
from ingestion.text_cleaner import clean_blocks
from ingestion.chunking import chunk_recursive


blocks = clean_blocks(load_pdf("data/raw/sample.pdf", extract_images=False))
chunks = chunk_recursive(blocks)

print("Input blocks:", len(blocks))
print("Output chunks:", len(chunks))
print("-" * 60)

for c in chunks[:5]:
    print(
        f"Chunk {c['chunk_id']:>2} | "
        f"page {c['page']} | "
        f"{len(c['text']):>4} chars | "
        f"{c['text'][:80]}..."
    )