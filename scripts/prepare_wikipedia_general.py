from pathlib import Path

from datasets import load_dataset

MIN_WORDS = 500
MAX_WORDS = 6000

# Language configurations: (language_code, target_docs, sample_size)
LANGUAGES = [
    ("en", 500, 3000),
    ("nl", 200, 1500),
    ("fr", 200, 1500),
]

output_dir = Path(__file__).parent.parent / "data" / "v2" / "documents"
output_dir.mkdir(parents=True, exist_ok=True)

print("Wikipedia Dataset Preparation")
print(f"Word count range: {MIN_WORDS}-{MAX_WORDS} words")
print(f"Languages: {', '.join([lang for lang, _, _ in LANGUAGES])}")
print("-" * 60)

total_saved = 0

for lang_code, target_docs, sample_size in LANGUAGES:
    print(f"\n[{lang_code.upper()}] Processing {target_docs} documents...")

    try:
        dataset_stream = load_dataset(
            "omarkamali/wikipedia-monthly",
            f"latest.{lang_code}",
            split="train",
            streaming=True,
        )
        print("  ✓ Dataset initialized")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

    filtered_docs = []
    docs_processed = 0

    for doc in dataset_stream:
        docs_processed += 1

        if docs_processed % 500 == 0:
            print(f"    Processed: {docs_processed} | Filtered: {len(filtered_docs)}")

        text = doc["text"]
        word_count = len(text.split())

        if MIN_WORDS <= word_count <= MAX_WORDS:
            filtered_docs.append({"text": text, "word_count": word_count})

        if docs_processed >= sample_size:
            break

    print(f"  ✓ Processed: {docs_processed} | Filtered: {len(filtered_docs)}")

    if not filtered_docs:
        print("  ✗ No documents found in range")
        continue

    num_docs_to_save = min(target_docs, len(filtered_docs))

    for i, doc in enumerate(filtered_docs[:num_docs_to_save], start=1):
        output_file = output_dir / f"wikipedia_general_{lang_code}_{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(doc["text"])

    print(f"  ✓ Saved {num_docs_to_save} documents")
    total_saved += num_docs_to_save

print(f"\n{'=' * 60}")
print(f"✓ Total documents saved: {total_saved}")
print(f"✓ Output directory: {output_dir}")
