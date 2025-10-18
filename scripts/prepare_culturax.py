from pathlib import Path

from datasets import load_dataset

MIN_WORDS = 500
MAX_WORDS = 6000

# Bin targets for uniform sampling: (start, end, fraction)
# We'll multiply these fractions by the target docs for each language
BIN_FRACTIONS = [
    (500, 1000, 0.15),
    (1000, 1500, 0.15),
    (1500, 2000, 0.15),
    (2000, 2500, 0.15),
    (2500, 3000, 0.12),
    (3000, 3500, 0.10),
    (3500, 4000, 0.08),
    (4000, 5000, 0.06),
    (5000, 6000, 0.04),
]

# Language configurations: (language_code, target_docs, sample_size)
LANGUAGES = [
    ("en", 500, 8000),
    ("nl", 200, 4000),
    ("fr", 100, 2500),
]

output_dir = Path(__file__).parent.parent / "data" / "v2" / "documents"
output_dir.mkdir(parents=True, exist_ok=True)

print("CulturaX Dataset Preparation")
print(f"Word count range: {MIN_WORDS}-{MAX_WORDS} words")
print(f"Languages: {', '.join([lang for lang, _, _ in LANGUAGES])}")
print("-" * 60)

total_saved = 0

for lang_code, target_docs, sample_size in LANGUAGES:
    print(f"\n[{lang_code.upper()}] Processing {target_docs} documents with uniform sampling...")

    # Calculate bin targets for this language
    bin_targets = []
    for start, end, fraction in BIN_FRACTIONS:
        target = max(1, int(target_docs * fraction))
        bin_targets.append((start, end, target))

    try:
        dataset_stream = load_dataset(
            "uonlp/CulturaX",
            lang_code,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        print("  ✓ Dataset initialized")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

    # Initialize bins
    bins = {i: [] for i in range(len(bin_targets))}

    docs_processed = 0

    for doc in dataset_stream:
        docs_processed += 1

        if docs_processed % 1000 == 0:
            filled_bins = sum(1 for i, b in bins.items() if len(b) >= bin_targets[i][2])
            total_collected = sum(len(b) for b in bins.values())
            print(f"    Processed: {docs_processed} | Collected: {total_collected}/{target_docs}")

        text = doc["text"]
        word_count = len(text.split())

        # Assign to appropriate bin
        if MIN_WORDS <= word_count <= MAX_WORDS:
            for i, (bin_start, bin_end, target) in enumerate(bin_targets):
                if bin_start <= word_count < bin_end:
                    if len(bins[i]) < target:
                        bins[i].append({"text": text, "word_count": word_count})
                    break

        # Stop if all bins are full or reached sample size
        all_bins_full = all(len(bins[i]) >= bin_targets[i][2] for i in range(len(bin_targets)))
        if all_bins_full or docs_processed >= sample_size:
            break

    total_collected = sum(len(b) for b in bins.values())
    print(f"  ✓ Processed: {docs_processed} | Collected: {total_collected}")

    # Show bin statistics
    print("  Bin statistics:")
    for i, (start, end, target) in enumerate(bin_targets):
        count = len(bins[i])
        status = "✓" if count >= target else "⚠"
        print(f"    {status} {int(start)}-{int(end)}: {count}/{target}")

    if total_collected == 0:
        print("  ✗ No documents collected!")
        continue

    # Save documents with sequential numbering
    doc_counter = 1
    for i in range(len(bin_targets)):
        for doc in bins[i]:
            output_file = output_dir / f"culturax_{lang_code}_{doc_counter}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(doc["text"])
            doc_counter += 1

    print(f"  ✓ Saved {total_collected} documents")
    total_saved += total_collected

print(f"\n{'=' * 60}")
print(f"✓ Total documents saved: {total_saved}")
print(f"✓ Output directory: {output_dir}")

