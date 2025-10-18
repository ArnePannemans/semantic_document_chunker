from pathlib import Path

from datasets import load_dataset

MIN_WORDS = 500
MAX_WORDS = 6000
TARGET_DOCS = 300
SAMPLE_SIZE = 4000

# Bin targets for uniform sampling: (start, end, fraction)
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

output_dir = Path(__file__).parent.parent / "data" / "v2" / "documents"
output_dir.mkdir(parents=True, exist_ok=True)

print("Open Textbooks Dataset Preparation")
print(f"Word count range: {MIN_WORDS}-{MAX_WORDS} words")
print(f"Target documents: {TARGET_DOCS}")
print("-" * 60)

# Calculate bin targets
bin_targets = []
for start, end, fraction in BIN_FRACTIONS:
    target = max(1, int(TARGET_DOCS * fraction))
    bin_targets.append((start, end, target))

print("\nLoading dataset...")

try:
    dataset_stream = load_dataset(
        "izumi-lab/open-text-books",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    print("✓ Dataset initialized")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Initialize bins
bins = {i: [] for i in range(len(bin_targets))}

docs_processed = 0

print("Processing documents with uniform sampling...")

for doc in dataset_stream:
    docs_processed += 1

    if docs_processed % 500 == 0:
        filled_bins = sum(1 for i, b in bins.items() if len(b) >= bin_targets[i][2])
        total_collected = sum(len(b) for b in bins.values())
        print(f"  Processed: {docs_processed} | Collected: {total_collected}/{TARGET_DOCS}")

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
    if all_bins_full or docs_processed >= SAMPLE_SIZE:
        break

total_collected = sum(len(b) for b in bins.values())
print(f"\n✓ Processed: {docs_processed} | Collected: {total_collected}")

# Show bin statistics
print("\nBin statistics:")
for i, (start, end, target) in enumerate(bin_targets):
    count = len(bins[i])
    status = "✓" if count >= target else "⚠"
    print(f"  {status} {int(start)}-{int(end)}: {count}/{target}")

if total_collected == 0:
    print("\n⚠ No documents collected!")
    exit(1)

# Save documents with sequential numbering
print(f"\nSaving {total_collected} documents...")
doc_counter = 1

for i in range(len(bin_targets)):
    for doc in bins[i]:
        output_file = output_dir / f"textbook_en_{doc_counter}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(doc["text"])
        doc_counter += 1

print(f"\n{'=' * 60}")
print(f"✓ Saved {total_collected} documents to: {output_dir}")
print(f"  Files: textbook_en_1.txt to textbook_en_{total_collected}.txt")

