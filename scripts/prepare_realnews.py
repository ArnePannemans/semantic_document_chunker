from pathlib import Path

from datasets import load_dataset

MIN_WORDS = 500
MAX_WORDS = 5000

# Bin targets: (start, end, target_docs)
# Lower targets for longer documents as they're rarer
BIN_TARGETS = [
    (500, 1000, 50),
    (1000, 1500, 50),
    (1500, 2000, 50),
    (2000, 2500, 40),
    (2500, 3000, 30),
    (3000, 3500, 30),
    (3500, 4000, 20),
    (4000, 4500, 15),
    (4500, 5000, 15),
]

SAMPLE_SIZE = 15000

output_dir = Path(__file__).parent.parent / "data" / "v2" / "documents"
output_dir.mkdir(parents=True, exist_ok=True)

total_target = sum(target for _, _, target in BIN_TARGETS)

print("RealNews Dataset Preparation")
print(f"Word count range: {MIN_WORDS}-{MAX_WORDS} words")
print(f"Target documents: {total_target} (uniform sampling)")
print("-" * 60)

print("\nLoading dataset...")

try:
    dataset_stream = load_dataset(
        "Hieuman/realnews",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    print("✓ Dataset initialized")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Initialize bins
bins = {i: [] for i in range(len(BIN_TARGETS))}

print("Processing documents with uniform sampling...")
docs_processed = 0

for doc in dataset_stream:
    docs_processed += 1

    if docs_processed % 1000 == 0:
        filled_bins = sum(1 for i, b in bins.items() if len(b) >= BIN_TARGETS[i][2])
        total_collected = sum(len(b) for b in bins.values())
        print(f"  Processed: {docs_processed} | Collected: {total_collected}/{total_target}")

    text = doc["text"]
    word_count = len(text.split())

    # Assign to appropriate bin
    if MIN_WORDS <= word_count <= MAX_WORDS:
        for i, (bin_start, bin_end, target) in enumerate(BIN_TARGETS):
            if bin_start <= word_count < bin_end:
                if len(bins[i]) < target:
                    bins[i].append({"text": text, "word_count": word_count})
                break

    # Stop if all bins are full or reached sample size
    all_bins_full = all(len(bins[i]) >= BIN_TARGETS[i][2] for i in range(len(BIN_TARGETS)))
    if all_bins_full or docs_processed >= SAMPLE_SIZE:
        break

total_collected = sum(len(b) for b in bins.values())
print(f"\n✓ Processed: {docs_processed} | Collected: {total_collected}")

# Show bin statistics
print("\nBin statistics:")
for i, (start, end, target) in enumerate(BIN_TARGETS):
    count = len(bins[i])
    status = "✓" if count >= target else "⚠"
    print(f"  {status} {int(start)}-{int(end)}: {count}/{target}")

if total_collected == 0:
    print("\n⚠ No documents collected!")
    exit(1)

# Save documents with sequential numbering
print(f"\nSaving {total_collected} documents...")
doc_counter = 1

for i in range(len(BIN_TARGETS)):
    for doc in bins[i]:
        output_file = output_dir / f"realnews_en_{doc_counter}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(doc["text"])
        doc_counter += 1

print(f"\n✓ Saved {total_collected} documents to: {output_dir}")
print(f"  Files: realnews_en_1.txt to realnews_en_{total_collected}.txt")

