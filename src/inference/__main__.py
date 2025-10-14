"""CLI entry point for inference module."""

import argparse
import json
from pathlib import Path

from src.inference.predictor import SemanticChunker


def run_text(
    text: str,
    adapter_path: str | None,
):
    """
    Chunk raw text and display results.

    Args:
        text: Raw document text to chunk
        adapter_path: Path to LoRA adapter directory (None for base model)
    """
    print("=" * 80)
    print("Chunking text...")
    print("=" * 80)
    print(f"Adapter: {adapter_path or 'None (base model)'}")
    print(f"Input: {len(text.split())} words")
    print("-" * 80)

    chunker = SemanticChunker(adapter_path=adapter_path)
    chunks = chunker.chunk_document(
        text=text,
    )

    print(f"\n✓ Created {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        print(f"Chunk {i} ({word_count} words):")
        print("-" * 80)
        print(chunk)
        print()


def run_test_sample(
    sample_path: str,
    adapter_path: str | None,
):
    """
    Test on a sample with ground truth indices.

    Args:
        sample_path: Path to JSON file with 'input' and 'output' keys
        adapter_path: Path to LoRA adapter directory (None for base model)
    """
    # Load sample
    with open(sample_path, encoding="utf-8") as f:
        sample = json.load(f)

    if "input" not in sample or "output" not in sample:
        raise ValueError("Not a valid test sample")

    tagged_text = sample["input"]
    ground_truth = sample["output"]

    print("=" * 80)
    print(f"Testing sample: {Path(sample_path).name}")
    print("=" * 80)
    print(f"Adapter: {adapter_path or 'None (base model)'}")
    print(f"Ground truth: {ground_truth}")
    print("-" * 80)

    chunker = SemanticChunker(adapter_path=adapter_path)

    # Predict
    model_output = chunker.predict_split_locations(
        tagged_text=tagged_text,
    )
    indices = chunker.parse_indices(model_output)

    print(f"\nPredicted: {indices}")
    print(f"Raw output: {model_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with semantic chunking model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input: either test sample or text
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--sample",
        type=str,
        help="JSON file with 'input' and 'output' keys",
    )
    input_group.add_argument(
        "--text",
        type=str,
        help="Raw text string to chunk",
    )

    # Optional LoRA adapter
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (omit for base model)",
    )

    args = parser.parse_args()

    if args.sample:
        run_test_sample(
            sample_path=args.sample,
            adapter_path=args.adapter,
        )
    else:
        run_text(
            text=args.text,
            adapter_path=args.adapter,
        )


if __name__ == "__main__":
    main()
