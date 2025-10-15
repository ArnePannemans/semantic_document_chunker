"""CLI entry point for data pipeline."""

import argparse
import sys

from src.data_pipeline.labeling import generate_labels
from src.data_pipeline.prepare_training_data import create_training_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Data pipeline for semantic chunker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Label command
    label_parser = subparsers.add_parser(
        "generate_labels",
        help="Generate labels for documents with semantic split markers using Gemini",
    )
    label_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with raw documents (.txt files)",
    )
    label_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for labeled documents",
    )

    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare_training_pairs",
        help="Prepare training pairs from labeled documents",
    )
    prepare_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with labeled documents (.txt files)",
    )
    prepare_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for training pair JSON files",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate_labels":
        print(f"Generating labels for documents: {args.input} → {args.output}")
        generate_labels(args.input, args.output)
        print("✓ Labels generated")

    elif args.command == "prepare_training_pairs":
        print(f"Preparing training pairs from labeled documents: {args.input} → {args.output}")
        create_training_pairs(args.input, args.output)
        print("✓ Training pairs created")


if __name__ == "__main__":
    main()
