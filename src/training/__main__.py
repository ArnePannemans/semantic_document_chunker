"""CLI entry point for training module."""

import argparse

from src.training.train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train semantic chunking model with Qwen3-14B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing training pair JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for model checkpoints and results",
    )

    args = parser.parse_args()

    print(f"Training: {args.input} → {args.output}")
    train_model(args.input, args.output)


if __name__ == "__main__":
    main()
