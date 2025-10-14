#!/usr/bin/env python3
"""
CLI for running inference on semantic chunking model.

Examples:
    # Test base model (untrained)
    python -m src.scripts.inference --sample data/dummy/training_pairs/dummy_doc.json

    # Test trained model with LoRA
    python -m src.scripts.inference --sample data/dummy/training_pairs/dummy_doc.json
     --model models/my_model

    # Test on custom tagged text
    python -m src.scripts.inference --text "<|loc_0|>First sentence.
    <|loc_1|>Second sentence."
"""

import argparse
import json

from src.utils.training_utils import load_model, predict_splits


def print_result(input_text: str, expected: str | None, predicted: str, show_input: bool):
    """Pretty print inference result."""
    print("\n" + "=" * 80)

    if show_input:
        if len(input_text) > 200:
            print(f"INPUT (truncated):\n{input_text[:200]}...\n")
        else:
            print(f"INPUT:\n{input_text}\n")

    if expected:
        print(f"EXPECTED OUTPUT:\n{expected}\n")

    print(f"PREDICTED OUTPUT:\n{predicted}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on semantic chunking model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sample",
        type=str,
        help="Path to JSON training sample file",
    )
    input_group.add_argument(
        "--text",
        type=str,
        help="Tagged text to process directly",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model/LoRA adapter (omit to use base model)",
    )

    # Generation options
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )

    # Display options
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="Don't display input text in output",
    )

    args = parser.parse_args()

    # Display settings
    print(f"\n{'Base model' if args.model is None else 'Trained model'} inference")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}\n")

    # Load model
    model, tokenizer = load_model(model_path=args.model)

    # Run inference
    if args.sample:
        # Load from sample file
        print(f"Loading sample from: {args.sample}")
        with open(args.sample, encoding="utf-8") as f:
            sample = json.load(f)

        prediction = predict_splits(
            model=model,
            tokenizer=tokenizer,
            tagged_text=sample["input"],
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )

        print_result(
            input_text=sample["input"],
            expected=sample["output"],
            predicted=prediction,
            show_input=not args.no_input,
        )
    else:
        # Use provided text
        prediction = predict_splits(
            model=model,
            tokenizer=tokenizer,
            tagged_text=args.text,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
        )

        print_result(
            input_text=args.text,
            expected=None,
            predicted=prediction,
            show_input=not args.no_input,
        )


if __name__ == "__main__":
    main()
