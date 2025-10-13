import argparse
import json
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm


def load_dataset_examples(data_dir: Path) -> list[dict]:
    """
    Load all dataset examples by matching files across directories.

    Reads files from documents/, labeled/, and training_pairs/ subdirectories,
    matches them by filename (without extension), and creates dataset examples.

    Args:
        data_dir: Root directory containing documents/, labeled/, training_pairs/

    Returns:
        List of example dictionaries with id, raw_text, labeled_text,
        tagged_input, and split_indices
    """
    documents_dir = data_dir / "documents"
    labeled_dir = data_dir / "labeled"
    training_pairs_dir = data_dir / "training_pairs"

    # Get all document files and use their stems as IDs
    doc_files = list(documents_dir.glob("*.txt"))
    examples = []

    for doc_file in tqdm(doc_files, desc="Loading dataset examples"):
        file_id = doc_file.stem

        # Construct paths for matching files
        labeled_file = labeled_dir / f"{file_id}.txt"
        training_pair_file = training_pairs_dir / f"{file_id}.json"

        # Skip if not all three files exist
        if not labeled_file.exists() or not training_pair_file.exists():
            print(f"Warning: Skipping {file_id} - missing matching files")
            continue

        # Read raw text
        raw_text = doc_file.read_text(encoding="utf-8")

        # Read labeled text
        labeled_text = labeled_file.read_text(encoding="utf-8")

        # Read training pair JSON
        training_pair = json.loads(training_pair_file.read_text(encoding="utf-8"))

        # Create example
        example = {
            "id": file_id,
            "raw_text": raw_text,
            "labeled_text": labeled_text,
            "tagged_input": training_pair["input"],
            "split_indices": json.loads(training_pair["output"]),
        }

        examples.append(example)

    return examples


def create_dataset_card() -> str:
    """
    Generate dataset card content for Hugging Face Hub.

    Reads the system prompt from the prompts directory and includes it
    in the dataset card along with usage examples and citation info.

    Returns:
        Markdown string for README.md
    """
    # Read system prompt
    prompt_path = Path("src/prompts/insert_split_marker/system_prompt.jinja2")
    system_prompt = prompt_path.read_text(encoding="utf-8")

    card = f"""---
language:
- en
- fr
- nl
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- rag
- chunking
- semantic-segmentation
- document-processing
pretty_name: Semantic Document Chunker Dataset
---

# Semantic Document Chunker Dataset

## Dataset Description

This dataset contains 639 documents annotated with semantic boundaries
by **Gemini 2.5 Pro** for training document chunking models for
Retrieval-Augmented Generation (RAG). Each document includes the raw text,
text with semantic split markers inserted by the LLM, and training pairs
with location tags.

## Dataset Structure

Each example contains:
- `id`: Unique document identifier
- `raw_text`: Original document text
- `labeled_text`: Document with `<SPLIT>` markers indicating semantic boundaries
- `tagged_input`: Text with `<|loc_N|>` location tags for each sentence chunk
- `split_indices`: List of integers indicating where original splits occurred

## Data Collection

Documents were labeled using **Gemini 2.5 Pro** with the following
semantic boundary detection prompt:

```jinja2
{system_prompt}
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("ArnePannemans/semantic-document-chunker")

# Access training pairs
for example in ds:
    print(f"ID: {{example['id']}}")
    print(f"Tagged input: {{example['tagged_input'][:100]}}...")
    print(f"Split indices: {{example['split_indices']}}")
```

## Citation

If you use this dataset, please cite the repository:

```bibtex
@misc{{pannemans2025semantic,
  author = {{Pannemans, Arne}},
  title = {{Semantic Document Chunker}},
  year = {{2025}},
  url = {{https://github.com/ArnePannemans/semantic_document_chunker}}
}}
```
"""

    return card


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and upload dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/v1",
        help="Directory containing documents/, labeled/, training_pairs/",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., username/dataset-name)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load all dataset examples
    print(f"Loading dataset from {data_dir}...")
    examples = load_dataset_examples(data_dir)

    if not examples:
        print("Error: No examples found!")
        return

    print(f"Loaded {len(examples)} examples")

    # Create Dataset
    print("Creating Hugging Face Dataset...")
    dataset = Dataset.from_dict(
        {
            "id": [ex["id"] for ex in examples],
            "raw_text": [ex["raw_text"] for ex in examples],
            "labeled_text": [ex["labeled_text"] for ex in examples],
            "tagged_input": [ex["tagged_input"] for ex in examples],
            "split_indices": [ex["split_indices"] for ex in examples],
        }
    )

    # Generate dataset card
    print("Generating dataset card...")
    card_content = create_dataset_card()

    # Save card locally for review
    card_path = Path("data/DATASET_CARD.md")
    card_path.write_text(card_content, encoding="utf-8")
    print(f"Dataset card saved to {card_path}")

    # Push to Hub
    print(f"Pushing dataset to {args.repo_id}...")
    dataset.push_to_hub(args.repo_id)

    print(
        f"\n✅ Success! Dataset uploaded to https://huggingface.co/datasets/{args.repo_id}"
    )


if __name__ == "__main__":
    main()
