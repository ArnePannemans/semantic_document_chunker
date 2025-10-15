"""Training pair preparation from labeled documents."""

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from src.config import DataPipelineConfig
from src.core.chunking import split_into_sentences, tag_sentences


class TrainingPairError(Exception):
    """Exception raised when training pair creation fails."""

    pass


@dataclass
class TrainingPair:
    """A single training example with tagged input and split positions."""

    tagged_input: str
    split_positions: list[int]

    def to_dict(self) -> dict:
        return {
            "input": self.tagged_input,
            "output": str(self.split_positions),
        }


def create_training_pair(text: str, config: DataPipelineConfig) -> TrainingPair:
    """
    Convert document with split markers into training pair.

    Process:
    1. Split by markers to get document sections
    2. Convert each section to sentences using nltk.sent_tokenize, track where splits occurred
    3. Tag all sentences with location markers (<|loc_0|>, <|loc_1|>, etc.)

    Example:
        Input: "First. Second.<SPLIT>Third. Fourth."
        Output:
            tagged_input: "<|loc_0|>First.<|loc_1|>Second.<|loc_2|>Third.<|loc_3|>Fourth."
            split_positions: [2]

    Args:
        text: Document with <SPLIT> markers
        config: Pipeline config with chunking settings

    Returns:
        TrainingPair with tagged input and split positions

    Raises:
        TrainingPairError: If text is empty or processing fails
    """
    if not text.strip():
        raise TrainingPairError("Empty text provided")

    # Split document by split markers
    chunks = text.split(config.chunking.split_marker)
    all_sentences = []
    split_positions = []

    for chunk_idx, chunk in enumerate(chunks):
        # Record position where this split marker occurred
        if chunk_idx > 0:
            split_positions.append(len(all_sentences))

        # Add sentences from this section
        all_sentences.extend(split_into_sentences(chunk, config.chunking.min_words_per_sentence))

    # Tag all sentences with location markers
    tagged_input = tag_sentences(all_sentences, config.chunking)

    return TrainingPair(tagged_input, split_positions)


def create_training_pairs(
    input_dir: str,
    output_dir: str,
    config: DataPipelineConfig | None = None,
) -> None:
    """
    Process all .txt files with <SPLIT> markers and save as JSON training pairs.

    Args:
        input_dir: Directory containing .txt files with split markers
        output_dir: Directory to write training pair JSON files
        config: Data pipeline configuration (uses default if None)
    """
    if config is None:
        config = DataPipelineConfig()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    document_paths = list(input_path.rglob("*.txt"))
    if not document_paths:
        print(f"No .txt files found in {input_dir}")
        return

    for document_path in tqdm(document_paths, desc="Preparing training pairs"):
        try:
            text = document_path.read_text(encoding="utf-8")
            training_pair = create_training_pair(text, config)

            # Preserve filename and change extension to .json
            output_file = output_path / document_path.with_suffix(".json").name
            output_file.write_text(
                json.dumps(training_pair.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except TrainingPairError as error:
            print(f"Training pair creation failed: {document_path.name} - {error}")
        except Exception as error:
            print(f"Unexpected error: {document_path.name} - {error}")
