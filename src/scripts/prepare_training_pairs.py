import argparse
import json
from pathlib import Path

import nltk
from tqdm import tqdm

from src.config import DataConfig

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def get_location_tag(index: int) -> str:
    return DataConfig.location_tag_format.format(index=index)


def split_into_sentences(text: str, min_words: int) -> list[str]:
    """
    Split text into sentence meeting minimum word count.

    Uses nltk.sent_tokenize to split into sentences, then groups consecutive
    sentences until min_words threshold is met. If final sentence is too small,
    merges it with the previous sentence.

    Args:
        text: Text to split into sentences
        min_words: Minimum words per sentence

    Returns:
        List of sentences
    """
    if not (text := text.strip()):
        return []

    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        num_words = len(sentence.split())

        # If we've reached min_words, finalize current chunk
        if word_count >= min_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0

        current_chunk.append(sentence)
        word_count += num_words

    # Handle final chunk
    if current_chunk:
        if word_count < min_words and chunks:
            # Merge with previous chunk if too small
            chunks[-1] = f"{chunks[-1]} {' '.join(current_chunk)}"
        else:
            chunks.append(" ".join(current_chunk))

    return chunks


def create_training_pair(text: str) -> tuple[str, str]:
    """
    Process document with split markers into training pair format.

    Algorithm:
    1. Split document into sections by split markers
    2. Split each section into sentences using nltk.sent_tokenize
    3. Tag each sentence with location markers (<|loc_0|>, <|loc_1|>, etc.)
    4. Record indices where original splits occurred

    Example:
        Input text:
            "First sentence. Second sentence.<SPLIT>Third sentence.
            Fourth sentence.<SPLIT>Fifth sentence."

        After sentence chunking:
            Section 1 → ["First sentence.", "Second sentence."]     (sentences 0, 1)
            Section 2 → ["Third sentence.", "Fourth sentence."]     (sentence 2, 3)
            Section 3 → ["Fifth sentence."]                         (sentence 4)

        Output tagged_input:
            "<|loc_0|>First sentence.<|loc_1|>Second sentence.<|loc_2|>Third
            sentence.<|loc_3|>Fourth sentence.<|loc_4|>Fifth sentence."

        Output split_indices:
            [2, 4]  # Original positions of split markers

    Args:
        text: Document text containing split markers

    Returns:
        Tuple of (tagged_input, split_indices)

    Raises:
        Exception: If text is empty or processing fails
    """
    if not text.strip():
        raise Exception("Empty text provided")

    # Split document by split markers
    sections = text.split(DataConfig.split_marker)

    sentences = []
    split_indices = []

    for i, section in enumerate(sections):
        # Record split index where original split occurred
        if i > 0:
            split_indices.append(len(sentences))

        # Split section into sentences and add to full list
        sentences.extend(
            split_into_sentences(section, DataConfig.min_words_per_sentence_chunk)
        )

    # Build tagged input by adding location tags to each sentence
    tagged_sentences = []
    for i, sentence in enumerate(sentences):
        location_tag = get_location_tag(i)
        # Add spacing if needed (chunk doesn't start with space and it's not first)
        if i > 0 and not sentence.startswith(" "):
            tagged_sentences.append(f"{location_tag} {sentence}")
        else:
            tagged_sentences.append(f"{location_tag}{sentence}")

    tagged_input = "".join(tagged_sentences)

    return tagged_input, split_indices


def prepare_training_pairs(input_dir: str, output_dir: str):
    """
    Process all documents in input directory and save as training pairs.

    Args:
        input_dir: Directory containing .txt files with split markers
        output_dir: Directory to write training pair JSON files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_path.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    for file_path in tqdm(txt_files, desc="Preparing training pairs"):
        try:
            text = file_path.read_text(encoding="utf-8")

            # Create training pair
            tagged_input, split_indices = create_training_pair(text)
            training_pair = {"input": tagged_input, "output": str(split_indices)}

            # Preserve filename, change extension to .json
            output_file = output_path / file_path.with_suffix(".json").name

            output_file.write_text(
                json.dumps(training_pair, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as error:
            print(f"Failed: {file_path.name} - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training pairs from split-marked documents"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with split-marked documents",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for training pair JSON files",
    )

    args = parser.parse_args()

    print(f"Preparing training pairs from {args.input} and saving to {args.output}")

    prepare_training_pairs(args.input, args.output)


if __name__ == "__main__":
    main()
