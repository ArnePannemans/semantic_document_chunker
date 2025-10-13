import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.config import DataConfig
from src.prompts.prompting import (
    get_split_marker_system_prompt,
    get_split_marker_user_prompt,
)
from src.utils.gemini import call_gemini


def get_chunk_bounds(word_count: int) -> tuple[int, int]:
    """
    Calculate the valid range of chunks a document should be split into.
    Uses the configured acceptable_range to determine how many semantic
    chunks a document should be allowed to be split into.

    Args:
        word_count: Total words in document

    Returns:
        Tuple of (min_chunks, max_chunks)
    """
    min_size, max_size = DataConfig.acceptable_range
    min_chunks = math.ceil(word_count / max_size) if max_size > 0 else 1
    max_chunks = math.floor(word_count / min_size) if min_size > 0 else word_count
    return min_chunks, max_chunks


def is_valid(text: str) -> bool:
    """
    Check if text meets word count requirements according to the configured
    min_words and max_words.

    Args:
        text: Document text to check

    Returns:
        True if text meets word count requirements, False otherwise
    """
    word_count = len(text.split())
    return DataConfig.min_words <= word_count <= DataConfig.max_words


def process_document(text: str) -> str:
    """
    Process document and insert split markers using Gemini.

    Args:
        text: Document text to process

    Returns:
        Processed text with split markers inserted

    Raises:
        ValueError: If text doesn't meet word count requirements
        Exception: If LLM call fails or returns empty response
    """
    if not is_valid(text):
        word_count = len(text.split())
        raise ValueError(
            f"Word count {word_count} outside acceptable range "
            f"({DataConfig.min_words}-{DataConfig.max_words})"
        )

    word_count = len(text.split())
    min_chunks, max_chunks = get_chunk_bounds(word_count)
    min_size, max_size = DataConfig.acceptable_range

    system_prompt = get_split_marker_system_prompt(
        split_marker=DataConfig.split_marker,
        target_chunk_size=DataConfig.target_chunk_size,
        min_chunk_size=min_size,
        max_chunk_size=max_size,
    )

    user_prompt = get_split_marker_user_prompt(
        doc_word_count=word_count,
        min_expected_splits=min_chunks,
        max_expected_splits=max_chunks,
        document=text,
    )

    result = call_gemini(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=DataConfig.model,
        temperature=DataConfig.temperature,
        max_output_tokens=DataConfig.max_output_tokens,
        thinking_budget=DataConfig.thinking_budget,
    )

    if not result or not result.strip():
        raise Exception("Empty response from LLM")

    return result.strip()


def insert_split_markers(input_dir: str, output_dir: str):
    """
    Process all documents in input directory and save with split markers.

    Args:
        input_dir: Directory containing .txt files
        output_dir: Directory to write processed files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_path.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    # Use ThreadPoolExecutor to make parallel LLM calls
    with ThreadPoolExecutor(max_workers=DataConfig.parallel_workers) as executor:
        futures = {
            executor.submit(
                process_document, file_path.read_text(encoding="utf-8")
            ): file_path
            for file_path in txt_files
        }

        for future in tqdm(
            as_completed(futures),
            total=len(txt_files),
            desc="Creating labeled documents",
        ):
            file_path = futures[future]
            try:
                # Write labeled document to output directory
                result = future.result()
                output_file = output_path / file_path.name
                output_file.write_text(result, encoding="utf-8")
            except Exception as error:
                print(f"Failed: {file_path.name} - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Insert semantic split markers using Gemini"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory with raw documents"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for labeled documents",
    )

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    print(f"Inserting split markers into {input_dir} and saving to {output_dir}")

    insert_split_markers(input_dir, output_dir)


if __name__ == "__main__":
    main()
