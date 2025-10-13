from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data preparation"""

    # Filter
    min_words: int = 400
    max_words: int = 3000

    # Chunking
    target_chunk_size: int = 200
    acceptable_range: tuple[int, int] = (100, 300)
    split_marker: str = "<SPLIT>"

    # Training pairs
    location_tag_format: str = "<|loc_{index}|>"
    min_words_per_sentence_chunk: int = 10

    parallel_workers: int = 5
    model: str = "gemini-2.5-flash-preview-09-2025"
    thinking_budget: int = 3000
    temperature: float = 0.0
    max_output_tokens: int = 8000
