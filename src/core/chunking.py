"""Core chunking domain logic."""

import math

import nltk

from src.config import ChunkingConfig

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def get_chunk_bounds(word_count: int, config: ChunkingConfig) -> tuple[int, int]:
    """
    Calculate valid chunk range for a document.

    Args:
        word_count: Total words in document
        config: Chunking configuration

    Returns:
        Tuple of (min_chunks, max_chunks)
    """
    min_size, max_size = config.acceptable_range
    min_chunks = math.ceil(word_count / max_size) if max_size > 0 else 1
    max_chunks = math.floor(word_count / min_size) if min_size > 0 else word_count
    return min_chunks, max_chunks


def validate_document(text: str, min_words: int, max_words: int) -> bool:
    """
    Check if document meets word count requirements.

    Args:
        text: Document text to check
        min_words: Minimum acceptable word count
        max_words: Maximum acceptable word count

    Returns:
        True if text meets requirements, False otherwise
    """
    word_count = len(text.split())
    return min_words <= word_count <= max_words


def get_location_tag(index: int, config: ChunkingConfig) -> str:
    """
    Format location tag for given index.

    Args:
        index: The index number for the tag
        config: Chunking configuration

    Returns:
        Formatted location tag string (e.g., "<|loc_0|>")
    """
    return config.location_tag_format.format(index=index)


def tag_sentences(sentences: list[str], config: ChunkingConfig) -> str:
    """
    Tag sentences with location markers and join them into a single string.

    Args:
        sentences: List of sentences to tag
        config: Chunking configuration

    Returns:
        Tagged text with location markers (e.g., "<|loc_0|>Text...<|loc_1|>More...")
    """
    tagged_parts = []
    for i, sentence in enumerate(sentences):
        tag = get_location_tag(i, config)
        # Add space before sentence if not first and doesn't start with space
        if i > 0 and not sentence.startswith(" "):
            tagged_parts.append(f"{tag} {sentence}")
        else:
            tagged_parts.append(f"{tag}{sentence}")
    return "".join(tagged_parts)


def split_into_sentences(text: str, min_words: int = 0) -> list[str]:
    """
    Split text into sentences, optionally grouping small ones.

    This is the canonical sentence splitting logic used by both:
    - Data preparation (with min_words grouping)
    - Inference (with min_words grouping)

    Uses nltk.sent_tokenize to split into sentences, then groups consecutive
    sentences until min_words threshold is met. If final sentence is too small,
    merges it with the previous sentence.

    Args:
        text: Text to split into sentences
        min_words: Minimum words per sentence chunk (0 = no grouping)

    Returns:
        List of sentence chunks
    """
    if not (text := text.strip()):
        return []

    sentences = nltk.sent_tokenize(text)

    # If no minimum, return raw sentences
    if min_words <= 0:
        return sentences

    # Otherwise, group sentences to meet minimum
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
