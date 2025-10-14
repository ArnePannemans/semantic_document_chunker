"""The prompting file provides utilities for generating prompts."""

from jinja2 import Environment, PackageLoader, select_autoescape

split_marker_prompt_client = Environment(
    loader=PackageLoader("src.prompts", "insert_split_marker"),
    autoescape=select_autoescape(),
)

training_prompt_client = Environment(
    loader=PackageLoader("src.prompts", "training"),
    autoescape=select_autoescape(),
)


def get_split_marker_system_prompt(
    *,
    split_marker: str,
    target_chunk_size: int,
    min_chunk_size: int,
    max_chunk_size: int,
) -> str:
    """Get the system prompt for inserting split markers.

    Args:
        split_marker: The marker string to insert (e.g., "<SPLIT>")
        target_chunk_size: Target chunk size in words
        min_chunk_size: Minimum acceptable chunk size in words
        max_chunk_size: Maximum acceptable chunk size in words

    Returns:
        Rendered system prompt
    """
    template = split_marker_prompt_client.get_template("system_prompt.jinja2")
    return template.render(
        split_marker=split_marker,
        target_chunk_size=target_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )


def get_split_marker_user_prompt(
    *,
    doc_word_count: int,
    min_expected_splits: int,
    max_expected_splits: int,
    document: str,
) -> str:
    """Get the user prompt for inserting split markers.

    Args:
        doc_word_count: Word count of the document
        min_expected_splits: Minimum expected number of splits
        max_expected_splits: Maximum expected number of splits
        document: The document text to process

    Returns:
        Rendered user prompt
    """
    template = split_marker_prompt_client.get_template("user_prompt.jinja2")
    return template.render(
        doc_word_count=doc_word_count,
        min_expected_splits=min_expected_splits,
        max_expected_splits=max_expected_splits,
        document=document,
    )


def get_training_system_prompt(
    *,
    target_chunk_size: int,
    min_chunk_size: int,
    max_chunk_size: int,
) -> str:
    """Get the system prompt for training.

    Args:
        target_chunk_size: Target chunk size in words
        min_chunk_size: Minimum acceptable chunk size in words
        max_chunk_size: Maximum acceptable chunk size in words

    Returns:
        Rendered system prompt
    """
    template = training_prompt_client.get_template("system_prompt.jinja2")
    return template.render(
        target_chunk_size=target_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )


def get_training_user_prompt(
    *,
    doc_text: str,
) -> str:
    """Get the user prompt for training.

    Args:
        doc_text: The tagged document text to process

    Returns:
        Rendered user prompt
    """
    template = training_prompt_client.get_template("user_prompt.jinja2")
    return template.render(doc_text=doc_text)
