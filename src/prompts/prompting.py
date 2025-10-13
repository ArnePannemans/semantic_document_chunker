"""The prompting file provides utilities for generating prompts."""

from jinja2 import Environment, PackageLoader, select_autoescape

prompt_client = Environment(
    loader=PackageLoader("src.prompts", "insert_split_marker"),
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
    template = prompt_client.get_template("system_prompt.jinja2")
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
    template = prompt_client.get_template("user_prompt.jinja2")
    return template.render(
        doc_word_count=doc_word_count,
        min_expected_splits=min_expected_splits,
        max_expected_splits=max_expected_splits,
        document=document,
    )
