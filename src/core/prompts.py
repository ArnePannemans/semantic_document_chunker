"""Unified prompt rendering for all use cases."""

from jinja2 import Environment, PackageLoader

from src.config import ChunkingConfig

# Initialize template loaders
labeling_env = Environment(
    loader=PackageLoader("src.prompt_templates", "labeling"),
    autoescape=False,
)
prediction_env = Environment(
    loader=PackageLoader("src.prompt_templates", "prediction"),
    autoescape=False,
)


def render_labeling_prompts(
    document: str,
    word_count: int,
    min_chunks: int,
    max_chunks: int,
    config: ChunkingConfig,
) -> tuple[str, str]:
    """
    Render system and user prompts for Gemini labeling.

    Args:
        document: The document text to label
        word_count: Word count of the document
        min_chunks: Minimum expected number of chunks
        max_chunks: Maximum expected number of chunks
        config: Chunking configuration

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    min_size, max_size = config.acceptable_range

    system = labeling_env.get_template("system.jinja2").render(
        split_marker=config.split_marker,
        target_chunk_size=config.target_chunk_size,
        min_chunk_size=min_size,
        max_chunk_size=max_size,
    )

    user = labeling_env.get_template("user.jinja2").render(
        doc_word_count=word_count,
        min_expected_splits=min_chunks,
        max_expected_splits=max_chunks,
        document=document,
    )

    return system, user


def render_prediction_prompts(
    document: str,
    config: ChunkingConfig,
) -> tuple[str, str]:
    """
    Render system and user prompts for model inference.

    Args:
        document: The tagged document text
        config: Chunking configuration

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    min_size, max_size = config.acceptable_range

    system = prediction_env.get_template("system.jinja2").render(
        target_chunk_size=config.target_chunk_size,
        min_chunk_size=min_size,
        max_chunk_size=max_size,
    )

    user = prediction_env.get_template("user.jinja2").render(
        doc_text=document,
    )

    return system, user


def format_as_chat_messages(
    system_prompt: str,
    user_prompt: str,
    assistant_response: str | None = None,
) -> list[dict[str, str]]:
    """
    Format prompts as chat messages

    Args:
        system_prompt: The system instruction
        user_prompt: The user message
        assistant_response: Optional assistant response (for training)

    Returns:
        List of chat message dicts
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_response:
        messages.append({"role": "assistant", "content": assistant_response})
    return messages
