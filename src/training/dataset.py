"""Dataset creation for training."""

from datasets import Dataset

from src.config import TrainingConfig
from src.core.prompts import format_as_chat_messages, render_prediction_prompts


def training_pair_to_chat_messages(
    training_pair: dict, config: TrainingConfig
) -> list[dict[str, str]]:
    """
    Convert a training sample to chat messages.

    Args:
        training_pair: Training pair with 'input' (tagged text) and 'output' (split indices)
        config: Training configuration

    Returns:
        List of chat message dicts [{"role": "system", "content": "..."}, ...]
    """
    system_prompt, user_prompt = render_prediction_prompts(
        document=training_pair["input"],
        config=config.chunking,
    )

    return format_as_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        assistant_response=training_pair["output"],
    )


def apply_chat_template(
    chat_messages: list[dict[str, str]],
    tokenizer,
) -> str:
    """
    Apply tokenizer's chat template to format messages as a string.

    Args:
        chat_messages: List of message dicts [{"role": ..., "content": ...}, ...]
        tokenizer: Model tokenizer

    Returns:
        Formatted text string ready for training
    """
    return tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=False,  # Training: we have the response
        enable_thinking=False,
    )


def create_training_dataset(
    training_pairs: list[dict],
    tokenizer,
    config: TrainingConfig,
) -> Dataset:
    """
    Convert training samples to HuggingFace Dataset ready for training.

    Process:
    1. For each sample, build user and system prompts
    2. Format as chat messages
    3. Apply Qwen3 tokenizer's chat template to format as text strings
    4. Create Dataset with 'text' field (required by SFTTrainer)

    Args:
        training_pairs: List of training pairs with 'input' and 'output' keys
        tokenizer: Model tokenizer
        config: Training configuration

    Returns:
        HuggingFace Dataset with 'text' field containing formatted conversations
    """
    formatted_texts = []

    for training_pair in training_pairs:
        # Step 1: built user and system prompts
        system_prompt, user_prompt = render_prediction_prompts(
            document=training_pair["input"],
            config=config.chunking,
        )

        # Step 2: format as chat messages
        chat_messages = format_as_chat_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_response=training_pair["output"],
        )

        # Step 3: apply Qwen3 tokenizer template to format as string
        formatted_text = apply_chat_template(chat_messages, tokenizer)

        formatted_texts.append({"text": formatted_text})

    # Step 4: create HuggingFace Dataset
    return Dataset.from_list(formatted_texts)
