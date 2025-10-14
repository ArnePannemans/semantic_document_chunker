"""Shared utilities for training and inference."""

import torch
from unsloth import FastLanguageModel

from src.config import TrainingConfig
from src.prompts.prompting import (
    get_training_system_prompt,
    get_training_user_prompt,
)


def format_sample_as_chat_messages(
    input: str,
    output: str | None = None,
    config: TrainingConfig | None = None,
) -> list[dict[str, str]]:
    """
    Format a tagged document and optional labels as chat messages for the model.

    This function creates the conversation structure used for both training and inference.
    It builds a three-part conversation:
    1. System message: Instructions for semantic chunking task with chunk size constraints
    2. User message: The tagged document text to be chunked
    3. Assistant message (optional): The expected split indices for training

        Args:
        input: Tagged document text with location markers
        (e.g., "<|loc_0|>Text...<|loc_1|>More text...")
        output: Optional ground truth split indices (e.g., "[3, 7, 12]").
                If provided, includes assistant message for training.
                If None, used for inference (generation).
        config: Training configuration (uses default if None)

    Returns:
        List of chat message dicts in the format:
        [
            {"role": "system", "content": "...instructions..."},
            {"role": "user", "content": "...tagged document..."},
            {"role": "assistant", "content": "...split indices..."} (if output provided)
        ]
    """
    min_chunk_size, max_chunk_size = TrainingConfig.acceptable_range

    # Generate system prompt with task instructions and constraints
    system_prompt = get_training_system_prompt(
        target_chunk_size=TrainingConfig.target_chunk_size,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )

    # Generate user prompt with the tagged document
    user_prompt = get_training_user_prompt(doc_text=input)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Include assistant response if output is provided (for training)
    if output is not None:
        messages.append({"role": "assistant", "content": output})

    return messages


def load_model(model_path: str | None = None):
    """
    Load model for inference. If model_path is None, load the base model.
    If model_path is provided, load the base model and then load the LoRA adapter.

    Args:
        model_path: Path to LoRA adapter directory (None for base model)

    Returns:
        Tuple of (model, tokenizer)
    """
    config = TrainingConfig()

    # Always load base model first
    print(f"Loading base model: {config.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_len,
        load_in_4bit=config.load_in_4bit,
    )

    # If model_path provided, load the LoRA adapter on top
    if model_path is not None:
        print(f"Loading LoRA adapter from: {model_path}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path)

    # Prepare for inference
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def predict_splits(
    model,
    tokenizer,
    tagged_text: str,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> str:
    """
    Predict split locations for a tagged document.

    Args:
        model: Loaded model
        tokenizer: Model tokenizer
        tagged_text: Document text with location tags
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated prediction (list of split indices as string)
    """
    # Build chat messages
    messages = format_sample_as_chat_messages(
        input=tagged_text,
    )

    # Apply chat template for generation
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=0.05,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode only the generated part
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return prediction
