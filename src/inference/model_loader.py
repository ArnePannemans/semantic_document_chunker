"""Model and LoRA loading for inference."""

from peft import PeftModel
from unsloth import FastLanguageModel

from src.config import InferenceConfig


def load_model(
    adapter_path: str | None = None,
    config: InferenceConfig | None = None,
):
    """
    Load model for inference. If adapter_path is None, load the base model only.
    If adapter_path is provided, load the base model then load the LoRA adapter.

    Args:
        adapter_path: Path to LoRA adapter directory (None for base model)
        config: Inference configuration (uses default if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    if config is None:
        config = InferenceConfig()

    # Always load base model first
    print(f"Loading base model: {config.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_len,
        load_in_4bit=config.load_in_4bit,
    )

    # If adapter_path provided, load the LoRA adapter on top
    if adapter_path is not None:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    # Prepare for inference
    FastLanguageModel.for_inference(model)

    return model, tokenizer
