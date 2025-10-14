from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DataConfig:
    """Configuration for data preparation"""

    # Filter documents
    min_words: int = 400
    max_words: int = 3000

    # Insert split markers
    target_chunk_size: int = 200
    acceptable_range: tuple[int, int] = (100, 300)
    split_marker: str = "<SPLIT>"

    parallel_workers: int = 5
    model: str = "gemini-2.5-flash-preview-09-2025"
    thinking_budget: int = 3000
    temperature: float = 0.0
    max_output_tokens: int = 8000

    # Create training pairs
    location_tag_format: str = "<|loc_{index}|>"
    min_words_per_sentence_chunk: int = 10


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    base_model: str = "unsloth/Qwen3-14B"
    max_seq_len: int = 8000
    load_in_4bit: bool = True

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=list)
    use_gradient_checkpointing: bool = True

    def __post_init__(self):
        """Initialize mutable default values"""
        if not self.lora_target_modules:
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    # Training hyperparameters
    batch_size_train: int = 3
    batch_size_eval: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_8bit"

    # Logging and checkpointing
    logging_steps: int = 7
    eval_steps: int = 7
    save_steps: int = 7
    save_total_limit: int = 2

    # Data split ratios
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # WandB settings
    wandb_project: str = "Semantic chunking"
    wandb_group: str = "Qwen3-14B"
    wandb_run_name: str = field(default_factory=lambda: f"qwen3-14b-run-{datetime.now():%H-%M-%S}")

    # Chunk size parameters for prompts
    target_chunk_size: int = 200
    acceptable_range: tuple[int, int] = (100, 300)

    seed: int = 3407
