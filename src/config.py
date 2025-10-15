"""Configuration classes for semantic chunker."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChunkingConfig:
    """Chunking strategy - shared across data pipeline and training."""

    target_chunk_size: int = 200
    acceptable_range: tuple[int, int] = (100, 300)
    split_marker: str = "<SPLIT>"
    location_tag_format: str = "<|loc_{index}|>"
    min_words_per_sentence: int = 10  # Minimum words when grouping sentences


@dataclass
class DataPipelineConfig:
    """Configuration for data preparation pipeline."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Document filtering
    min_words: int = 400
    max_words: int = 3000

    # Gemini labeling
    gemini_model: str = "gemini-2.5-flash-preview-09-2025"
    gemini_temperature: float = 0.0
    gemini_max_tokens: int = 10000
    gemini_thinking_budget: int = 4000
    parallel_workers: int = 5


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Model
    base_model: str = "unsloth/Qwen3-14B"
    load_in_4bit: bool = True
    max_seq_len: int = 8000

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: bool = True

    # Training hyperparameters
    batch_size_train: int = 4
    batch_size_eval: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_8bit"

    # Checkpointing
    logging_steps: int = 7
    eval_steps: int = 7
    save_steps: int = 7
    save_total_limit: int = 2

    # Data splits
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 3407

    # WandB
    wandb_project: str = "Semantic chunking"
    wandb_group: str = "Qwen3-14B"
    wandb_run_name: str = field(default_factory=lambda: f"qwen3-14b-{datetime.now():%H%M%S}")


@dataclass
class InferenceConfig:
    """Configuration for model inference (used by both API and CLI)."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Model settings
    base_model: str = "unsloth/Qwen3-14B"
    adapter_path: str | None = "models/qwen3-14-recreation-run-v1/checkpoint-133"
    max_seq_len: int = 8000
    load_in_4bit: bool = True

    # Generation settings
    temperature: float = 0.05
    max_new_tokens: int = 256


@dataclass
class APIConfig:
    """Configuration for API server."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"

    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    inference: InferenceConfig = field(default_factory=InferenceConfig)
