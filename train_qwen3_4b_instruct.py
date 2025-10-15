"""Training script for Qwen3-4B-Instruct following Unsloth reference exactly."""

import json
import os
import sys
from pathlib import Path

# Fix for Cursor AppImage issue with wandb
if "cursor" in sys.executable.lower() or ".appimage" in sys.executable.lower():
    import shutil

    python_path = shutil.which("python3") or shutil.which("python")
    if python_path:
        sys.executable = python_path

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import wandb


def load_training_data(input_dir: str) -> list[dict]:
    """Load all JSON training pairs from input directory."""
    input_path = Path(input_dir)
    json_files = list(input_path.rglob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    samples = []
    for file_path in json_files:
        with open(file_path, encoding="utf-8") as f:
            samples.append(json.load(f))

    print(f"✓ Loaded {len(samples)} training samples from {input_dir}")
    return samples


def convert_to_conversations(samples: list[dict]) -> list[list[dict]]:
    """
    Convert training samples to conversation format.

    Each sample has 'input' (tagged document) and 'output' (split indices).
    We convert to ChatML format with system, user, assistant messages.
    """
    from src.config import ChunkingConfig
    from src.core.prompts import render_prediction_prompts

    config = ChunkingConfig()
    conversations = []

    for sample in samples:
        system_prompt, user_prompt = render_prediction_prompts(
            document=sample["input"],
            config=config,
        )

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": sample["output"]},
        ]
        conversations.append(conversation)

    return conversations


def main():
    """Main training function."""

    # Configuration (matching Unsloth reference for 4B-Instruct)
    INPUT_DIR = "data/v1/training_pairs"
    OUTPUT_DIR = "models/qwen3-4b-instruct-v2"

    BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
    MAX_SEQ_LENGTH = 8000
    LOAD_IN_4BIT = True  # Use 4-bit for memory efficiency (your original config)

    # LoRA settings (matching your original config exactly)
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.0

    # Training settings (matching your original config exactly)
    BATCH_SIZE = 4  # Your batch_size_train
    GRADIENT_ACCUMULATION = 2  # Your gradient_accumulation_steps
    NUM_EPOCHS = 2
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.1  # Your original warmup (10% of training)

    # Splits
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10
    SEED = 3407

    print("=" * 80)
    print("Training Qwen3-4B-Instruct for Semantic Chunking")
    print("=" * 80)

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading training data...")
    samples = load_training_data(INPUT_DIR)

    # Split data
    print("\n2. Splitting data...")
    train_val_samples, test_samples = train_test_split(
        samples, test_size=TEST_RATIO, random_state=SEED
    )
    val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)
    train_samples, val_samples = train_test_split(
        train_val_samples, test_size=val_ratio_adjusted, random_state=SEED
    )
    print(f"   Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Save test samples
    with open(output_path / "test_samples.json", "w") as f:
        json.dump({"test_samples": test_samples}, f, indent=2)

    # Load model
    print("\n3. Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Configure chat template (CRITICAL for Qwen3-Instruct!)
    # Use the official qwen3-instruct template (without <think> tags for non-reasoning)
    print("   Configuring qwen3-instruct chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-instruct",  # Official template from Unsloth
    )

    # Apply LoRA
    print("\n4. Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized mode
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    # Convert data to conversations
    print("\n5. Converting data to conversation format...")
    train_conversations = convert_to_conversations(train_samples)
    val_conversations = convert_to_conversations(val_samples)

    # Apply chat template to create text dataset
    print("   Applying chat template...")
    train_texts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,  # No thinking for our task
        )
        for conv in train_conversations
    ]
    val_texts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        for conv in val_conversations
    ]

    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    print(f"   Train dataset: {len(train_dataset)} samples")
    print(f"   Val dataset: {len(val_dataset)} samples")

    # Show example
    print("\n6. Example formatted training sample:")
    print("-" * 80)
    print(train_texts[0][:500] + "..." if len(train_texts[0]) > 500 else train_texts[0])
    print("-" * 80)

    # Initialize WandB
    print("\n7. Initializing Weights & Biases...")
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_LOG_MODEL"] = "end"

    wandb.init(
        project="Semantic chunking",
        name=f"qwen3-4b-instruct-{Path(OUTPUT_DIR).name}",
        group="Qwen3-4B-Instruct",
        config={
            "base_model": BASE_MODEL,
            "max_seq_len": MAX_SEQ_LENGTH,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
    )

    # Create trainer
    print("\n8. Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_ratio=WARMUP_RATIO,  # Using ratio instead of steps
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=7,
            eval_strategy="steps",
            eval_steps=7,
            save_strategy="steps",
            save_steps=7,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_8bit",  # Your original optimizer
            weight_decay=0.01,  # Your original weight_decay
            lr_scheduler_type="cosine",  # Your original scheduler
            seed=SEED,
            output_dir=str(output_path),
            report_to="wandb",
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,
        ),
    )

    # Show memory stats
    print("\n9. Memory stats:")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"   GPU: {gpu_stats.name}")
    print(f"   Max memory: {max_memory} GB")
    print(f"   Reserved: {start_gpu_memory} GB")

    # Train!
    print("\n10. Starting training...")
    print("=" * 80)
    trainer_stats = trainer.train()
    print("=" * 80)
    print("✓ Training completed!")

    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print("\n11. Final stats:")
    print(f"   Training time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes")
    print(f"   Peak memory: {used_memory} GB")
    print(f"   Memory for LoRA: {used_memory_for_lora} GB")

    # Save model
    print(f"\n12. Saving model to {output_path / 'final_model'}...")
    model.save_pretrained(str(output_path / "final_model"))
    tokenizer.save_pretrained(str(output_path / "final_model"))

    wandb.finish()
    print("\n" + "=" * 80)
    print("✓ Training finished successfully!")
    print(f"✓ Model saved to: {output_path / 'final_model'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
