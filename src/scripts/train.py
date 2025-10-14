#!/usr/bin/env python3
"""
Training script for semantic chunk prediction using Qwen3-14B
"""

import argparse
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

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

import wandb
from src.config import TrainingConfig
from src.utils.training_utils import format_sample_as_chat_messages


def load_training_data(input_dir: str) -> list[dict]:
    """
    Load all JSON training pairs from input directory.

    Args:
        input_dir: Directory containing JSON training pair files

    Returns:
        List of training samples with 'input' and 'output' keys
    """
    input_path = Path(input_dir)
    json_files = list(input_path.rglob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    samples = []
    for file_path in json_files:
        with open(file_path, encoding="utf-8") as f:
            samples.append(json.load(f))

    print(f"Loaded {len(samples)} training samples from {input_dir}")
    return samples


def create_data_splits(samples: list[dict], config: TrainingConfig):
    """
    Split data into train/val/test sets using sklearn.

    Args:
        samples: List of training samples
        config: Training configuration

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    # First split: separate test set
    train_val_samples, test_samples = train_test_split(
        samples,
        test_size=config.test_ratio,
        random_state=config.seed,
    )

    # Second split: separate validation from training
    # Adjust val_ratio to be relative to train_val set
    val_ratio_adjusted = config.val_ratio / (1 - config.test_ratio)
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=val_ratio_adjusted,
        random_state=config.seed,
    )

    print(
        f"Data split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )
    return train_samples, val_samples, test_samples


def create_dataset(samples: list[dict], tokenizer) -> Dataset:
    """
    Create a HuggingFace Dataset from training samples.

    Args:
        samples: List of training samples
        tokenizer: Model tokenizer

    Returns:
        Dataset ready for training
    """
    conversations = [
        {
            "conversations": format_sample_as_chat_messages(
                input=sample["input"],
                output=sample["output"],
            )
        }
        for sample in samples
    ]
    dataset = Dataset.from_list(conversations)

    def format_conversations(examples):
        texts = []
        for conv in examples["conversations"]:
            # Apply Qwen3 chat template
            formatted_text = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(formatted_text)
        return {"text": texts}

    return dataset.map(format_conversations, batched=True)


def save_test_files(test_samples: list[dict], output_dir: Path, input_dir: str):
    """
    Save test sample references for later evaluation.

    Args:
        test_samples: List of test samples
        output_dir: Directory to save test file info
        input_dir: Original input directory
    """
    test_data = {
        "input_dir": input_dir,
        "num_samples": len(test_samples),
        "test_samples": test_samples,
    }

    test_file_path = output_dir / "test_samples.json"
    with open(test_file_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"Saved test sample references to {test_file_path}")


def setup_wandb(output_dir: Path, config: TrainingConfig):
    """
    Initialize Weights & Biases logging.

    Args:
        output_dir: Output directory for this run
        config: Training configuration
    """
    os.environ["WANDB_WATCH"] = "all"
    os.environ["WANDB_LOG_MODEL"] = "end"

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        group=config.wandb_group,
        config={
            "base_model": config.base_model,
            "max_seq_len": config.max_seq_len,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "batch_size_train": config.batch_size_train,
            "batch_size_eval": config.batch_size_eval,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_epochs": config.num_train_epochs,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "lr_scheduler_type": config.lr_scheduler_type,
            "optimizer": config.optimizer,
            "seed": config.seed,
            "target_chunk_size": config.target_chunk_size,
            "acceptable_range": config.acceptable_range,
        },
    )


def train(input_dir: str, output_dir: str):
    """
    Main training function.

    Args:
        input_dir: Directory containing training pair JSON files
        output_dir: Directory to save trained model and outputs
    """
    print("=== Starting Training ===")

    config = TrainingConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and split data
    samples = load_training_data(input_dir)
    train_samples, val_samples, test_samples = create_data_splits(samples, config)

    # Save test samples for later evaluation
    save_test_files(test_samples, output_path, input_dir)

    print(f"Loading model: {config.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_len,
        load_in_4bit=config.load_in_4bit,
    )

    print("Applying LoRA adapter")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
    )

    # Create datasets
    print("Creating datasets")
    train_dataset = create_dataset(train_samples, tokenizer)
    val_dataset = create_dataset(val_samples, tokenizer)

    # Initialize WandB
    setup_wandb(output_path, config)

    # Create trainer
    print("Setting up trainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        args=SFTConfig(
            output_dir=str(output_path),
            seed=config.seed,
            per_device_train_batch_size=config.batch_size_train,
            per_device_eval_batch_size=config.batch_size_eval,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config.logging_steps,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            optim=config.optimizer,
            report_to="wandb",
            gradient_checkpointing=config.use_gradient_checkpointing,
            dataset_text_field="text",
            max_seq_length=config.max_seq_len,
            packing=False,
        ),
    )

    # Configure trainer to train on responses only
    print("Configuring response-only training")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user",
        response_part="<|im_start|>assistant",
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("✓ Training completed")

    # Save final model
    print(f"Saving model to {output_path}")
    model.save_pretrained(str(output_path / "final_model"))
    tokenizer.save_pretrained(str(output_path / "final_model"))

    # Clean up WandB
    wandb.finish()
    print("=== Training Finished Successfully ===")


def main():
    parser = argparse.ArgumentParser(description="Train semantic chunking model on training pairs")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with training pair JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for trained model",
    )

    args = parser.parse_args()

    print(f"Training on data from: {args.input}")
    print(f"Saving model to: {args.output}")

    train(args.input, args.output)


if __name__ == "__main__":
    main()
