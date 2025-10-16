#!/usr/bin/env python3
"""Test script for vLLM API - compare base model vs LoRA."""

import sys
import re
import ast
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
from src.config import ChunkingConfig
from src.core.prompts import render_prediction_prompts

BASE_URL = "http://localhost:8000/v1"
BASE_MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"
LORA_MODEL = "qwen3_lora"


def load_test_doc():
    """Load test document."""
    with open(Path(__file__).parent / "test_sample.txt", 'r') as f:
        return f.read()


def test_model(model_name: str, doc_text: str):
    """Test a model and return parsed results."""
    config = ChunkingConfig()
    system_prompt, user_prompt = render_prediction_prompts(doc_text, config)
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.05,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).json()
    
    output = response['choices'][0]['message']['content']
    
    # Parse splits
    try:
        splits = ast.literal_eval(output.strip())
        return splits, output
    except:
        return None, output


def main():
    print("="*70)
    print("Testing Base Model vs LoRA")
    print("="*70)
    
    doc = load_test_doc()
    num_locs = len(re.findall(r'<\|loc_\d+\|>', doc))
    print(f"Document: {len(doc)} chars, {num_locs} locations\n")
    
    # Test base model
    print(f"[1/2] Testing Base Model: {BASE_MODEL}")
    base_splits, base_output = test_model(BASE_MODEL, doc)
    print(f"  Output: {base_output[:100]}...")
    if base_splits:
        print(f"  Splits: {base_splits}")
        print(f"  Chunks: {len(base_splits) + 1}")
    else:
        print(f"  ⚠️  Failed to parse")
    
    print()
    
    # Test LoRA model
    print(f"[2/2] Testing LoRA Model: {LORA_MODEL}")
    lora_splits, lora_output = test_model(LORA_MODEL, doc)
    print(f"  Output: {lora_output[:100]}...")
    if lora_splits:
        print(f"  Splits: {lora_splits}")
        print(f"  Chunks: {len(lora_splits) + 1}")
    else:
        print(f"  ⚠️  Failed to parse")
    
    print()
    
    # Compare
    print("="*70)
    if base_splits and lora_splits:
        if base_splits == lora_splits:
            print("⚠️  Same splits - LoRA might not be active!")
        else:
            print("✅ Different splits - LoRA is active!")
    print("="*70)


if __name__ == "__main__":
    main()


