"""
SFT training on GSM8K with dense checkpoint saving.
Baseline comparison for the GRPO capability degradation study.
Uses the same LoRA config and checkpointing as GRPO experiments.
"""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "then give the final numerical answer after ####.\n"
    "Format: reasoning steps, then #### followed by the number."
)


def prepare_dataset(dataset, use_system_prompt=True):
    """Convert GSM8K to chat format for SFT."""
    def format_example(example):
        if use_system_prompt:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        else:
            messages = [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        return {"messages": messages}
    return dataset.map(format_example, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading dataset: openai/gsm8k")
    raw_dataset = load_dataset("openai/gsm8k", "main", split="train")
    use_system = "gemma" not in args.model.lower() and "llama" not in args.model.lower()
    dataset = prepare_dataset(raw_dataset, use_system_prompt=use_system)
    print(f"Dataset ready: {len(dataset)} examples")

    max_steps = 2 if args.dry_run else -1
    print(f"Model: {args.model}")

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0},
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        push_to_hub=False,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        bf16=True,
        max_length=1024,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.checkpoint_interval,
        save_total_limit=999,
        eval_strategy="no",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=config,
        peft_config=peft_config,
    )

    print("Starting SFT training with dense checkpointing...")
    trainer.train()

    metrics_file = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)
    print(f"Metrics saved to {metrics_file}")
    print("Done!")


if __name__ == "__main__":
    main()
