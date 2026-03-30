"""
DPO training with dense checkpoint saving for capability degradation research.
Saves checkpoints every N steps for longitudinal evaluation.
Uses UltraFeedback Binarized dataset.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"


def prepare_dataset(dataset):
    """Convert conversational format to flat strings for DPO."""
    def to_flat(example):
        chosen_msgs = example["chosen"]
        rejected_msgs = example["rejected"]
        chosen_text = ""
        rejected_text = ""
        for msg in chosen_msgs:
            if msg["role"] == "assistant":
                chosen_text = msg["content"]
        for msg in rejected_msgs:
            if msg["role"] == "assistant":
                rejected_text = msg["content"]
        return {
            "prompt": example["prompt"],
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    dataset = dataset.map(to_flat, remove_columns=[
        c for c in dataset.column_names if c not in ("prompt", "chosen", "rejected")
    ])
    dataset = dataset.filter(lambda x: len(x["chosen"]) > 0 and len(x["rejected"]) > 0)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train_prefs")
    dataset = prepare_dataset(dataset)
    print(f"Dataset ready: {len(dataset)} preference pairs")

    # Use a subset to match GRPO training time roughly
    # GRPO trains on 7.5K examples, DPO on ~60K is much more
    # Use first 10K for comparable training duration
    dataset = dataset.select(range(min(10000, len(dataset))))
    print(f"Using subset: {len(dataset)} examples")

    max_steps = 2 if args.dry_run else -1
    print(f"Model: {args.model}")
    print(f"Checkpoint interval: {args.checkpoint_interval} steps")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    config = DPOConfig(
        output_dir=args.output_dir,
        push_to_hub=False,

        beta=0.1,

        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        warmup_steps=20,
        lr_scheduler_type="cosine",

        gradient_checkpointing=True,
        bf16=True,
        max_length=768,

        logging_steps=5,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.checkpoint_interval,
        save_total_limit=999,  # Keep ALL checkpoints

        eval_strategy="no",

        report_to="none",
        seed=42,
    )

    trainer = DPOTrainer(
        model=args.model,
        train_dataset=dataset,
        args=config,
        peft_config=peft_config,
    )

    print("Starting DPO training with dense checkpointing...")
    trainer.train()

    metrics_file = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)
    print(f"Metrics saved to {metrics_file}")
    print("Done!")


if __name__ == "__main__":
    main()
