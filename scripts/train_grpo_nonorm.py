"""
GRPO training with the group-relative advantage normalization disabled (scale_rewards=False),
used for the group-normalization ablation. Advantages are mean-centred but not divided by the
within-group reward standard deviation. Identical to train_grpo_checkpointed.py otherwise.
"""

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "then give the final numerical answer after ####.\n"
    "Format: reasoning steps, then #### followed by the number."
)


def extract_answer(text):
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        pred = extract_answer(content)
        gt_val = extract_answer(gt) if isinstance(gt, str) else str(gt)
        if pred is not None and gt_val is not None:
            try:
                rewards.append(1.0 if float(pred) == float(gt_val) else 0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def prepare_dataset(dataset, use_system_prompt=True):
    def format_example(example):
        if use_system_prompt:
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ]
        else:
            # For models that don't support system role (e.g., Gemma)
            prompt = [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + example["question"]},
            ]
        gt_answer = extract_answer(example["answer"])
        return {"prompt": prompt, "answer": gt_answer if gt_answer else "0"}
    return dataset.map(format_example, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading dataset: openai/gsm8k")
    raw_dataset = load_dataset("openai/gsm8k", "main", split="train")
    # Gemma models don't support system role in chat template
    use_system = "gemma" not in args.model.lower()
    dataset = prepare_dataset(raw_dataset, use_system_prompt=use_system)
    if not use_system:
        print("Note: System prompt merged into user message (model does not support system role)")
    print(f"Dataset ready: {len(dataset)} examples")

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

    config = GRPOConfig(
        output_dir=args.output_dir,
        push_to_hub=False,

        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_steps=20,
        lr_scheduler_type="cosine",

        num_generations=4,
        max_completion_length=512,
        scale_rewards=False,  # ABLATION: mean-only baseline, no group-std normalization

        gradient_checkpointing=True,
        bf16=True,

        logging_steps=5,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.checkpoint_interval,
        save_total_limit=999,  # Keep ALL checkpoints for research

        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=correctness_reward,
        peft_config=peft_config,
    )

    print("Starting GRPO training with dense checkpointing...")
    trainer.train()

    # Save metrics
    metrics_file = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)
    print(f"Metrics saved to {metrics_file}")
    print("Done!")


if __name__ == "__main__":
    main()
