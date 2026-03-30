"""
Full experiment orchestrator.
For each model: train GRPO with dense checkpoints, then evaluate every checkpoint.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time


MODELS = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "gemma-2b": "google/gemma-2-2b-it",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3.8b": "microsoft/Phi-3.5-mini-instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def run_cmd(cmd, desc=""):
    """Run a command and stream output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
    return result.returncode


def get_checkpoints(output_dir):
    """Get sorted list of checkpoint directories."""
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(pattern)
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model keys to run (default: all)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only train")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models_to_run = args.models or list(MODELS.keys())
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    for model_key in models_to_run:
        if model_key not in MODELS:
            print(f"Unknown model key: {model_key}. Available: {list(MODELS.keys())}")
            continue

        model_name = MODELS[model_key]
        output_dir = os.path.join(results_dir, model_key)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_key} ({model_name})")
        print(f"{'#'*60}")

        # Phase 1: Train GRPO with dense checkpoints
        train_dir = os.path.join(output_dir, "training")
        if not args.skip_training:
            train_cmd = [
                sys.executable, "train_grpo_checkpointed.py",
                "--model", model_name,
                "--output-dir", train_dir,
                "--checkpoint-interval", str(args.checkpoint_interval),
            ]
            if args.dry_run:
                train_cmd.append("--dry-run")

            run_cmd(train_cmd, f"Training {model_key}")

        # Phase 2: Evaluate each checkpoint
        if not args.skip_eval:
            checkpoints = get_checkpoints(train_dir)
            print(f"\nFound {len(checkpoints)} checkpoints to evaluate")

            # First evaluate the base model
            base_eval_path = os.path.join(output_dir, "eval_base.json")
            if not os.path.exists(base_eval_path):
                eval_cmd = [
                    sys.executable, "evaluate_checkpoint.py",
                    "--base-model", model_name,
                    "--checkpoint", "base",
                    "--output", base_eval_path,
                ]
                run_cmd(eval_cmd, f"Evaluating {model_key} BASE")

            # Then evaluate each checkpoint
            for cp_path in checkpoints:
                step = cp_path.split("-")[-1]
                eval_path = os.path.join(output_dir, f"eval_step_{step}.json")
                if os.path.exists(eval_path):
                    print(f"  Skipping step {step} (already evaluated)")
                    continue

                eval_cmd = [
                    sys.executable, "evaluate_checkpoint.py",
                    "--base-model", model_name,
                    "--checkpoint", cp_path,
                    "--output", eval_path,
                ]
                run_cmd(eval_cmd, f"Evaluating {model_key} step {step}")

        print(f"\nCompleted {model_key}!")

    print(f"\n{'='*60}")
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Results in: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
