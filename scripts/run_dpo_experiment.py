"""
DPO experiment orchestrator.
Trains DPO with dense checkpoints, then evaluates every checkpoint on all 13 benchmarks.
"""

import argparse
import glob
import json
import os
import subprocess
import sys


MODELS = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
}


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
    return result.returncode


def get_checkpoints(output_dir):
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(pattern)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models_to_run = args.models or list(MODELS.keys())
    results_dir = os.path.join(os.path.dirname(__file__), "results")

    for model_key in models_to_run:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}")
            continue

        model_name = MODELS[model_key]
        # Store DPO results separately from GRPO
        output_dir = os.path.join(results_dir, f"{model_key}-dpo")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'#'*60}")
        print(f"  DPO MODEL: {model_key} ({model_name})")
        print(f"{'#'*60}")

        train_dir = os.path.join(output_dir, "training")
        if not args.skip_training:
            train_cmd = [
                sys.executable, "train_dpo_checkpointed.py",
                "--model", model_name,
                "--output-dir", train_dir,
                "--checkpoint-interval", str(args.checkpoint_interval),
            ]
            if args.dry_run:
                train_cmd.append("--dry-run")
            run_cmd(train_cmd, f"DPO Training {model_key}")

        # Evaluate checkpoints
        checkpoints = get_checkpoints(train_dir)
        print(f"\nFound {len(checkpoints)} checkpoints to evaluate")

        # Base model eval
        base_eval_path = os.path.join(output_dir, "eval_base.json")
        if not os.path.exists(base_eval_path):
            # Copy from GRPO results if available
            grpo_base = os.path.join(results_dir, model_key, "eval_base.json")
            if os.path.exists(grpo_base):
                import shutil
                shutil.copy(grpo_base, base_eval_path)
                print(f"  Copied base eval from GRPO results")
            else:
                eval_cmd = [
                    sys.executable, "evaluate_checkpoint.py",
                    "--base-model", model_name,
                    "--checkpoint", "base",
                    "--output", base_eval_path,
                ]
                run_cmd(eval_cmd, f"Evaluating {model_key} BASE")

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
            run_cmd(eval_cmd, f"Evaluating DPO {model_key} step {step}")

        print(f"\nCompleted DPO {model_key}!")

    print(f"\n{'='*60}")
    print("  ALL DPO EXPERIMENTS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
