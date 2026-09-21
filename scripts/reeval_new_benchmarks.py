"""
Re-evaluate existing checkpoints on the 3 NEW benchmarks only.
Merges new results into existing eval JSON files without overwriting old data.
"""

import argparse
import glob
import json
import os
import time

import torch

# Make the repository root importable so 'benchmarks' resolves from any directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.base import load_model_from_checkpoint
from benchmarks.arc_bench import ARCBenchmark
from benchmarks.truthfulqa_bench import TruthfulQABenchmark
from benchmarks.winogrande_bench import WinograndeBenchmark

NEW_BENCHMARKS = [
    ARCBenchmark(num_examples=200),
    TruthfulQABenchmark(num_examples=200),
    WinograndeBenchmark(num_examples=200),
]

MODELS = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "phi-3.8b": "microsoft/Phi-3.5-mini-instruct",
    "gemma-2b": "google/gemma-2-2b-it",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def evaluate_new_benchmarks(model, tokenizer):
    """Run only the 3 new benchmarks."""
    # Detect system role support
    supports_system = True
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}],
            tokenize=False,
        )
    except Exception:
        supports_system = False

    results = {}
    for bench in NEW_BENCHMARKS:
        print(f"    {bench.name}...", end=" ", flush=True)
        t0 = time.time()
        result = bench.evaluate(model, tokenizer)
        elapsed = time.time() - t0
        result["time_seconds"] = elapsed
        results[bench.name] = result
        print(f"{result['score']:.3f} ({elapsed:.0f}s)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    models_to_run = args.models or list(MODELS.keys())

    for model_key in models_to_run:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}")
            continue

        model_name = MODELS[model_key]
        model_dir = os.path.join(RESULTS_DIR, model_key)
        eval_files = sorted(glob.glob(os.path.join(model_dir, "eval_*.json")))

        if not eval_files:
            print(f"No eval files for {model_key}, skipping")
            continue

        print(f"\n{'#'*60}")
        print(f"  Re-evaluating {model_key} ({len(eval_files)} checkpoints)")
        print(f"{'#'*60}")

        for eval_path in eval_files:
            fname = os.path.basename(eval_path)

            # Check if already has new benchmarks
            with open(eval_path) as f:
                existing = json.load(f)

            if "arc_challenge" in existing.get("results", {}):
                print(f"  Skipping {fname} (already has new benchmarks)")
                continue

            # Determine checkpoint path
            if fname == "eval_base.json":
                checkpoint = "base"
                print(f"\n  Evaluating BASE on new benchmarks...")
            else:
                step = fname.replace("eval_step_", "").replace(".json", "")
                train_dir = os.path.join(model_dir, "training")
                checkpoint = os.path.join(train_dir, f"checkpoint-{step}")
                if not os.path.exists(checkpoint):
                    print(f"  Skipping {fname} (checkpoint dir missing)")
                    continue
                print(f"\n  Evaluating step {step} on new benchmarks...")

            # Load model
            model, tokenizer = load_model_from_checkpoint(checkpoint, model_name)

            # Run new benchmarks
            new_results = evaluate_new_benchmarks(model, tokenizer)

            # Merge into existing results
            existing["results"].update(new_results)
            with open(eval_path, "w") as f:
                json.dump(existing, f, indent=2)

            # Free GPU
            del model
            torch.cuda.empty_cache()

        print(f"  Done with {model_key}!")

    print("\nAll re-evaluations complete!")


if __name__ == "__main__":
    main()
