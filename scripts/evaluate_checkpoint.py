"""
Evaluate a single checkpoint (or base model) on all 10 benchmarks.
Outputs results as a JSON file.
"""

import argparse
import json
import os
import time

import torch

from benchmarks.base import load_model_from_checkpoint
from benchmarks.math_bench import MathBenchmark
from benchmarks.mmlu_bench import MMLUBenchmark
from benchmarks.ifeval_bench import IFEvalBenchmark
from benchmarks.creative_bench import CreativeWritingBenchmark
from benchmarks.summarization_bench import SummarizationBenchmark
from benchmarks.translation_bench import TranslationBenchmark
from benchmarks.coding_bench import CodingBenchmark
from benchmarks.safety_bench import SafetyBenchmark
from benchmarks.commonsense_bench import CommonsenseBenchmark
from benchmarks.conversation_bench import ConversationBenchmark


ALL_BENCHMARKS = [
    MathBenchmark(num_examples=200),
    MMLUBenchmark(num_examples=200),
    IFEvalBenchmark(),
    CreativeWritingBenchmark(),
    SummarizationBenchmark(num_examples=100),
    TranslationBenchmark(num_examples=100),
    CodingBenchmark(),
    SafetyBenchmark(),
    CommonsenseBenchmark(num_examples=200),
    ConversationBenchmark(),
]


def evaluate_all(model, tokenizer, benchmarks=None):
    """Run all benchmarks and return results dict."""
    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS

    results = {}
    for bench in benchmarks:
        print(f"  Running {bench.name}...", end=" ", flush=True)
        t0 = time.time()
        result = bench.evaluate(model, tokenizer)
        elapsed = time.time() - t0
        result["time_seconds"] = elapsed
        results[bench.name] = result
        print(f"{result['score']:.3f} ({elapsed:.0f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Base model name")
    parser.add_argument("--checkpoint", default="base", help="Checkpoint path or 'base'")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--benchmarks", nargs="*", default=None,
                        help="Specific benchmarks to run (default: all)")
    args = parser.parse_args()

    print(f"Loading model: {args.base_model}")
    if args.checkpoint != "base":
        print(f"  With checkpoint: {args.checkpoint}")

    model, tokenizer = load_model_from_checkpoint(args.checkpoint, args.base_model)

    # Filter benchmarks if specified
    benchmarks = ALL_BENCHMARKS
    if args.benchmarks:
        benchmarks = [b for b in ALL_BENCHMARKS if b.name in args.benchmarks]

    print(f"Running {len(benchmarks)} benchmarks...")
    results = evaluate_all(model, tokenizer, benchmarks)

    # Add metadata
    output = {
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
