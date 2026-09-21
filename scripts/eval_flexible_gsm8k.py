"""
Evaluate GSM8K with the same flexible extraction heuristic used during GRPO training.
Tests the hypothesis that the GSM8K score decrease is due to evaluation-format mismatch.
"""

import argparse
import json
import re
import os
import glob

import torch
from datasets import load_dataset
# Make the repository root importable so 'benchmarks' resolves from any directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.base import load_model_from_checkpoint


def extract_answer_strict(text):
    """Strict: only extract after ####"""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_answer_flexible(text):
    """Flexible: try #### first, then fall back to last number"""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def evaluate_gsm8k(model, tokenizer, num_examples=200):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(ds.select(range(min(num_examples, len(ds)))))

    strict_correct = 0
    flexible_correct = 0
    total = 0

    system_prompt = (
        "You are a math tutor. Solve the problem step by step, "
        "then give the final numerical answer after ####.\n"
        "Format: reasoning steps, then #### followed by the number."
    )

    for ex in examples:
        gt_match = re.search(r"####\s*([\d,]+)", ex["answer"])
        gt = gt_match.group(1).replace(",", "") if gt_match else "0"

        # Try with system prompt
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["question"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [
                {"role": "user", "content": system_prompt + "\n\n" + ex["question"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Strict extraction
        pred_strict = extract_answer_strict(response)
        if pred_strict:
            try:
                if float(pred_strict) == float(gt):
                    strict_correct += 1
            except ValueError:
                pass

        # Flexible extraction
        pred_flex = extract_answer_flexible(response)
        if pred_flex:
            try:
                if float(pred_flex) == float(gt):
                    flexible_correct += 1
            except ValueError:
                pass

        total += 1

    return {
        "strict_accuracy": strict_correct / total,
        "flexible_accuracy": flexible_correct / total,
        "total": total,
        "strict_correct": strict_correct,
        "flexible_correct": flexible_correct,
    }


def get_step(path):
    fname = os.path.basename(path)
    if fname == "eval_base.json":
        return 0
    return int(fname.replace("eval_step_", "").replace(".json", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["qwen-1.5b", "qwen-3b", "phi-3.8b", "gemma-2b", "llama-3b"])
    parser.add_argument("--output", default="flexible_gsm8k_results.json")
    args = parser.parse_args()

    MODELS = {
        "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
        "phi-3.8b": "microsoft/Phi-3.5-mini-instruct",
        "gemma-2b": "google/gemma-2-2b-it",
        "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    }

    results = {}
    base_dir = os.path.join(os.path.dirname(__file__), "results")

    for mk in args.models:
        model_name = MODELS[mk]
        print(f"\n{'='*60}")
        print(f"  {mk}: Evaluating base and final checkpoint")
        print(f"{'='*60}")

        # Evaluate base model
        print(f"  Loading base model...")
        model, tokenizer = load_model_from_checkpoint("base", model_name)
        base_result = evaluate_gsm8k(model, tokenizer)
        print(f"  Base: strict={base_result['strict_accuracy']:.3f}, flexible={base_result['flexible_accuracy']:.3f}")
        del model
        torch.cuda.empty_cache()

        # Evaluate final GRPO checkpoint
        evals = sorted(glob.glob(os.path.join(base_dir, mk, "eval_*.json")), key=get_step)
        train_dir = os.path.join(base_dir, mk, "training")
        checkpoints = sorted(glob.glob(os.path.join(train_dir, "checkpoint-*")),
                           key=lambda x: int(x.split("-")[-1]))
        if checkpoints:
            final_cp = checkpoints[-1]
            print(f"  Loading final checkpoint: {os.path.basename(final_cp)}")
            model, tokenizer = load_model_from_checkpoint(final_cp, model_name)
            final_result = evaluate_gsm8k(model, tokenizer)
            print(f"  Final: strict={final_result['strict_accuracy']:.3f}, flexible={final_result['flexible_accuracy']:.3f}")
            del model
            torch.cuda.empty_cache()
        else:
            final_result = None
            print(f"  No checkpoints found for {mk}")

        results[mk] = {
            "base": base_result,
            "final": final_result,
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
