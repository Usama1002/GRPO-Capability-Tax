"""GSM8K math reasoning benchmark (target task)."""

import re
from datasets import load_dataset
from .base import Benchmark


class MathBenchmark(Benchmark):
    name = "math_reasoning"
    metric_name = "accuracy"

    def __init__(self, num_examples=200):
        self.num_examples = num_examples
        self._examples = None

    def get_examples(self):
        if self._examples is None:
            ds = load_dataset("openai/gsm8k", "main", split="test")
            self._examples = []
            for ex in ds.select(range(min(self.num_examples, len(ds)))):
                answer = re.search(r"####\s*([\d,]+)", ex["answer"])
                gt = answer.group(1).replace(",", "") if answer else "0"
                self._examples.append({
                    "prompt": ex["question"],
                    "answer": gt,
                })
        return self._examples

    def format_prompt(self, example):
        return [
            {"role": "system", "content": "Solve the math problem step by step. Give the final answer after ####."},
            {"role": "user", "content": example["prompt"]},
        ]

    def score(self, example, response):
        match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", response)
        if match:
            pred = match.group(1).replace(",", "")
        else:
            numbers = re.findall(r"[\d,]+(?:\.\d+)?", response)
            pred = numbers[-1].replace(",", "") if numbers else ""

        try:
            return 1.0 if float(pred) == float(example["answer"]) else 0.0
        except (ValueError, IndexError):
            return 0.0
