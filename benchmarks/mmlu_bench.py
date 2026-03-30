"""MMLU general knowledge benchmark."""

import random
from datasets import load_dataset
from .base import Benchmark


class MMLUBenchmark(Benchmark):
    name = "general_knowledge"
    metric_name = "accuracy"

    def __init__(self, num_examples=200):
        self.num_examples = num_examples
        self._examples = None

    def get_examples(self):
        if self._examples is None:
            ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
            self._examples = []
            for i, ex in enumerate(ds):
                if i >= self.num_examples:
                    break
                choices = ex["choices"]
                correct_idx = ex["answer"]
                letters = ["A", "B", "C", "D"]
                choices_text = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices))
                self._examples.append({
                    "prompt": f"{ex['question']}\n\n{choices_text}\n\nAnswer with just the letter (A, B, C, or D).",
                    "answer": letters[correct_idx],
                })
        return self._examples

    def score(self, example, response):
        response = response.strip().upper()
        correct = example["answer"]
        # Check if the response starts with or contains the correct letter
        if response and response[0] == correct:
            return 1.0
        if f"({correct})" in response or f" {correct}." in response or f" {correct} " in response:
            return 1.0
        return 0.0
