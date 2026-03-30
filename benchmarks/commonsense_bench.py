"""Commonsense reasoning benchmark using HellaSwag."""

from datasets import load_dataset
from .base import Benchmark


class CommonsenseBenchmark(Benchmark):
    name = "commonsense"
    metric_name = "accuracy"

    def __init__(self, num_examples=200):
        self.num_examples = num_examples
        self._examples = None

    def get_examples(self):
        if self._examples is None:
            ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
            self._examples = []
            for i, ex in enumerate(ds):
                if i >= self.num_examples:
                    break
                endings = ex["endings"]
                correct_idx = int(ex["label"])
                letters = ["A", "B", "C", "D"]
                choices_text = "\n".join(
                    f"{letters[j]}. {e}" for j, e in enumerate(endings)
                )
                self._examples.append({
                    "prompt": f"{ex['ctx']}\n\nWhich ending makes the most sense?\n{choices_text}\n\nAnswer with just the letter.",
                    "answer": letters[correct_idx],
                })
        return self._examples

    def score(self, example, response):
        response = response.strip().upper()
        correct = example["answer"]
        if response and response[0] == correct:
            return 1.0
        if f"({correct})" in response or f" {correct}." in response:
            return 1.0
        return 0.0
