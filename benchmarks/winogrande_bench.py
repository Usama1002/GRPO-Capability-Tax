"""Winogrande benchmark -- coreference resolution / language understanding."""

from datasets import load_dataset
from .base import Benchmark


class WinograndeBenchmark(Benchmark):
    name = "winogrande"
    metric_name = "accuracy"

    def __init__(self, num_examples=200):
        self.num_examples = num_examples
        self._examples = None

    def get_examples(self):
        if self._examples is None:
            ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation", streaming=True)
            self._examples = []
            for i, ex in enumerate(ds):
                if i >= self.num_examples:
                    break
                sentence = ex["sentence"]
                option1 = ex["option1"]
                option2 = ex["option2"]
                # answer is "1" or "2"
                answer = ex["answer"]
                correct_letter = "A" if answer == "1" else "B"

                self._examples.append({
                    "prompt": (
                        f"Complete the sentence by choosing A or B:\n\n"
                        f"{sentence}\n\n"
                        f"A. {option1}\n"
                        f"B. {option2}\n\n"
                        f"Answer with just the letter (A or B)."
                    ),
                    "answer": correct_letter,
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
