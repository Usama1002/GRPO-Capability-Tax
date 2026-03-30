"""TruthfulQA benchmark -- tests factual accuracy and resistance to common misconceptions."""

from datasets import load_dataset
from .base import Benchmark


class TruthfulQABenchmark(Benchmark):
    name = "truthfulqa"
    metric_name = "accuracy"

    def __init__(self, num_examples=200):
        self.num_examples = num_examples
        self._examples = None

    def get_examples(self):
        if self._examples is None:
            ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", streaming=True)
            self._examples = []
            for i, ex in enumerate(ds):
                if i >= self.num_examples:
                    break
                # mc1_targets: single correct answer format
                choices = ex["mc1_targets"]["choices"]
                labels_list = ex["mc1_targets"]["labels"]

                # Find the correct answer (label=1)
                correct_idx = labels_list.index(1) if 1 in labels_list else 0
                letters = [chr(65 + j) for j in range(len(choices))]  # A, B, C, ...

                choices_text = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices))
                self._examples.append({
                    "prompt": f"{ex['question']}\n\n{choices_text}\n\nAnswer with just the letter.",
                    "answer": letters[correct_idx],
                })
        return self._examples

    def score(self, example, response):
        response = response.strip().upper()
        correct = example["answer"].upper()
        if response and response[0] == correct:
            return 1.0
        if f"({correct})" in response or f" {correct}." in response or f" {correct} " in response:
            return 1.0
        return 0.0
