"""ARC-Challenge benchmark -- science reasoning multiple choice."""

from datasets import load_dataset
from .base import Benchmark


class ARCBenchmark(Benchmark):
    name = "arc_challenge"
    metric_name = "accuracy"

    def __init__(self, num_examples=200):
        self.num_examples = num_examples
        self._examples = None

    def get_examples(self):
        if self._examples is None:
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", streaming=True)
            self._examples = []
            for i, ex in enumerate(ds):
                if i >= self.num_examples:
                    break
                choices = ex["choices"]
                labels = choices["label"]
                texts = choices["text"]
                answer_key = ex["answerKey"]

                choices_text = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
                self._examples.append({
                    "prompt": f"{ex['question']}\n\n{choices_text}\n\nAnswer with just the letter.",
                    "answer": answer_key,
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
