"""Summarization benchmark using XSum subset with ROUGE-L."""

from datasets import load_dataset
from .base import Benchmark


def rouge_l(reference, hypothesis):
    """Simple ROUGE-L implementation (longest common subsequence)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words or not hyp_words:
        return 0.0

    # LCS using dynamic programming
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


class SummarizationBenchmark(Benchmark):
    name = "summarization"
    metric_name = "rouge_l"

    def __init__(self, num_examples=100):
        self.num_examples = num_examples
        self._examples = None

    def get_examples(self):
        if self._examples is None:
            ds = load_dataset("EdinburghNLP/xsum", split="test", streaming=True)
            self._examples = []
            for i, ex in enumerate(ds):
                if i >= self.num_examples:
                    break
                # Only use articles that are reasonable length
                if 100 < len(ex["document"].split()) < 500:
                    self._examples.append({
                        "prompt": ex["document"],
                        "summary": ex["summary"],
                    })
                    if len(self._examples) >= self.num_examples:
                        break
        return self._examples

    def format_prompt(self, example):
        return [
            {"role": "user", "content": f"Summarize the following article in one sentence:\n\n{example['prompt'][:1500]}"},
        ]

    def score(self, example, response):
        return rouge_l(example["summary"], response)
