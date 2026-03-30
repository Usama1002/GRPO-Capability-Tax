"""Translation benchmark (English to German) with BLEU scoring."""

import re
import math
from collections import Counter
from datasets import load_dataset
from .base import Benchmark


def bleu_score(reference, hypothesis, max_n=4):
    """Simple BLEU implementation."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(clipped / total if total > 0 else 0)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


class TranslationBenchmark(Benchmark):
    name = "translation"
    metric_name = "bleu"

    def __init__(self, num_examples=100):
        self.num_examples = num_examples
        self._examples = None

    def get_examples(self):
        if self._examples is None:
            ds = load_dataset("wmt14", "de-en", split="test", streaming=True)
            self._examples = []
            for i, ex in enumerate(ds):
                if i >= self.num_examples:
                    break
                en_text = ex["translation"]["en"]
                de_text = ex["translation"]["de"]
                if 5 < len(en_text.split()) < 50:
                    self._examples.append({
                        "prompt": en_text,
                        "reference": de_text,
                    })
                    if len(self._examples) >= self.num_examples:
                        break
        return self._examples

    def format_prompt(self, example):
        return [
            {"role": "user", "content": f"Translate the following English sentence to German. Output ONLY the German translation, nothing else.\n\n{example['prompt']}"},
        ]

    def score(self, example, response):
        # Clean response (take first line, strip quotes)
        response = response.strip().split("\n")[0].strip('"\'')
        return bleu_score(example["reference"], response)
