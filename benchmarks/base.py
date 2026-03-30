"""Base benchmark class and utilities."""

import json
import re
import time
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str = "base"
    metric_name: str = "score"

    @abstractmethod
    def get_examples(self) -> list[dict]:
        """Return list of evaluation examples."""
        ...

    @abstractmethod
    def score(self, example: dict, response: str) -> float:
        """Score a single model response. Returns 0-1."""
        ...

    def format_prompt(self, example: dict) -> list[dict]:
        """Format example into chat messages. Override for custom formatting."""
        return [{"role": "user", "content": example["prompt"]}]

    def evaluate(self, model, tokenizer, max_examples: int = None) -> dict:
        """Run the benchmark on a model. Returns metrics dict."""
        examples = self.get_examples()
        if max_examples:
            examples = examples[:max_examples]

        # Detect if model supports system role
        supports_system = True
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}],
                tokenize=False,
            )
        except Exception:
            supports_system = False

        scores = []
        for ex in examples:
            messages = self.format_prompt(ex)
            # Merge system into user if model doesn't support system role
            if not supports_system and messages and messages[0]["role"] == "system":
                sys_content = messages[0]["content"]
                messages = messages[1:]
                if messages:
                    messages[0]["content"] = sys_content + "\n\n" + messages[0]["content"]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            score = self.score(ex, response)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "benchmark": self.name,
            "metric": self.metric_name,
            "score": avg_score,
            "num_examples": len(scores),
        }


def load_model_from_checkpoint(checkpoint_path: str, base_model_name: str):
    """Load a PEFT model from a checkpoint directory."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    if checkpoint_path != "base":
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer
