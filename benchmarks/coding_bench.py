"""Coding benchmark -- checks if generated Python code is syntactically valid and reasonable."""

import ast
from .base import Benchmark


CODING_PROMPTS = [
    {"prompt": "Write a Python function that checks if a number is prime.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function to reverse a string.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that computes the factorial of a number.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function to find the maximum element in a list.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that checks if a string is a palindrome.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function to count vowels in a string.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that computes the Fibonacci sequence up to n terms.", "must_contain": ["def "]},
    {"prompt": "Write a Python function to sort a list using bubble sort.", "must_contain": ["def "]},
    {"prompt": "Write a Python function that merges two sorted lists.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function to find the GCD of two numbers.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that flattens a nested list.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function to check if two strings are anagrams.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that removes duplicates from a list.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function to convert Celsius to Fahrenheit.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that counts word frequency in a string.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python class for a simple stack with push, pop, and peek.", "must_contain": ["class ", "def "]},
    {"prompt": "Write a Python function to binary search a sorted list.", "must_contain": ["def "]},
    {"prompt": "Write a Python function that generates all permutations of a list.", "must_contain": ["def "]},
    {"prompt": "Write a Python function to check if brackets in a string are balanced.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that implements a simple calculator (add, subtract, multiply, divide).", "must_contain": ["def "]},
    {"prompt": "Write a Python function to find the second largest number in a list.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that rotates a list by k positions.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function to compute the power of a number without using **.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function that transposes a 2D matrix.", "must_contain": ["def ", "return"]},
    {"prompt": "Write a Python function to validate an email address using basic rules.", "must_contain": ["def ", "return"]},
]


class CodingBenchmark(Benchmark):
    name = "coding"
    metric_name = "code_quality"

    def get_examples(self):
        return CODING_PROMPTS

    def format_prompt(self, example):
        return [
            {"role": "system", "content": "You are a Python programmer. Write clean, correct Python code. Output only the code, no explanations."},
            {"role": "user", "content": example["prompt"]},
        ]

    def score(self, example, response):
        # Extract code from response (handle markdown code blocks)
        code = response
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        score = 0.0

        # 1. Syntactic validity (0.5 points)
        try:
            ast.parse(code)
            score += 0.5
        except SyntaxError:
            return 0.0  # Can't evaluate further if syntax is broken

        # 2. Contains required elements (0.3 points)
        must_contain = example.get("must_contain", [])
        if must_contain:
            found = sum(1 for m in must_contain if m in code)
            score += 0.3 * (found / len(must_contain))

        # 3. Reasonable length (0.2 points)
        lines = [l for l in code.strip().split("\n") if l.strip()]
        if 3 <= len(lines) <= 50:
            score += 0.2
        elif len(lines) > 0:
            score += 0.1

        return min(1.0, score)
