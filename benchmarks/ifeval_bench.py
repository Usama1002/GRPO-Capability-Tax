"""Instruction following benchmark with verifiable constraints."""

from .base import Benchmark


IFEVAL_PROMPTS = [
    {"prompt": "List exactly 5 benefits of exercise. Number them 1-5.", "check": "count_items", "expected": 5},
    {"prompt": "Write a sentence that contains exactly 10 words.", "check": "word_count", "expected": 10},
    {"prompt": "Name 3 countries in Europe. Use bullet points.", "check": "has_bullets", "expected": True},
    {"prompt": "Explain gravity in exactly 2 sentences.", "check": "sentence_count", "expected": 2},
    {"prompt": "Write a paragraph about dogs. Do NOT mention the word 'pet'.", "check": "forbidden_word", "expected": "pet"},
    {"prompt": "List 4 colors of the rainbow. Use a numbered list.", "check": "count_items", "expected": 4},
    {"prompt": "Respond to this in ALL CAPS: What is the capital of Japan?", "check": "all_caps", "expected": True},
    {"prompt": "Write exactly 3 sentences about the moon.", "check": "sentence_count", "expected": 3},
    {"prompt": "Name 5 fruits. Separate them with commas on a single line.", "check": "comma_separated", "expected": 5},
    {"prompt": "Explain what water is in under 20 words.", "check": "max_words", "expected": 20},
    {"prompt": "List exactly 3 programming languages. Use bullet points.", "check": "count_items", "expected": 3},
    {"prompt": "Write 4 sentences about the sun. Number each sentence.", "check": "count_items", "expected": 4},
    {"prompt": "Describe a cat in exactly 1 sentence.", "check": "sentence_count", "expected": 1},
    {"prompt": "Name 6 musical instruments. Use a numbered list.", "check": "count_items", "expected": 6},
    {"prompt": "Write a response that is exactly 3 paragraphs long.", "check": "paragraph_count", "expected": 3},
    {"prompt": "Give me 2 reasons to drink water. Start each with 'First' and 'Second'.", "check": "starts_with_markers", "expected": ["first", "second"]},
    {"prompt": "Explain photosynthesis. Your response must contain the word 'sunlight'.", "check": "required_word", "expected": "sunlight"},
    {"prompt": "List 3 types of clouds. Do NOT use any punctuation marks.", "check": "no_punctuation", "expected": True},
    {"prompt": "Write about rain in exactly 5 sentences.", "check": "sentence_count", "expected": 5},
    {"prompt": "Name 3 oceans. Put each on its own line.", "check": "line_count", "expected": 3},
    {"prompt": "Write a haiku (3 lines, 5-7-5 syllables) about winter.", "check": "line_count", "expected": 3},
    {"prompt": "Explain addition in under 30 words.", "check": "max_words", "expected": 30},
    {"prompt": "List exactly 7 days of the week.", "check": "count_items", "expected": 7},
    {"prompt": "Write about space. Use the words 'star', 'planet', and 'galaxy'.", "check": "required_words", "expected": ["star", "planet", "galaxy"]},
    {"prompt": "Give 2 advantages and 2 disadvantages of social media. Use bullet points.", "check": "count_items", "expected": 4},
    {"prompt": "Describe an apple using exactly 3 adjectives, separated by commas.", "check": "comma_separated", "expected": 3},
    {"prompt": "Write a response about trees. Every sentence must end with an exclamation mark.", "check": "ends_with_excl", "expected": True},
    {"prompt": "List 5 animals that can swim. Number them.", "check": "count_items", "expected": 5},
    {"prompt": "Explain what a computer is in 2 sentences or less.", "check": "max_sentences", "expected": 2},
    {"prompt": "Write 3 facts about Mars. Start each fact with a dash (-).", "check": "count_dashes", "expected": 3},
]


class IFEvalBenchmark(Benchmark):
    name = "instruction_following"
    metric_name = "constraint_satisfaction"

    def get_examples(self):
        return IFEVAL_PROMPTS

    def score(self, example, response):
        import re
        check = example["check"]
        expected = example["expected"]
        response_lower = response.lower().strip()

        if check == "count_items":
            items = re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]\s*|-\s*|\*\s*)", response)
            return 1.0 if len(items) == expected else 0.0

        elif check == "word_count":
            words = response.split()
            return 1.0 if len(words) == expected else 0.0

        elif check == "has_bullets":
            return 1.0 if re.search(r"[-\*\u2022]", response) else 0.0

        elif check == "sentence_count":
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            return 1.0 if len(sentences) == expected else (0.5 if abs(len(sentences) - expected) <= 1 else 0.0)

        elif check == "forbidden_word":
            return 1.0 if expected not in response_lower else 0.0

        elif check == "all_caps":
            alpha_chars = [c for c in response if c.isalpha()]
            if not alpha_chars:
                return 0.0
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            return 1.0 if upper_ratio > 0.8 else 0.0

        elif check == "comma_separated":
            commas = response.count(",")
            return 1.0 if commas >= expected - 1 else 0.0

        elif check == "max_words":
            return 1.0 if len(response.split()) <= expected else 0.0

        elif check == "paragraph_count":
            paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
            return 1.0 if len(paragraphs) == expected else 0.0

        elif check == "starts_with_markers":
            return 1.0 if all(m in response_lower for m in expected) else 0.0

        elif check == "required_word":
            return 1.0 if expected in response_lower else 0.0

        elif check == "required_words":
            return 1.0 if all(w in response_lower for w in expected) else 0.0

        elif check == "no_punctuation":
            punct_count = sum(1 for c in response if c in ".,;:!?")
            return 1.0 if punct_count == 0 else 0.0

        elif check == "line_count":
            lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
            return 1.0 if len(lines) >= expected else 0.0

        elif check == "max_sentences":
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            return 1.0 if len(sentences) <= expected else 0.0

        elif check == "count_dashes":
            dashes = re.findall(r"(?:^|\n)\s*-\s*", response)
            return 1.0 if len(dashes) == expected else 0.0

        elif check == "ends_with_excl":
            sentences = [s.strip() for s in response.strip().split("\n") if s.strip()]
            if not sentences:
                return 0.0
            excl_count = sum(1 for s in sentences if s.endswith("!"))
            return excl_count / len(sentences)

        return 0.5
