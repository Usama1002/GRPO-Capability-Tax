"""Creative writing benchmark -- measures vocabulary diversity and response richness."""

from .base import Benchmark


CREATIVE_PROMPTS = [
    "Write a short story about a lighthouse keeper who discovers something unusual in the fog.",
    "Describe a bustling market in a medieval town. Use all five senses.",
    "Write a poem about the passage of time.",
    "Create a dialogue between the sun and the moon meeting for the first time.",
    "Describe what silence sounds like in a dense forest.",
    "Write a letter from a sailor lost at sea to someone waiting at home.",
    "Tell a story about a door that only appears at midnight.",
    "Describe the taste of a color without naming the color.",
    "Write about a city where it never stops raining.",
    "Create a myth explaining why birds can fly but humans cannot.",
    "Describe a sunset as if you were explaining it to someone who has never seen one.",
    "Write a fairy tale that takes place entirely underground.",
    "Describe the sound of a thunderstorm from the perspective of a small insect.",
    "Write about the last tree on Earth.",
    "Create a conversation between two old books on a dusty shelf.",
    "Describe a dream that keeps changing every time you try to remember it.",
    "Write about a musician who can only play one note, but plays it perfectly.",
    "Describe the moment just before a wave crashes on the shore.",
    "Write a story where the main character is an emotion, not a person.",
    "Describe a forgotten garden coming back to life after a hundred years.",
    "Write about meeting your childhood self.",
    "Describe a library at the bottom of the ocean.",
    "Write a monologue from the perspective of a clock in an empty room.",
    "Describe the feeling of flying without using the word 'free'.",
    "Write about a bridge that connects two completely different worlds.",
]


class CreativeWritingBenchmark(Benchmark):
    name = "creative_writing"
    metric_name = "quality_score"

    def get_examples(self):
        return [{"prompt": p} for p in CREATIVE_PROMPTS]

    def score(self, example, response):
        if len(response.strip()) < 20:
            return 0.0

        words = response.lower().split()
        word_count = len(words)

        # Vocabulary diversity (type-token ratio)
        unique_words = len(set(words))
        ttr = unique_words / max(word_count, 1)

        # Length score (reward 50-300 words, penalize very short or very long)
        if word_count < 20:
            length_score = 0.2
        elif word_count < 50:
            length_score = 0.5
        elif word_count <= 300:
            length_score = 1.0
        elif word_count <= 500:
            length_score = 0.8
        else:
            length_score = 0.6

        # Descriptive richness (presence of adjectives/adverbs as proxy)
        descriptive_markers = [
            "ly ", "beautiful", "dark", "bright", "soft", "cold", "warm",
            "ancient", "gentle", "fierce", "silent", "whisper", "glow",
            "shadow", "shimmer", "echo", "drift", "fade", "bloom",
        ]
        desc_count = sum(1 for m in descriptive_markers if m in response.lower())
        richness = min(1.0, desc_count / 5)

        # Combine scores
        score = 0.4 * ttr + 0.3 * length_score + 0.3 * richness
        return min(1.0, score)
