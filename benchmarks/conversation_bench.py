"""Conversational quality benchmark -- multi-turn helpfulness scoring."""

from .base import Benchmark


CONVERSATION_PROMPTS = [
    "I just moved to a new city and don't know anyone. What should I do?",
    "Can you help me plan a surprise birthday party for my best friend?",
    "I'm feeling overwhelmed with work. Any advice?",
    "What's a good way to start learning to cook?",
    "I had an argument with my friend and I'm not sure how to fix things.",
    "I want to start a small garden on my balcony. Where do I begin?",
    "Can you recommend some strategies for better time management?",
    "I'm trying to decide between two job offers. How should I think about it?",
    "What are some fun indoor activities for a rainy weekend?",
    "I want to start reading more. How do I build a reading habit?",
    "My neighbor's dog keeps barking at night. What should I do?",
    "I want to reduce my screen time. Any practical tips?",
    "How do I start saving money when I have a tight budget?",
    "I'm nervous about a presentation I have to give next week.",
    "What's the best way to learn a new language as an adult?",
    "I want to start exercising but I hate going to the gym.",
    "How can I make my home more energy efficient?",
    "I'm having trouble sleeping. What can I try?",
    "What should I consider before adopting a pet?",
    "How do I politely decline a social invitation when I need alone time?",
    "I want to volunteer in my community. How do I find opportunities?",
    "What are some ways to make long commutes more bearable?",
    "I'm thinking about going back to school. Is it worth it?",
    "How can I be more environmentally friendly in my daily life?",
    "I want to learn photography as a hobby. Where do I start?",
]


class ConversationBenchmark(Benchmark):
    name = "conversation"
    metric_name = "helpfulness"

    def get_examples(self):
        return [{"prompt": p} for p in CONVERSATION_PROMPTS]

    def score(self, example, response):
        if len(response.strip()) < 20:
            return 0.0

        words = response.split()
        word_count = len(words)
        score = 0.0

        # 1. Appropriate length (0.3 points)
        if 30 <= word_count <= 300:
            score += 0.3
        elif 15 <= word_count <= 500:
            score += 0.15

        # 2. Structured response (0.2 points)
        import re
        has_structure = bool(re.search(r"(\d+[\.\)]\s|-\s|\*\s)", response))
        has_paragraphs = "\n" in response.strip()
        if has_structure:
            score += 0.2
        elif has_paragraphs:
            score += 0.1

        # 3. Empathetic/helpful tone (0.25 points)
        helpful_markers = [
            "you could", "you might", "consider", "try", "suggest",
            "recommend", "one way", "another option", "here are",
            "it's normal", "that's understandable", "great question",
        ]
        found = sum(1 for m in helpful_markers if m in response.lower())
        score += min(0.25, found * 0.05)

        # 4. Actionable advice (0.25 points)
        action_markers = [
            "start by", "first", "then", "next", "finally",
            "step", "make sure", "remember to", "don't forget",
            "tip", "key", "important",
        ]
        found = sum(1 for m in action_markers if m in response.lower())
        score += min(0.25, found * 0.05)

        return min(1.0, score)
