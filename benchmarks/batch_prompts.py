"""Prompts of DELIBERATELY different lengths, to exercise padding + the mask.
Separate from the frozen single-stream prompt so stage 0/2 stay comparable."""

BATCH_PROMPTS = [
    "Explain how a transformer generates text one token at a time.",
    "What is attention?",
    "Describe the difference between prefill and decode in LLM inference, and why one is compute-bound while the other is memory-bound.",
    "Why do we cache keys and values but not queries?",
    "List three ways to make inference faster.",
    "Hello.",
]