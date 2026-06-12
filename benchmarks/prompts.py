"""
Every stage benchmarks the exact same workload: same prompt, same number
of generated tokens, same run count. If this file changes, numbers across
stages are no longer comparable and the golden-output correctness check
breaks (greedy decoding is only deterministic for a fixed prompt).

The engine evolves; the workload stays frozen.
"""

PROMPT = (
    "Explain, step by step, how a transformer language model generates "
    "text one token at a time, and why caching past computations makes "
    "this process faster and more efficient."
)

#decode length: long enough to see TPOT (time per output token) trends
MAX_NEW_TOKENS = 128   
#timed runs to average (after one discarded warmup)
N_RUNS = 3             