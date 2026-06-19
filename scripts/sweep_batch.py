"""Stage-3 experiment: throughput vs batch size.
Run:  python scripts/sweep_batch.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.loader import load_model
from engine.generator import batched_generate
from benchmarks.harness import run_batch_benchmark
from benchmarks.batch_prompts import BATCH_PROMPTS
from benchmarks.prompts import MAX_NEW_TOKENS, N_RUNS

BATCH_SIZES = [1, 2, 4, 8, 16, 32]

def main():
    model, tokenizer = load_model()
    results = []
    for b in BATCH_SIZES:
        # cycle the prompt pool up to size b so every batch is full
        prompts = [BATCH_PROMPTS[i % len(BATCH_PROMPTS)] for i in range(b)]
        m = run_batch_benchmark(model, tokenizer, batched_generate, prompts,
                                MAX_NEW_TOKENS, N_RUNS, stage_name="stage3_batching")
        results.append((b, m["throughput_tok_s"], m["per_seq_tok_s"], m["peak_mem_gb"]))

    print("\n=== THROUGHPUT SWEEP ===")
    print(f"{'batch':>6} {'tok/s total':>12} {'tok/s/seq':>11} {'mem GB':>8}")
    for b, thr, pseq, mem in results:
        print(f"{b:>6} {thr:>12.1f} {pseq:>11.1f} {mem:>8.2f}")

if __name__ == "__main__":
    main()