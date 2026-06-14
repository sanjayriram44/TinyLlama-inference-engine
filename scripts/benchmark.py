"""Run the stage-0 baseline:  python scripts/benchmark.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))   # repo root importable

from benchmarks.harness import run_benchmark
from benchmarks.prompts import MAX_NEW_TOKENS, N_RUNS, PROMPT
from engine.generator import naive_generate
from engine.loader import load_model


def main():
    model, tokenizer = load_model()
    run_benchmark(model, tokenizer, naive_generate,
                  PROMPT, MAX_NEW_TOKENS, N_RUNS, stage_name="stage0_baseline")


if __name__ == "__main__":
    main()
