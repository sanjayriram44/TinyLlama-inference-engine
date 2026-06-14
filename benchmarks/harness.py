"""
Scoreboard it produces:
  ttft_s        time to first token (~ prefill cost)
  tpot_ms       mean time per output token after the first (~ decode cost)
  tokens_per_s  max_new_tokens / total time
  peak_mem_gb   torch.cuda.max_memory_allocated
  output check  saved + compared against golden_output.txt

GPU utilization is observed live instead: run `watch -n 0.5 nvidia-smi`
in a second terminal while the benchmark runs.
"""

import json
import statistics
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).parent / "results"
GOLDEN_FILE = RESULTS_DIR / "golden_output.txt"


def run_benchmark(model, tokenizer, generate_fn, prompt: str,
                  max_new_tokens: int, n_runs: int, stage_name: str) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    #warmup
   #One throwaway generation of just 8 tokens. Its result is ignored — its only job is to absorb one-time costs (CUDA context init, cuDNN kernel autotuning, allocator warm-up) that would otherwise inflate the first timed run. Standard benchmarking hygiene: never measure the cold start.
    generate_fn(model, input_ids, 8)

    # ---- timed runs ----
    torch.cuda.reset_peak_memory_stats()
    all_times, ids = [], None
    for _ in range(n_runs):
        ids, token_times = generate_fn(model, input_ids, max_new_tokens)
        all_times.append(token_times)

    # ---- metrics, averaged over runs ----
    # token_times are CUMULATIVE seconds-since-start:
    #   entry 0       = prefill + first token  -> TTFT
    #   diffs between consecutive entries      -> per-token decode cost -> TPOT
    #   last entry    = total generation time
    ttft = statistics.mean(t[0] for t in all_times)
    tpot = statistics.mean(
        statistics.mean(t[i] - t[i - 1] for i in range(1, len(t)))
        for t in all_times
    )
    total = statistics.mean(t[-1] for t in all_times)

    output_text = tokenizer.decode(ids[0], skip_special_tokens=True)

    metrics = {
        "stage": stage_name,
        "prompt_tokens": input_ids.shape[1],
        "new_tokens": max_new_tokens,
        "ttft_s": round(ttft, 4),
        "tpot_ms": round(tpot * 1000, 2),
        "tokens_per_s": round(max_new_tokens / total, 2),
        "total_s": round(total, 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "output_matches_golden": _check_golden(output_text),
    }

    (RESULTS_DIR / f"{stage_name}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def _check_golden(output_text: str):
    """Stage 0 CREATES the golden output; later stages COMPARE against it.
    Greedy decoding is deterministic, so a lossless optimization (KV cache)
    must reproduce it exactly -- a mismatch there is a BUG. Lossy stages
    (quantization) will legitimately mismatch; that's the cue to measure
    quality loss instead of equality."""
    if not GOLDEN_FILE.exists():
        GOLDEN_FILE.write_text(output_text)
        print(f"golden output created -> {GOLDEN_FILE}")
        return "created"
    return GOLDEN_FILE.read_text() == output_text



# Tokenize the prompt and place it on the GPU.
#
# tokenizer(prompt, return_tensors="pt") returns a dict-like BatchEncoding:
#   {
#     "input_ids":      tensor([[ 1, 6573, 274, ... ]]),  # token IDs, shape (1, T)
#     "attention_mask": tensor([[ 1,    1,   1, ... ]]),  # 1 = real token, 0 = padding
#   }
#   - return_tensors="pt" => values come back as PyTorch tensors ("pt"); "np" = NumPy.
#   - The tokenizer adds a leading batch dim, so IDs are (1, T), not (T,).
#
# ATTENTION MASK (the padding mask):
#   Marks which positions are real tokens (1) vs padding filler (0). Here it is
#   ALL 1s -- a single prompt has no padding, so the mask is a no-op and we ignore it.
#   It only becomes meaningful when batching sequences of DIFFERENT lengths: shorter
#   ones get padded with filler tokens to fill the rectangle, and the 0s tell attention's
#   softmax to assign zero weight to those pad positions so real tokens never attend to
#   garbage. (Relevant in the batching stage, not here.)
#
#   NOTE: this is NOT the causal mask. Two different masks:
#     - padding mask (this one): "ignore filler tokens used to pad a batch." Comes from
#       the tokenizer; all 1s for a single sequence.
#     - causal mask (triangular): "each position may only attend to itself and earlier
#       positions." Built automatically inside the decoder; nothing to do with padding.
#     Inside attention the two get combined, but they solve unrelated problems.
#
# .input_ids  -> pluck just the ID tensor (1, T), discard the (all-1s) mask.
# .to(model.device) -> move IDs from CPU (where the tokenizer ran) to the GPU (where the
#                      weights live). An op can only combine tensors on the SAME device,
#                      so input and weights must be co-located before the forward pass.
