# TinyLlama Inference Engine

An LLM inference engine built from scratch on **TinyLlama-1.1B-Chat**, optimized
stage by stage, with each optimization's effect measured against a frozen
benchmark. The goal is *understanding*, not the engine: every stage isolates one
technique and proves its effect with numbers.

> **Philosophy:** Hugging Face is used only for the tokenizer and the weights
> (`model(ids) → logits`). Everything above the forward pass — the generation
> loop, KV caching, batching, sampling, serving — is built by hand.
> `.generate()` and vLLM are the *finish-line benchmarks*, not the starting point.

## Setup

```bash
pip install -r requirements.txt          # transformers (torch ships with the GPU image)
export HF_HOME=/path/to/persistent/cache # so the ~2.2GB download persists
python scripts/benchmark.py              # run the current single-stream engine
python scripts/sweep_batch.py            # run the batch-throughput sweep
```

Hardware: single **NVIDIA A40 (46 GB)**, rented hourly.

## Results

Frozen single-stream workload: 36-token prompt, 128 generated tokens, greedy,
averaged over 3 runs. Same prompt and `golden_output.txt` correctness anchor
across every lossless stage.

| Stage | Optimization | TTFT (ms) | TPOT (ms) | Tokens/s | Peak mem (GB) | Matches golden |
|-------|--------------|-----------|-----------|----------|---------------|----------------|
| 0 | Naive baseline (no cache) | 15–21 | 16–19 | 52–64 | 2.23 | created |
| 2 | KV cache | ~15 | ~13 | 74–77 | 2.22 | **true** |
| 3 | Static batching | — | — | up to ~7900 (batch) | 4.7 @ b512 | true |

*(Stage-0/2 numbers shown as ranges — run-to-run GPU clock/thermal noise moves
them, which is why the harness averages over 3 runs.)*

## Metrics, defined

- **TTFT** — time to first token; dominated by **prefill**. Compute-bound.
- **TPOT** — mean time per output token after the first; dominated by **decode**. Memory-bound.
- **Tokens/s** — single-stream throughput (stages 0/2); aggregate batch throughput (stage 3).
- **Peak mem** — `torch.cuda.max_memory_allocated`; weights + KV cache + activations.
- **Output matches golden** — exact-match correctness vs the stage-0 output. Greedy
  is deterministic, so *lossless* stages (KV cache, batching) **must** stay `true`;
  the *lossy* stage (quantization) is expected to break it — the cue to switch to a
  quality metric.

---

## Stage 0 — Naive baseline + harness

The baseline runs at ~16 ms/token (~60 tok/s). Two findings matter more than the speed.

### Finding 1 — Prefill ≈ one decode step → decode is memory-bound

Processing the **36-token prompt** (TTFT ≈ 15–20 ms) took about the **same
wall-time as generating one token** (TPOT ≈ 16–19 ms), despite ~36× the compute.

| Phase | Tokens | Relative compute | Wall-time |
|-------|--------|------------------|-----------|
| Prefill | 36 | ~36× | ~15–20 ms |
| One decode step | 1 | 1× | ~16–19 ms |

A decode step does almost no compute but must **stream all 2.2 GB of weights
from VRAM** to produce one token — that streaming, not the math, sets the time.
**Decode is bound by memory bandwidth, not compute** — the central fact of LLM
inference, and the lens for every stage that follows.

### Finding 2 — Per-token time is FLAT, not climbing

The naive loop re-feeds the whole growing sequence each step, so per-token time
was predicted to climb. It didn't:

| Token index | 10 | 60 | 120 |
|-------------|------|------|------|
| Time (ms) | 15.48 | 15.81 | 15.71 |

Each step costs (a) streaming the weights — fixed ~2.2 GB regardless of length —
plus (b) attention over the sequence so far — which grows. At 1.1B params and
~150 tokens, **(a) dwarfs (b)**: the growing cost is invisible under the fixed
weight-streaming floor. Implication: a KV cache won't collapse TPOT at this
scale; its payoff appears at long sequences.

## Stage 1 — Profiling

Not a code stage — reading the stage-0 numbers produced the two findings above
(the memory-bound diagnosis and the flat-curve surprise). The profiler
(`scripts/profile_tpot.py`) dumps per-token timings to expose the curve.

## Stage 2 — KV cache (lossless)

Prefill the prompt once; each decode step feeds **only the new token** plus the
carried-forward cache, attending against stored K/V instead of recomputing them.

**Correctness is the deliverable, and it passed:** `output_matches_golden`
stayed `true` — byte-identical to the naive baseline, proving the cache is
lossless. (A mismatch would have signalled an off-by-one or a feed-the-wrong-
tensor bug. None occurred.)

| Metric | Naive | KV cache | Change |
|--------|-------|----------|--------|
| TPOT | ~17 ms | ~13 ms | ~25% faster |
| Tokens/s | ~57 | ~77 | ~35% higher |

The modest speedup confirms the stage-0 finding from the other side: TPOT is
pinned by weight-streaming, which the cache doesn't touch — it only removes the
(small, at this scale) recompute cost. **You can't optimize away a cost that
isn't your bottleneck.** The cache's real payoff is at long sequences.

## Stage 3 — Static batching (lossless)

Many varied-length prompts, **left-padded** (so each sequence's last real token
lands in the final column where `logits[:, -1, :]` reads) with a growing
attention mask, run through one KV-cached decode loop. Metric shifts from
per-token latency to **aggregate throughput** = (batch × tokens) / wall time.

### Throughput sweep (A40, 128 tokens/seq)

| Batch | Throughput (tok/s) | Per-seq (tok/s) | Wall time (s) | Peak mem (GB) |
|-------|--------------------|-----------------|---------------|---------------|
| 1 | 60 | 60 | 2.1 | 2.21 |
| 8 | 533 | 67 | 1.9 | 2.25 |
| 32 | 1693 | 53 | 2.4 | 2.37 |
| 64 | 3544 | 55 | 2.3 | 2.53 |
| 128 | 6288 | 49 | 2.6 | 2.84 |
| 256 | 7621 | 30 | 4.3 | 3.47 |
| 512 | 7871 | 15 | 8.3 | 4.73 |

### Finding — batching is nearly free until the knee at ~128

Throughput scales **linearly up to batch ~128** (32→64→128 roughly doubles each
step, wall time stays flat ~2.3–2.6 s — extra sequences are *free*), then the
A40 **saturates at ~7900 tok/s**. Past the knee: throughput barely moves
(128→256 only 1.21×, 256→512 just 1.03×), per-sequence speed collapses
(49→30→15 tok/s), and wall time scales linearly with batch (4.3 s, 8.3 s) — the
signature of compute-bound.

**The knee at ~128 is the memory-bound → compute-bound transition, found
empirically.** Below it, decode streams the weights once per step while compute
units sit idle, so added sequences fill that idle capacity for nothing. Above
it, compute is saturated — every added sequence waits its turn. Confirmed live:
`nvidia-smi` util climbed from ~50% (small batch, starved) toward ~100% (large
batch). Memory stayed tiny throughout (4.7 GB of 46 GB), so this knee is
compute/bandwidth-driven, not memory-driven.

**Serving tradeoff:** aggregate throughput (serve more users) and per-sequence
latency (each user's speed) are in direct tension past the knee — the core
serving decision. Static batching's flaw (all sequences wait for the slowest) is
what **continuous batching** fixes next.

## Project structure

pending