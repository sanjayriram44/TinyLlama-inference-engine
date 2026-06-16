# TinyLlama Inference Engine

Building an LLM inference engine from scratch on **TinyLlama-1.1B**, then
optimizing it stage by stage — measuring the effect of each optimization
against a frozen benchmark. The goal is the *understanding*, not the engine:
every stage isolates one technique (KV cache, batching, quantization,
FlashAttention, tensor parallelism) and proves its effect with numbers.

> **Philosophy:** use Hugging Face only for the tokenizer and the weights
> (`model(ids) → logits`). Everything above the forward pass — the generation
> loop, caching, batching, sampling, serving — is built by hand. `.generate()`
> and vLLM are the *finish-line benchmarks*, not the starting point.

## Setup

```bash
pip install -r requirements.txt          # transformers (torch comes with the GPU image)
export HF_HOME=/path/to/persistent/cache # so the ~2.2GB download persists
python scripts/benchmark.py              # runs the current engine, writes results JSON
```

Hardware used: single **NVIDIA A40 (46 GB)**, rented hourly.

## Results

Frozen workload: 36-token prompt, 128 generated tokens, greedy decoding,
averaged over 3 runs. Same prompt and the same `golden_output.txt` correctness
anchor across every stage.

| Stage | Optimization | TTFT (ms) | TPOT (ms) | Tokens/s | Peak mem (GB) | Output matches golden |
|-------|--------------|-----------|-----------|----------|---------------|----------------------|
| 0 | Naive baseline (no cache) | 15–21 | 16–19 | 52–64 | 2.23 | — (created) |
| 2 | KV cache | _todo_ | _todo_ | _todo_ | _todo_ | must be `true` |
| 3 | Continuous batching | _todo_ | _todo_ | _todo_ | _todo_ | `true` |
| 4 | Quantization (INT8/INT4) | _todo_ | _todo_ | _todo_ | _todo_ | lossy → quality metric |
| 5 | FlashAttention + GQA | _todo_ | _todo_ | _todo_ | _todo_ | `true` |
| 6 | Tensor parallelism (2× GPU) | _todo_ | _todo_ | _todo_ | _todo_ | `true` |
| 8 | vs. vLLM | _todo_ | _todo_ | _todo_ | _todo_ | reference |

*(Stage-0 numbers shown as ranges because run-to-run GPU clock/thermal noise
moves them; this is why the harness averages over 3 runs.)*

## Metrics, defined

- **TTFT** — time to first token; dominated by **prefill** (processing the prompt). Compute-bound.
- **TPOT** — mean time per output token after the first; dominated by **decode**. Memory-bound.
- **Tokens/s** — overall generation throughput.
- **Peak mem** — `torch.cuda.max_memory_allocated`; weights + KV cache + activations.
- **Output matches golden** — exact-match correctness check vs the stage-0 output.
  Greedy is deterministic, so *lossless* stages (KV cache, batching, FlashAttn,
  TP) **must** stay `true`; the *lossy* stage (quantization) is expected to
  break it, which is the cue to switch to a quality metric.

## Stage 0 — findings

The baseline runs at ~16 ms/token (~60 tok/s) on the A40. Two findings matter
more than the headline speed:

### 1. Prefill ≈ one decode step → decode is memory-bound

Processing the **36-token prompt** (TTFT ≈ 15–20 ms) took about the **same
wall-time as generating one token** (TPOT ≈ 16–19 ms) — despite prefill doing
~36× the compute.

| Phase | Tokens processed | Relative compute | Wall-time |
|-------|------------------|------------------|-----------|
| Prefill | 36 | ~36× | ~15–20 ms |
| One decode step | 1 | 1× | ~16–19 ms |

The explanation: a decode step does almost no compute but must **stream all
2.2 GB of weights from VRAM** to produce one token. That weight-streaming —
not the math — sets the time. Prefill gets 36 tokens processed "for free"
because the GPU's compute was idle anyway. **Decode is bound by memory
bandwidth, not compute** — the central fact of LLM inference.

### 2. Per-token time is FLAT, not climbing (the surprise)

The naive loop re-feeds the entire growing sequence every step, so we
predicted per-token time would **climb** with sequence length. It didn't:

| Token index | Per-token time |
|-------------|----------------|
| 10 | 15.48 ms |
| 60 | 15.81 ms |
| 120 | 15.71 ms |

Each decode step has two costs: **(a) streaming the weights** — a fixed ~2.2 GB
regardless of length — and **(b) attention over the sequence so far** — which
grows with length. We expected (b) to dominate. At 1.1B params and ~150 tokens,
**(a) utterly dwarfs (b)**: moving 2.2 GB is enormous next to the attention
work over a few hundred tokens, so the growing cost is invisible under the
fixed weight-streaming floor.

**Implication for stage 2:** the KV cache eliminates the redundant
recomputation, but at *this* scale TPOT is pinned by weight-streaming, which
the cache doesn't touch — so the headline TPOT won't collapse. The cache's
effect becomes visible only at **long sequences**, where (b) finally climbs out
from under (a). So stage 2's real experiment is: *(i) prove the cache is
lossless (golden stays `true`), and (ii) find the sequence length where
recomputation starts to bite* — the crossover between the naive and cached
curves.

## Project structure