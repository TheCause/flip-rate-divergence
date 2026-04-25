# Code — FlipRateDivergence

Reproducibility code for **"FlipRateDivergence: A Pre-registered Null Report on the 'Two Regimes' Hypothesis for KV-Cache Compression Divergence in LLMs"**.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+. CUDA 12.8, torch 2.10+, transformers 4.57+ for the benchmark stage. Analysis stage runs on CPU.

## Files

| File | Description |
|------|-------------|
| `benchmark.py` | Phase 2 runner. Loads a model in 4-bit, applies a KV-eviction method (uniform, recent, h2o_approx), generates T=128 tokens per conversation, dumps per-step token IDs and flip flags. |
| `metrics.py` | flip_rate, PDP_2, KL_mean primitives. Imported by `benchmark.py`. Compression methods (uniform, recent, h2o_approx) defined here at lines 199–266. |
| `analyze_phase2.py` | H1/H2/H3 verdict computation per the pre-registered contract v0.4. Bonferroni-Spearman for H1, KS-test per retention for H2, grouped 5-fold CV (group=conv_id, 30 repeats) for H3. Outputs `phase2_final_analysis.json`. |
| `requirements.txt` | Python dependencies (top-level pins). |

## Reproducing Phase 2 (benchmark stage)

Per-model runs (example, mistral-7B on synthetic):

```bash
python benchmark.py \
  --model mistral-7b \
  --methods uniform,recent,h2o_approx \
  --retention 0.2,0.5,0.8,0.9 \
  --conversations 50 --T 128 --dataset synthetic \
  --dump-env --include-steps \
  --tag phase2_mistral_synth
```

Repeat for `qwen-7b` and `qwen-14b`, datasets `synthetic` and `sharegpt`. Outputs go to the working directory and should be moved into `../results/` for the analysis stage.

## Reproducing the verdicts (analysis stage)

```bash
python analyze_phase2.py \
  --tag-prefix phase2 \
  --out phase2_final_analysis.json
```

Reads all `lfcm_*_phase2_*.json` files under `../results/`. Outputs the global H1/H2/H3 verdicts and the outcome gate. Pre-computed outputs are provided in `../results/phase2_final_analysis.json` and `phase2_final_analysis.log`.

## Hardware

- Benchmark stage: RTX 3090 (24 GB VRAM), CUDA 12.8. ShareGPT runs use `max_context_tokens=1024` (mistral) or `512` (qwen-14b) to fit in 24 GB.
- Analysis stage: any machine with Python 3.10+.

## Pre-registration

The contract `../contract/contract_validation.md` (v0.4) was frozen in git (`phase2_contract_v0.4_frozen` tag in the parent `RQZ_papers` repo) before any Phase 2 data was collected. The decision rules implemented in `analyze_phase2.py` follow the contract verbatim.

## License

CC-BY-4.0
