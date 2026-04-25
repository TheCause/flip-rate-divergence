#!/usr/bin/env python3
"""LFCM Benchmark — Logit-Faithful Context Memory.

Measures flip rate at temperature=0 for KV-cache compression methods.
Teacher-forced evaluation: same prefix at each step, compare argmax.

Usage:
    python benchmark.py --model mistral-7b --retention 0.5
    python benchmark.py --model mistral-7b --retention 0.2,0.5,0.8 --method h2o,snapkv
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import compute_flip, StepResult, ConversationResult, corpus_flip_rate


MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
}

T_DEFAULT = 128
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_synthetic_conversations(max_conversations=100):
    """Synthetic multi-turn conversations for quick testing."""
    conversations = []
    topics = [
        "quantum computing", "climate change", "artificial intelligence",
        "space exploration", "genetic engineering", "renewable energy",
        "machine learning", "cybersecurity", "blockchain technology",
        "neuroscience", "autonomous vehicles", "data privacy",
        "sustainable agriculture", "ocean conservation", "urban planning",
        "democratic governance", "economic inequality", "mental health",
        "education reform", "cultural preservation",
    ]
    for i in range(max_conversations):
        topic = topics[i % len(topics)]
        conversations.append({
            "id": f"synth_{i:04d}",
            "text": (
                f"You are a knowledgeable assistant. "
                f"User: Explain the key challenges and recent advances in {topic}. "
                f"Be specific and provide concrete examples. "
                f"Assistant: I'll provide a detailed analysis of {topic}. "
                f"User: What are the most promising research directions? "
                f"Assistant:"
            ),
        })
    return conversations


def load_sharegpt_conversations(max_conversations=100, min_tokens=500):
    """Load real conversations from ShareGPT dataset (HuggingFace).

    Filters for conversations with at least min_tokens of context.
    Returns conversations formatted as single strings.
    """
    from datasets import load_dataset

    print(f"Loading ShareGPT dataset...")
    try:
        ds = load_dataset("RyokoAI/ShareGPT52K", split="train", streaming=True)
        key_conversations = "conversations"
        key_role = "from"
        key_content = "value"
    except Exception:
        # Fallback
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        key_conversations = "conversation"
        key_role = "role"
        key_content = "content"

    conversations = []
    for item in ds:
        if len(conversations) >= max_conversations:
            break

        turns_raw = item.get(key_conversations, [])
        if not turns_raw or len(turns_raw) < 3:
            continue

        # Parse turns (may be dicts or JSON strings)
        turns = []
        for t in turns_raw:
            if isinstance(t, str):
                try:
                    import json as _json
                    turns.append(_json.loads(t))
                except (ValueError, TypeError):
                    continue
            elif isinstance(t, dict):
                turns.append(t)

        if len(turns) < 3:
            continue

        text = ""
        for turn in turns:
            role = turn.get(key_role, "human")
            content = turn.get(key_content, "")
            if role in ("human", "user"):
                text += f"User: {content} "
            elif role in ("gpt", "assistant"):
                text += f"Assistant: {content} "

        # Filter by length (rough token estimate: 1 token ~ 4 chars)
        if len(text) < min_tokens * 4:
            continue

        # Truncate to ~1K tokens worth of text to fit in 24GB VRAM
        # (eager attention on 7B model needs ~20GB for 1K context)
        max_chars = 4000  # ~1K tokens
        if len(text) > max_chars:
            text = text[:max_chars]

        conversations.append({
            "id": f"sgpt_{len(conversations):04d}",
            "text": text,
        })

    print(f"Loaded {len(conversations)} ShareGPT conversations (min ~{min_tokens} tokens)")
    return conversations


def load_mtbench_conversations(max_conversations=80):
    """Load MT-Bench multi-turn prompts from HuggingFace."""
    from datasets import load_dataset

    print("Loading MT-Bench dataset...")
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")

    conversations = []
    for item in ds:
        if len(conversations) >= max_conversations:
            break
        turns = item.get("prompt", [])
        if not turns:
            continue
        text = ""
        for i, turn in enumerate(turns):
            if i % 2 == 0:
                text += f"User: {turn} "
            else:
                text += f"Assistant: {turn} "
        text += "Assistant:"
        conversations.append({"id": f"mtb_{len(conversations):04d}", "text": text})

    print(f"Loaded {len(conversations)} MT-Bench conversations")
    return conversations


def load_longbench_conversations(max_conversations=30, max_tokens=2048):
    """Load LongBench from THUDM/LongBench (open). Cap at max_tokens."""
    from datasets import load_dataset

    print("Loading LongBench v2 dataset...")
    conversations = []
    try:
        ds = load_dataset("THUDM/LongBench-v2", split="train", streaming=True)
    except Exception as e:
        print(f"  Error loading LongBench-v2: {e}")
        return conversations

    for item in ds:
        if len(conversations) >= max_conversations:
            break
        context = item.get("context", "")
        question = item.get("question", "")
        if not context or not question:
            continue
        text = f"Context: {context}\n\nUser: {question}\nAssistant:"
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        if len(text) < 500 * 4:
            continue
        conversations.append({
            "id": f"lb_{len(conversations):04d}",
            "text": text,
        })

    print(f"Loaded {len(conversations)} LongBench conversations (cap ~{max_tokens} tokens)")
    return conversations


def compress_kv_uniform(past_key_values, retention, **kwargs):
    """Uniform token eviction: keep first r*N tokens per layer.
    Keeps attention sinks (first tokens) by design.
    """
    kv_tuples = kv_to_tuples(past_key_values)
    compressed = []
    for key, value in kv_tuples:
        seq_len = key.shape[2]
        n_keep = min(seq_len, max(1, int(seq_len * retention)))
        compressed.append((
            key[:, :, :n_keep, :].clone(),
            value[:, :, :n_keep, :].clone(),
        ))
    return tuples_to_cache(compressed)


def compress_kv_h2o(past_key_values, retention, attention_scores=None, **kwargs):
    """h2o_approx: keep sinks + tokens with highest mean prefix-attention.
    NOT a faithful H2O implementation (which uses running statistics during
    decoding); this is a static prefix-attention-topk proxy.
    """
    kv_tuples = kv_to_tuples(past_key_values)
    compressed = []
    for layer_idx, (key, value) in enumerate(kv_tuples):
        seq_len = key.shape[2]
        n_sinks = min(4, seq_len)
        n_keep = min(seq_len, max(n_sinks + 1 if seq_len > n_sinks else seq_len,
                                   int(seq_len * retention)))

        if attention_scores is not None and layer_idx < len(attention_scores):
            attn = attention_scores[layer_idx]
            importance = attn.mean(dim=(0, 1, 2))
            importance[:n_sinks] = float('inf')
            _, top_indices = importance.topk(n_keep)
            top_indices = top_indices.sort().values
            compressed.append((
                key[:, :, top_indices, :].clone(),
                value[:, :, top_indices, :].clone(),
            ))
        else:
            compressed.append((
                key[:, :, :n_keep, :].clone(),
                value[:, :, :n_keep, :].clone(),
            ))
    return tuples_to_cache(compressed)


def compress_kv_recent(past_key_values, retention, **kwargs):
    """StreamingLLM-style: keep first 4 (sinks) + last tokens."""
    kv_tuples = kv_to_tuples(past_key_values)
    compressed = []
    for key, value in kv_tuples:
        seq_len = key.shape[2]
        n_sinks = min(4, seq_len)
        n_keep = min(seq_len, max(n_sinks + 1 if seq_len > n_sinks else seq_len,
                                   int(seq_len * retention)))
        n_recent = max(0, n_keep - n_sinks)
        sink_idx = list(range(n_sinks))
        recent_start = max(n_sinks, seq_len - n_recent)
        recent_idx = list(range(recent_start, seq_len))
        indices = torch.tensor(sorted(set(sink_idx + recent_idx)), device=key.device)
        compressed.append((
            key[:, :, indices, :].clone(),
            value[:, :, indices, :].clone(),
        ))
    return tuples_to_cache(compressed)


COMPRESSION_METHODS = {
    "uniform": compress_kv_uniform,
    "h2o_approx": compress_kv_h2o,
    "recent": compress_kv_recent,
}


def kv_to_tuples(cache):
    """Convert DynamicCache to tuple of (key, value) per layer.

    Handles transformers 5.x DynamicCache which iterates as (key, value, None) tuples.
    """
    result = []
    for item in cache:
        if isinstance(item, (tuple, list)):
            # (key, value, None) or (key, value)
            result.append((item[0].clone(), item[1].clone()))
        else:
            raise ValueError(f"Unexpected KV-cache item type: {type(item)}")
    return tuple(result)


def tuples_to_cache(kv_tuples):
    """Convert tuple of (key, value) back to DynamicCache."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(kv_tuples):
        cache.update(key, value, layer_idx)
    return cache


def clone_kv(past_key_values):
    """Deep clone KV-cache (handles both DynamicCache and tuple format)."""
    tuples = kv_to_tuples(past_key_values)
    return tuples_to_cache(tuples)


def run_benchmark(model_name, methods, retentions, max_conversations=20, T=T_DEFAULT,
                   dataset="synthetic", include_steps=False, tag=None,
                   load_in_4bit=False, save_hidden=False, save_logits_topk=False,
                   indices_file=None, out_name=None, dump_env=False,
                   max_context_tokens=None):
    """Run LFCM benchmark. See PSEUDOCODE.md for detailed algorithm."""

    print(f"=== LFCM Benchmark ===")
    print(f"Model: {model_name}")
    print(f"Methods: {methods}")
    print(f"Retentions: {retentions}")
    print(f"T: {T}, Conversations: {max_conversations}")
    if load_in_4bit:
        print(f"Quantization: 4-bit (bitsandbytes)")
    print()

    model_id = MODELS[model_name]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # SDPA is memory-efficient on long contexts, but attention outputs may
    # be unavailable or unreliable for methods that require explicit
    # attention tensors (e.g. h2o_approx). Switch to eager when needed.
    needs_eager = "h2o_approx" in methods
    attn_impl = "eager" if needs_eager else "sdpa"
    if needs_eager:
        print(f"[attn] h2o_approx requested -> attn_implementation='eager' "
              f"(expect ~2-3x higher VRAM on long contexts)")
    load_kwargs = dict(
        device_map="auto",
        attn_implementation=attn_impl,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    else:
        # HF renamed torch_dtype -> dtype during the 4.x line. Try the new
        # name first; fall back to the old one on TypeError. Avoids both
        # version-guessing and deprecation warnings.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, **load_kwargs
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, **load_kwargs
            )
    model.eval()  # noqa: B010 — PyTorch eval mode, not builtin eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    # With device_map="auto", model.device is unreliable on sharded models.
    # Use the device of the first parameter (embedding) as the input device.
    input_device = next(model.parameters()).device
    print(f"Model loaded. EOS={eos_id}, Input device={input_device}")

    torch.manual_seed(42)
    try:
        torch.use_deterministic_algorithms(True)
        print("Determinism: best-effort (PyTorch deterministic algorithms enabled)")
    except RuntimeError:
        print("WARNING: Deterministic mode not available on this hardware")

    # Phase 0bis: pre-registered sub-sampling via indices file
    # If indices_file is provided, load the full dataset and pick the listed indices.
    if indices_file is not None:
        with open(indices_file, "r") as f:
            indices_meta = json.load(f)
        n_total_required = indices_meta["n_total"]
        indices = indices_meta["indices"]
        print(f"Phase 0bis: loading full {dataset} dataset ({n_total_required} convs required) "
              f"to subsample {len(indices)} pre-registered indices: {indices}")
        if dataset == "sharegpt":
            full_conversations = load_sharegpt_conversations(n_total_required)
        elif dataset == "mtbench":
            full_conversations = load_mtbench_conversations(n_total_required)
        elif dataset == "longbench":
            full_conversations = load_longbench_conversations(n_total_required)
        else:
            full_conversations = load_synthetic_conversations(n_total_required)
        if len(full_conversations) < n_total_required:
            raise ValueError(f"Dataset only has {len(full_conversations)} conversations, "
                             f"indices file requires at least {n_total_required}.")
        conversations = [full_conversations[i] for i in indices]
    else:
        if dataset == "sharegpt":
            conversations = load_sharegpt_conversations(max_conversations)
        elif dataset == "mtbench":
            conversations = load_mtbench_conversations(max_conversations)
        elif dataset == "longbench":
            conversations = load_longbench_conversations(max_conversations)
        else:
            conversations = load_synthetic_conversations(max_conversations)
    print(f"Loaded {len(conversations)} conversations\n")

    # Phase 0bis: dump environment for reproducibility
    if dump_env:
        try:
            import transformers as _tx
        except ImportError:
            _tx = None
        env = {
            "model_name_cli": model_name,
            "model_id": model_id,
            "dataset": dataset,
            "indices_file": indices_file,
            "n_conversations": len(conversations),
            "T": T,
            "methods": methods,
            "retentions": retentions,
            "load_in_4bit": load_in_4bit,
            "save_hidden": save_hidden,
            "save_logits_topk": save_logits_topk,
            "torch_version": torch.__version__,
            "transformers_version": getattr(_tx, "__version__", "unknown") if _tx else "unknown",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": "bfloat16" if not load_in_4bit else "nf4",
            "seed_torch": 42,
            "deterministic": True,
        }
        env_path = RESULTS_DIR / f"env_{out_name or model_name}.json"
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2, default=str)
        print(f"Environment dumped to {env_path}")

    all_results = []
    t0 = time.time()

    for conv_idx, conv in enumerate(conversations):
        print(f"--- [{conv_idx+1}/{len(conversations)}] {conv['id']} ---")

        inputs = tokenizer(conv["text"], return_tensors="pt").to(input_device)
        input_ids = inputs["input_ids"]
        if max_context_tokens is not None and input_ids.shape[1] > max_context_tokens:
            original_len = input_ids.shape[1]
            input_ids = input_ids[:, :max_context_tokens]
            inputs["input_ids"] = input_ids
            print(f"  Truncated context {original_len} -> {max_context_tokens} tokens")
        N = input_ids.shape[1]

        if N < 2:
            print(f"  Skipping {conv['id']} (context too short: {N})")
            continue

        # Log context length and GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  Context: {N} tokens, GPU mem: {gpu_mem:.1f} GB")
        else:
            print(f"  Context: {N} tokens")

        need_attn = "h2o_approx" in methods

        with torch.no_grad():
            # Phase 0: Encode context prefix (all but last token) + generate reference
            # The last context token is used as the first input for generation,
            # so the KV-cache covers positions 0..N-2 and the first forward pass
            # produces logits for position N-1 (the true "next token after context").
            prefix = input_ids[:, :-1]       # tokens 0..N-2
            last_ctx = input_ids[:, -1:]     # token N-1

            outputs = model(prefix, use_cache=True, output_attentions=need_attn)
            kv_ctx = outputs.past_key_values  # cache for positions 0..N-2
            attn_ctx = outputs.attentions if need_attn else None
            if need_attn and attn_ctx is not None:
                print(f"  [attn_ctx] captured: {len(attn_ctx)} layers, shape[0]={tuple(attn_ctx[0].shape)}")

            # First generation step: forward last context token with prefix cache
            kv_gen = clone_kv(kv_ctx)
            out_0 = model(last_ctx, past_key_values=kv_gen, use_cache=True,
                          output_hidden_states=save_hidden)
            kv_gen = out_0.past_key_values
            logits_ref = [out_0.logits[0, -1, :].float()]
            hidden_ref = [out_0.hidden_states[-1][0, -1, :].float() if save_hidden else None]
            x_ref = [logits_ref[0].argmax().item()]

            # Autoregressive generation for steps 1..T-1
            for t in range(1, T):
                inp = torch.tensor([[x_ref[t-1]]], device=input_device)
                out = model(inp, past_key_values=kv_gen, use_cache=True,
                            output_hidden_states=save_hidden)
                kv_gen = out.past_key_values
                logits_t = out.logits[0, -1, :].float()
                logits_ref.append(logits_t)
                hidden_ref.append(out.hidden_states[-1][0, -1, :].float() if save_hidden else None)
                x_ref.append(logits_t.argmax().item())
                if x_ref[-1] == eos_id:
                    break

            T_actual = len(x_ref)
            print(f"  Generated {T_actual} tokens (ref)")

            # Phase 1: Evaluate each (method, retention)
            for method_name in methods:
                compress_fn = COMPRESSION_METHODS[method_name]
                for r in retentions:
                    # Compress the context cache (prefix only, not including last_ctx)
                    kv_comp = compress_fn(kv_ctx, r, attention_scores=attn_ctx)
                    kv_comp_gen = clone_kv(kv_comp)

                    # First step: forward last context token with compressed cache
                    out_comp_0 = model(last_ctx, past_key_values=kv_comp_gen, use_cache=True,
                                       output_hidden_states=save_hidden)
                    kv_comp_gen = out_comp_0.past_key_values

                    h_comp_0 = out_comp_0.hidden_states[-1][0, -1, :].float() if save_hidden else None
                    step_0 = compute_flip(
                        logits_ref[0],
                        out_comp_0.logits[0, -1, :].float(),
                        eos_id,
                        hidden_full=hidden_ref[0],
                        hidden_comp=h_comp_0,
                        save_logits_topk=save_logits_topk,
                    )
                    step_0.t = 0
                    steps = [step_0]

                    # Teacher-forced steps 1..T_actual-1: same prefix tokens as reference
                    for t in range(1, T_actual):
                        inp = torch.tensor([[x_ref[t-1]]], device=input_device)
                        out_comp = model(inp, past_key_values=kv_comp_gen, use_cache=True,
                                         output_hidden_states=save_hidden)
                        kv_comp_gen = out_comp.past_key_values

                        h_comp_t = out_comp.hidden_states[-1][0, -1, :].float() if save_hidden else None
                        step = compute_flip(
                            logits_ref[t],
                            out_comp.logits[0, -1, :].float(),
                            eos_id,
                            hidden_full=hidden_ref[t],
                            hidden_comp=h_comp_t,
                            save_logits_topk=save_logits_topk,
                        )
                        step.t = t
                        steps.append(step)

                    result = ConversationResult(
                        conversation_id=conv["id"],
                        method=method_name,
                        retention=r,
                        steps=steps,
                    )
                    all_results.append(result)

                    n_flips = sum(s.flip for s in steps)
                    if n_flips > 0:
                        print(f"  {method_name} r={r}: Phi={result.flip_rate:.3f} "
                              f"({n_flips}/{T_actual}) FFP={result.first_flip_position} "
                              f"KL={result.kl_mean:.4f}")
                    else:
                        print(f"  {method_name} r={r}: Phi=0 (transparent) KL={result.kl_mean:.4f}")

        del kv_ctx, kv_gen, logits_ref, hidden_ref, x_ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t0
    print(f"\n=== DONE in {total_time:.0f}s ===\n")

    # Phase 2: Aggregate
    print("=== Corpus Results ===")
    print(f"{'Method':<10} {'Ret':>4} {'Phi_mean':>8} {'Phi_std':>8} {'ZFR':>5} {'KL_mean':>10}")
    print("=" * 55)

    summary = {}
    for method_name in methods:
        for r in retentions:
            subset = [res for res in all_results
                      if res.method == method_name and res.retention == r]
            agg = corpus_flip_rate(subset)
            key = f"{method_name}_r{r}"
            summary[key] = agg
            print(f"{method_name:<10} {r:>4.1f} {agg['flip_rate_mean']:>8.4f} "
                  f"{agg['flip_rate_std']:>8.4f} {agg['zero_flip_rate']:>5.2f} "
                  f"{agg['kl_mean']:>10.6f}")

    output = {
        "model": model_name,
        "dataset": dataset,
        "methods": methods,
        "retentions": retentions,
        "n_conversations": len(conversations),
        "T": T,
        "summary": summary,
        "details": [r.to_dict(include_steps=include_steps) for r in all_results],
        "timing_s": total_time,
    }
    if out_name is not None:
        out_path = RESULTS_DIR / f"{out_name}.json"
    else:
        suffix = f"_{tag}" if tag else ""
        out_path = RESULTS_DIR / f"lfcm_{model_name}_{dataset}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFCM Benchmark")
    parser.add_argument("--model", default="mistral-7b", choices=list(MODELS.keys()))
    parser.add_argument("--methods", default="uniform,h2o_approx,recent")
    parser.add_argument("--retention", default="0.2,0.5,0.8")
    parser.add_argument("--conversations", type=int, default=20)
    parser.add_argument("--T", type=int, default=T_DEFAULT)
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "sharegpt", "mtbench", "longbench"])
    parser.add_argument("--include-steps", action="store_true",
                        help="Include per-step token IDs in output JSON")
    parser.add_argument("--tag", default=None,
                        help="Tag for output filename")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (bitsandbytes NF4)")
    # Phase 0bis: capture flags
    parser.add_argument("--save-hidden", action="store_true",
                        help="Capture pre-LM-head hidden states (Phase 0bis rigidity test)")
    parser.add_argument("--save-logits-topk", action="store_true",
                        help="Capture top-32 log-softmax logits (Phase 0bis)")
    parser.add_argument("--indices-file", default=None,
                        help="JSON file with pre-registered conversation indices (Phase 0bis)")
    parser.add_argument("--out-name", default=None,
                        help="Custom output filename (without .json extension). Overrides default naming.")
    parser.add_argument("--dump-env", action="store_true",
                        help="Dump environment metadata (versions, seeds, model_id) to env_<out_name>.json")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Truncate input contexts longer than this many tokens. "
                             "Required for h2o_approx on real-world datasets (eager attn VRAM).")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        methods=args.methods.split(","),
        retentions=[float(r) for r in args.retention.split(",")],
        max_conversations=args.conversations,
        T=args.T,
        dataset=args.dataset,
        include_steps=args.include_steps,
        tag=args.tag,
        load_in_4bit=args.load_in_4bit,
        save_hidden=args.save_hidden,
        save_logits_topk=args.save_logits_topk,
        indices_file=args.indices_file,
        out_name=args.out_name,
        dump_env=args.dump_env,
        max_context_tokens=args.max_context_tokens,
    )
