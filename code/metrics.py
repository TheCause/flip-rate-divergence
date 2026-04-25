"""LFCM Metrics — Logit-Faithful Context Memory.

Implements the flip rate metric for evaluating KV-cache compression fidelity.
All metrics derive from the divergence profile Δ(t) = log p_full - log p_comp.

Reference: LFCM paper Section 3 (Rigaud, 2026).
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepResult:
    """Metrics for a single generation step.

    Optional fields hidden_full/hidden_comp and logits_topk_full/logits_topk_comp
    are populated only when Phase 0bis capture flags are active; they remain None
    in the default LFCM V2 pipeline and do not affect the existing JSON schema.
    """
    t: int
    flip: bool
    kl: float
    token_full: int
    token_comp: int
    eos_flip: bool = False
    hidden_full: Optional[List[float]] = None
    hidden_comp: Optional[List[float]] = None
    logits_topk_full: Optional[List[List[float]]] = None
    logits_topk_comp: Optional[List[List[float]]] = None


@dataclass
class ConversationResult:
    """Aggregated metrics for one conversation."""
    conversation_id: str
    method: str
    retention: float
    steps: List[StepResult]

    @property
    def T(self) -> int:
        return len(self.steps)

    @property
    def flip_rate(self) -> float:
        """Eq. 9: Φ = (1/T) Σ flip(t)"""
        if not self.steps:
            return 0.0
        return sum(s.flip for s in self.steps) / len(self.steps)

    @property
    def first_flip_position(self) -> float:
        """Eq. 10: FFP = min{t >= 1 : flip(t) = 1} (1-indexed)"""
        for s in self.steps:
            if s.flip:
                return s.t + 1  # convert 0-indexed to 1-indexed per Definition 3
        return float('inf')

    @property
    def ffp_normalized(self) -> float:
        """FFP / T for cross-experiment comparability."""
        ffp = self.first_flip_position
        if ffp == float('inf'):
            return float('inf')
        return ffp / self.T

    @property
    def kl_mean(self) -> float:
        """D_mean = (1/T) Σ D(t)"""
        if not self.steps:
            return 0.0
        return sum(s.kl for s in self.steps) / len(self.steps)

    @property
    def kl_max(self) -> float:
        """D_max = max_t D(t)"""
        if not self.steps:
            return 0.0
        return max(s.kl for s in self.steps)

    @property
    def eos_flip_count(self) -> int:
        return sum(s.eos_flip for s in self.steps)

    @property
    def zero_flip(self) -> bool:
        """True if no flip occurred (Φ = 0)."""
        return self.flip_rate == 0.0

    def to_dict(self, include_steps=False) -> dict:
        d = {
            "conversation_id": self.conversation_id,
            "method": self.method,
            "retention": self.retention,
            "T": self.T,
            "flip_rate": self.flip_rate,
            "ffp": self.first_flip_position,
            "ffp_normalized": self.ffp_normalized,
            "kl_mean": self.kl_mean,
            "kl_max": self.kl_max,
            "eos_flip_count": self.eos_flip_count,
            "zero_flip": self.zero_flip,
        }
        if include_steps:
            d["steps"] = []
            for s in self.steps:
                step_d = {
                    "t": s.t, "flip": s.flip, "kl": s.kl,
                    "token_full": s.token_full, "token_comp": s.token_comp,
                    "eos_flip": s.eos_flip,
                }
                if s.hidden_full is not None:
                    step_d["hidden_full"] = s.hidden_full
                    step_d["hidden_comp"] = s.hidden_comp
                if s.logits_topk_full is not None:
                    step_d["logits_topk_full"] = s.logits_topk_full
                    step_d["logits_topk_comp"] = s.logits_topk_comp
                d["steps"].append(step_d)
        return d


def compute_flip(logits_full: torch.Tensor, logits_comp: torch.Tensor,
                 eos_token_id: int,
                 hidden_full: Optional[torch.Tensor] = None,
                 hidden_comp: Optional[torch.Tensor] = None,
                 save_logits_topk: bool = False,
                 topk: int = 32) -> StepResult:
    """Compute flip and KL for a single generation step.

    Args:
        logits_full: (V,) logits from full context, float32
        logits_comp: (V,) logits from compressed context, float32
        eos_token_id: token ID for EOS
        hidden_full: (d_model,) pre-LM-head hidden state for reference (optional, Phase 0bis)
        hidden_comp: (d_model,) pre-LM-head hidden state for compressed (optional, Phase 0bis)
        save_logits_topk: if True, store top-k log-softmax for both trajectories
        topk: number of top tokens to store (default 32)

    Returns:
        StepResult with flip, kl, tokens, eos_flip (+ optional hidden states and top-k logits)
    """
    # Eq. 8: Flip (tie-breaking by lowest index via argmax behavior)
    token_full = logits_full.argmax().item()
    token_comp = logits_comp.argmax().item()
    flip = (token_full != token_comp)

    # EOS handling
    eos_flip = (token_full == eos_token_id) != (token_comp == eos_token_id)

    # Eq. 11: KL divergence via log-softmax (float32, numerically stable)
    log_p_full = F.log_softmax(logits_full, dim=-1)
    log_p_comp = F.log_softmax(logits_comp, dim=-1)
    p_full = log_p_full.exp()
    kl = F.kl_div(log_p_comp, p_full, reduction='sum', log_target=False).item()

    # Phase 0bis: optional hidden state capture (float16 to keep JSON small)
    hidden_full_list = hidden_full.to(torch.float16).cpu().tolist() if hidden_full is not None else None
    hidden_comp_list = hidden_comp.to(torch.float16).cpu().tolist() if hidden_comp is not None else None

    # Phase 0bis: optional top-k log-softmax capture
    topk_full = None
    topk_comp = None
    if save_logits_topk:
        vals_f, idx_f = torch.topk(log_p_full, topk)
        vals_c, idx_c = torch.topk(log_p_comp, topk)
        topk_full = [[int(i), float(v)] for i, v in zip(idx_f.cpu().tolist(), vals_f.cpu().tolist())]
        topk_comp = [[int(i), float(v)] for i, v in zip(idx_c.cpu().tolist(), vals_c.cpu().tolist())]

    return StepResult(
        t=0,  # caller sets this
        flip=flip,
        kl=kl,
        token_full=token_full,
        token_comp=token_comp,
        eos_flip=eos_flip,
        hidden_full=hidden_full_list,
        hidden_comp=hidden_comp_list,
        logits_topk_full=topk_full,
        logits_topk_comp=topk_comp,
    )


def corpus_flip_rate(results: List[ConversationResult]) -> dict:
    """Eq. 10: Corpus-level aggregation.

    Returns dict with corpus flip rate, std, zero-flip rate, etc.
    """
    if not results:
        return {}

    phis = [r.flip_rate for r in results]
    ffps = [r.first_flip_position for r in results if r.first_flip_position != float('inf')]
    kls = [r.kl_mean for r in results]

    n = len(phis)
    phi_mean = sum(phis) / n
    phi_std = (sum((p - phi_mean)**2 for p in phis) / max(1, n - 1)) ** 0.5  # sample std (n-1)
    zero_flip_rate = sum(1 for r in results if r.zero_flip) / n

    return {
        "n_conversations": n,
        "flip_rate_mean": phi_mean,
        "flip_rate_std": phi_std,
        "zero_flip_rate": zero_flip_rate,
        "ffp_mean": sum(ffps) / len(ffps) if ffps else float('inf'),
        "ffp_median": sorted(ffps)[len(ffps)//2] if ffps else float('inf'),
        "kl_mean": sum(kls) / n,
        "kl_max": max(r.kl_max for r in results),
        "eos_flip_total": sum(r.eos_flip_count for r in results),
    }
