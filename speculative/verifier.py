"""
Transformer Verifier — Qwen-0.5B based draft token verifier.

Two verification modes:
  (a) verify()          — single forward pass over full (context+draft),
                          no state.  Simple but slower: O(full_seq^2) attention.
  (b) init_kv_cache() + verify_step()
                        — KV-cached pipeline.  Context is encoded once;
                          each speculative step only processes draft tokens
                          with the cached context.  Timing is fair vs baseline.

Device priority: CUDA → MPS (Apple Silicon) → CPU.
Temperature is always 0.0 (greedy / argmax) for reproducibility.
"""

import logging
from typing import Any, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_DEFAULT_VERIFIER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Type alias for HuggingFace KV cache (tuple of (k, v) tensors per layer)
KVCache = Any


class TransformerVerifier:
    """Wraps a small causal LM for greedy (temperature=0) draft verification."""

    def __init__(
        self,
        model_name: str = _DEFAULT_VERIFIER_MODEL,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.device = device or _auto_device()
        self.dtype = dtype or _auto_dtype(self.device)
        # temperature=0.0 → greedy (argmax), fully reproducible
        self.temperature = temperature

        logger.info(
            "Loading verifier %s on %s (%s) ...",
            model_name, self.device, self.dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Verifier ready.")

    # ------------------------------------------------------------------
    # KV-cached pipeline  (use these for accurate speedup measurement)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_kv_cache(
        self, context_ids: List[int]
    ) -> Tuple[KVCache, torch.Tensor]:
        """
        Encode the full context and return its KV cache + last-position logits.

        Call this once at generation start; then pass the returned values to
        verify_step() for each speculative step.

        Returns:
            past_kv:     KV cache for the context (seq_len = len(context_ids)).
            last_logits: 1-D tensor (vocab_size,) predicting the *first* new token.
        """
        ctx = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ctx, use_cache=True)
        # out.logits: (1, ctx_len, vocab)  →  last position predicts next token
        return out.past_key_values, out.logits[0, -1]

    @torch.no_grad()
    def verify_step(
        self,
        draft_ids: List[int],
        past_kv: KVCache,
        last_logits: torch.Tensor,
    ) -> Tuple[List[int], int, KVCache, torch.Tensor]:
        """
        KV-cached speculative verification of `draft_ids`.

        Runs ONE forward pass over the draft tokens (context already cached in
        past_kv), making it O(draft_len × ctx_len) instead of O((ctx+draft)^2).

        Acceptance rule: greedy (temperature=0) — accept draft[i] iff
        argmax(pred_logits_at_i) == draft[i].

        Returns:
            accepted:       Accepted draft token ids.
            bonus_token:    Verifier's preferred token at first rejection
                            (or after full acceptance).
            new_past_kv:    Updated KV cache (includes accepted + bonus token).
            new_last_logits: Logits predicting the next token after bonus.
        """
        if not draft_ids:
            bonus = int(last_logits.argmax().item())
            new_past_kv, new_last_logits = self._extend_kv(
                past_kv, [bonus]
            )
            return [], bonus, new_past_kv, new_last_logits

        # Single forward pass: draft tokens attend to cached context
        draft_t = torch.tensor([draft_ids], dtype=torch.long, device=self.device)
        draft_out = self.model(
            input_ids=draft_t, past_key_values=past_kv, use_cache=True
        )
        # draft_logits[i] predicts the token *after* draft[i]
        draft_logits = draft_out.logits[0]  # (draft_len, vocab)
        full_kv = draft_out.past_key_values  # KV for ctx + draft

        ctx_kv_len = _kv_seq_len(past_kv)

        accepted: List[int] = []
        for i, dtok in enumerate(draft_ids):
            # Logits that predict draft[i]:
            #   i == 0 → last_logits (the last context position)
            #   i  > 0 → draft_logits[i-1] (draft[i-1] position)
            pred_logits = last_logits if i == 0 else draft_logits[i - 1]
            greedy_tok = int(pred_logits.argmax().item())

            if greedy_tok == draft_ids[i]:
                accepted.append(dtok)
            else:
                # Rejection: greedy_tok is the bonus
                bonus = greedy_tok
                # Slice KV to context + accepted positions
                sliced_kv = _slice_kv(full_kv, ctx_kv_len + len(accepted))
                # Append bonus token to get new KV + next logits
                new_past_kv, new_last_logits = self._extend_kv(sliced_kv, [bonus])
                return accepted, bonus, new_past_kv, new_last_logits

        # All draft tokens accepted; bonus from last draft-position logits
        bonus = int(draft_logits[-1].argmax().item())
        # full_kv already contains ctx + all draft tokens; append bonus
        new_past_kv, new_last_logits = self._extend_kv(full_kv, [bonus])
        return accepted, bonus, new_past_kv, new_last_logits

    @torch.no_grad()
    def _extend_kv(
        self, past_kv: KVCache, new_ids: List[int]
    ) -> Tuple[KVCache, torch.Tensor]:
        """Append `new_ids` to KV cache; return updated cache + last logits."""
        tok_t = torch.tensor([new_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=tok_t, past_key_values=past_kv, use_cache=True)
        return out.past_key_values, out.logits[0, -1]

    # ------------------------------------------------------------------
    # Simple verification (no KV cache — for correctness testing only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify(
        self,
        context_ids: List[int],
        draft_ids: List[int],
    ) -> Tuple[List[int], int]:
        """
        Stateless verification: one forward pass over context + draft.

        NOTE: This re-encodes the full context every call (no KV cache).
        Use verify_step() + init_kv_cache() for fair timing benchmarks.
        """
        if not draft_ids:
            return [], self._greedy_next(context_ids)

        full_ids = context_ids + draft_ids
        input_t = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        logits = self.model(input_ids=input_t).logits[0]  # (seq_len, vocab)

        accepted: List[int] = []
        ctx_len = len(context_ids)
        for i, dtok in enumerate(draft_ids):
            pred = int(logits[ctx_len - 1 + i].argmax().item())
            if pred == dtok:
                accepted.append(dtok)
            else:
                return accepted, pred

        bonus = int(logits[ctx_len - 1 + len(draft_ids)].argmax().item())
        return accepted, bonus

    # ------------------------------------------------------------------
    # Baseline autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_baseline(
        self,
        context_ids: List[int],
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """
        Pure autoregressive generation (KV-cached internally by `model.generate`).

        Uses greedy decoding (do_sample=False, temperature=0) for reproducibility.
        """
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        eos = eos_token_id or self.tokenizer.eos_token_id

        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # temperature=0, reproducible
            temperature=1.0,       # ignored when do_sample=False
            eos_token_id=eos,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return out[0, len(context_ids):].tolist()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _greedy_next(self, context_ids: List[int]) -> int:
        input_t = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        return int(self.model(input_ids=input_t).logits[0, -1].argmax().item())

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


# ------------------------------------------------------------------
# KV cache utilities
# ------------------------------------------------------------------

def _kv_seq_len(past_kv: KVCache) -> int:
    """Return sequence length stored in the KV cache."""
    # past_kv: tuple of (key, value) tensors, each (batch, heads, seq_len, d)
    return past_kv[0][0].shape[2]


def _slice_kv(past_kv: KVCache, seq_len: int) -> KVCache:
    """Slice KV cache to keep only the first `seq_len` positions."""
    return tuple(
        (k[:, :, :seq_len, :], v[:, :, :seq_len, :])
        for k, v in past_kv
    )


# ------------------------------------------------------------------
# Device / dtype helpers
# ------------------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_dtype(device: str) -> torch.dtype:
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32
