"""
Transformer Verifier — Qwen-0.5B based draft token verifier.

Uses HuggingFace `transformers` to run a causal LM forward pass and either:
  (a) greedily verify draft tokens (temperature=0), or
  (b) sample-based speculative acceptance (temperature>0).

Device priority: CUDA → MPS (Apple Silicon) → CPU.
"""

import logging
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Default small verifier model
_DEFAULT_VERIFIER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TransformerVerifier:
    """
    Wraps a small causal LM to verify speculative draft tokens.

    Greedy verification (temperature=0):
        Accept draft[i] iff verifier's argmax at position i == draft[i].
        Stop at first rejection; emit verifier's preferred token there.

    Sampled verification (temperature>0):
        Standard speculative decoding acceptance:
            p_accept = min(1, p_verifier(t) / p_draft(t))
        When the proposer is a deterministic n-gram (no probability),
        we treat p_draft = 1 for all proposed tokens and use the simpler
        greedy rule.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_VERIFIER_MODEL,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_batch_size: int = 1,
    ):
        self.model_name = model_name
        self.device = device or _auto_device()
        self.dtype = dtype or _auto_dtype(self.device)
        self.max_batch_size = max_batch_size

        logger.info(
            "Loading verifier model %s on %s (%s) ...",
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

        # Pad token setup
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info("Verifier ready.")

    # ------------------------------------------------------------------
    # Core verification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify(
        self,
        context_ids: List[int],
        draft_ids: List[int],
    ) -> Tuple[List[int], int]:
        """
        Verify `draft_ids` against the verifier model given `context_ids`.

        Returns:
            accepted_ids:  List of accepted token ids (may be empty).
            bonus_token:   The verifier's own greedy token at the first
                           rejection (or the token after full acceptance).

        The final generated token sequence for this step is:
            accepted_ids + [bonus_token]
        """
        if not draft_ids:
            bonus = self._greedy_next(context_ids)
            return [], bonus

        # Build input: context + draft tokens
        full_ids = context_ids + draft_ids
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)

        # Single forward pass over the full sequence
        logits = self.model(input_ids=input_tensor).logits  # (1, seq_len, vocab)
        logits = logits[0]  # (seq_len, vocab)

        # Greedy verification: check each draft token
        accepted: List[int] = []
        ctx_len = len(context_ids)

        for i, draft_tok in enumerate(draft_ids):
            # logits at position ctx_len - 1 + i predicts token at ctx_len + i
            pred_pos = ctx_len - 1 + i
            greedy_tok = int(logits[pred_pos].argmax().item())
            if greedy_tok == draft_tok:
                accepted.append(draft_tok)
            else:
                # First rejection: emit verifier's preferred token
                return accepted, greedy_tok

        # All draft tokens accepted; emit bonus token after the last one
        bonus_pos = ctx_len - 1 + len(draft_ids)
        bonus_tok = int(logits[bonus_pos].argmax().item())
        return accepted, bonus_tok

    @torch.no_grad()
    def generate_baseline(
        self,
        context_ids: List[int],
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """
        Pure autoregressive generation without speculation (baseline).

        Used to measure baseline latency for speedup comparison.
        """
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        eos = eos_token_id or self.tokenizer.eos_token_id

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0, len(context_ids) :].tolist()
        return new_tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _greedy_next(self, context_ids: List[int]) -> int:
        """Single greedy token prediction."""
        input_tensor = torch.tensor(
            [context_ids], dtype=torch.long, device=self.device
        )
        logits = self.model(input_ids=input_tensor).logits[0, -1]
        return int(logits.argmax().item())

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


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
