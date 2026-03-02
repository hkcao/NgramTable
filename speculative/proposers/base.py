"""Abstract base class for all n-gram draft proposers."""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseProposer(ABC):
    """
    Base interface for n-gram speculative decoding proposers.

    All proposers share the same API:
        propose(context_ids, num_tokens) -> List[int]

    The proposer scans the current context token sequence to predict the next
    `num_tokens` tokens without a neural network forward pass.
    """

    def __init__(self, min_n: int = 2, max_n: int = 5):
        """
        Args:
            min_n: Minimum n-gram length to consider for matching.
            max_n: Maximum n-gram length to consider for matching.
        """
        if min_n < 1:
            raise ValueError("min_n must be >= 1")
        if max_n < min_n:
            raise ValueError("max_n must be >= min_n")
        self.min_n = min_n
        self.max_n = max_n

    @abstractmethod
    def propose(
        self,
        context_ids: List[int],
        num_tokens: int,
    ) -> List[int]:
        """
        Propose up to `num_tokens` draft tokens given the context.

        Args:
            context_ids: Full token sequence seen so far (prompt + generated).
            num_tokens:  Number of tokens to propose.

        Returns:
            List of proposed token ids (may be shorter than num_tokens if no
            n-gram match is found).
        """

    @abstractmethod
    def update(self, context_ids: List[int]) -> None:
        """
        Update internal state after new tokens have been appended to the
        context.  Called once per accepted token batch.

        For stateless proposers (KMP) this is a no-op.
        For stateful proposers (Hash, Trie) this triggers incremental updates.
        """

    def reset(self) -> None:
        """Reset per-request state.  Override if needed."""
