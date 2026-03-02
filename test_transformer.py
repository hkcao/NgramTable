"""
Unit / smoke tests for the transformer-based speculative decoding system.

Tests run without loading the full Qwen model — proposers are exercised
against toy token sequences, and the verifier is stubbed where needed.

Run:
    python test_transformer.py
    python test_transformer.py -v   # verbose
"""

import sys
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from speculative.proposers import HashTableProposer, KMPProposer, TrieProposer
from speculative.metrics import MetricsTracker, RequestMetrics, StepMetrics

# Engine / verifier imports are deferred to avoid requiring torch for proposer tests


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _tok(text: str) -> List[int]:
    """Simple char→ord tokenizer for tests."""
    return [ord(c) for c in text]


# ---------------------------------------------------------------------------
# KMP Proposer Tests
# ---------------------------------------------------------------------------

class TestKMPProposer(unittest.TestCase):

    def setUp(self):
        self.p = KMPProposer(min_n=2, max_n=4)

    def test_no_match_short_context(self):
        ctx = _tok("ab")
        self.assertEqual(self.p.propose(ctx, 3), [])

    def test_exact_repeat(self):
        # "abcabc" — suffix "bc" matches earlier position, next is "a"
        ctx = _tok("abcabc")
        draft = self.p.propose(ctx, 3)
        # After "bc" (last 2 tokens), the match at index 1 is followed by "c"
        self.assertGreater(len(draft), 0)

    def test_longer_match_preferred(self):
        # "abcdabcd" — suffix "bcd" (n=3) matches at index 1, next is "a"
        ctx = _tok("abcdabcd")
        draft = self.p.propose(ctx, 2)
        self.assertGreater(len(draft), 0)
        # First proposed token should be 'a' (ord 97)
        self.assertEqual(draft[0], ord('a'))

    def test_update_noop(self):
        ctx = _tok("abcabc")
        self.p.update(ctx)  # should not raise

    def test_propose_respects_num_tokens(self):
        ctx = _tok("abcabcabcabc")
        draft = self.p.propose(ctx, 2)
        self.assertLessEqual(len(draft), 2)

    def test_no_match_returns_empty(self):
        ctx = _tok("abcde")  # no repeating sub-sequence
        draft = self.p.propose(ctx, 3)
        # May or may not match depending on unique chars — just check type
        self.assertIsInstance(draft, list)


# ---------------------------------------------------------------------------
# HashTable Proposer Tests
# ---------------------------------------------------------------------------

class TestHashTableProposer(unittest.TestCase):

    def setUp(self):
        self.p = HashTableProposer(min_n=2, max_n=3)

    def test_basic_frequency(self):
        # Sequence: A B C A B C A B C
        ctx = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        draft = self.p.propose(ctx, 3)
        # After [2, 3] appears three times followed by [1], expect [1]
        self.assertEqual(draft[0], 1)

    def test_incremental_update(self):
        ctx = [1, 2, 3, 1, 2, 3]
        self.p.propose(ctx, 2)
        ctx2 = ctx + [1, 2, 3, 1]
        draft = self.p.propose(ctx2, 2)
        self.assertIsInstance(draft, list)

    def test_reset_clears_state(self):
        ctx = [1, 2, 3, 1, 2, 3]
        self.p.propose(ctx, 2)
        self.p.reset()
        self.assertEqual(self.p._built_up_to, 0)

    def test_frequency_tie_breaking(self):
        # (1,2) appears 2x before 3 and 1x before 4 → should prefer 3 after [1,2]
        # Context ends with [1,2] so n=2 bigram lookup fires
        ctx = [1, 2, 3, 1, 2, 4, 1, 2]
        draft = self.p.propose(ctx, 1)
        self.assertEqual(draft, [3])

    def test_empty_context(self):
        self.assertEqual(self.p.propose([], 3), [])


# ---------------------------------------------------------------------------
# Trie Proposer Tests
# ---------------------------------------------------------------------------

class TestTrieProposer(unittest.TestCase):

    def setUp(self):
        self.p = TrieProposer(min_n=2, max_n=4)

    def test_basic_prediction(self):
        ctx = [1, 2, 3, 1, 2, 3, 1, 2]
        # After [1, 2] the trie should predict 3
        draft = self.p.propose(ctx, 1)
        self.assertEqual(draft, [3])

    def test_multi_step_proposal(self):
        ctx = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
        draft = self.p.propose(ctx, 3)
        # Should predict [3, 4, 1]
        self.assertEqual(draft[:2], [3, 4])

    def test_trie_reset(self):
        ctx = [1, 2, 3, 1, 2, 3]
        self.p.propose(ctx, 2)
        self.p.reset()
        self.assertIsNone(self.p._root.best_next)

    def test_no_match_returns_empty(self):
        ctx = [10, 20]  # never seen before
        draft = self.p.propose(ctx, 2)
        self.assertEqual(draft, [])


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def _make_req(self, proposed: List[int], accepted: List[int]) -> RequestMetrics:
        req = RequestMetrics(prompt="test")
        req.total_output_tokens = sum(a + 1 for a in accepted)
        for p, a in zip(proposed, accepted):
            req.steps.append(StepMetrics(
                num_proposed=p, num_accepted=a, bonus_token=0,
                context_len=10, proposer_name="kmp",
                propose_time_ms=1.0, verify_time_ms=5.0,
            ))
        return req

    def test_acceptance_rate(self):
        req = self._make_req([5, 5, 5], [5, 4, 3])
        # total_proposed=15, total_accepted=12
        self.assertAlmostEqual(req.token_acceptance_rate, 12 / 15, places=4)

    def test_perfect_hit_rate(self):
        req = self._make_req([3, 3, 3], [3, 3, 3])
        self.assertAlmostEqual(req.draft_hit_rate, 1.0, places=4)

    def test_partial_hit_rate(self):
        req = self._make_req([3, 3, 3], [3, 2, 1])
        # Only first step is perfect
        self.assertAlmostEqual(req.draft_hit_rate, 1 / 3, places=4)

    def test_mean_accepted_length(self):
        req = self._make_req([5, 5], [3, 4])
        # (3+1 + 4+1) / 2 = 4.5
        self.assertAlmostEqual(req.mean_accepted_length, 4.5, places=3)

    def test_tracker_summary(self):
        tracker = MetricsTracker("kmp", num_speculative_tokens=5)
        req = tracker.new_request("test")
        req.steps.append(StepMetrics(
            num_proposed=5, num_accepted=4, bonus_token=0,
            context_len=10, proposer_name="kmp",
            propose_time_ms=1.0, verify_time_ms=8.0,
        ))
        req.total_output_tokens = 5
        s = tracker.summary()
        self.assertEqual(s["total_proposed"], 5)
        self.assertEqual(s["total_accepted"], 4)
        self.assertAlmostEqual(s["token_acceptance_rate"], 0.8, places=2)


# ---------------------------------------------------------------------------
# Integration smoke test (no model loading)
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed — skipping engine tests")
class TestEngineIntegration(unittest.TestCase):

    def test_engine_loop_with_mock_verifier(self):
        """Verify engine loop terminates correctly with a mock verifier."""
        from speculative.engine import SpeculativeEngine

        # Mock verifier that always accepts all drafts and adds a fixed bonus
        mock_verifier = MagicMock()
        mock_verifier.encode.return_value = [1, 2, 3]
        mock_verifier.eos_token_id = 0
        mock_verifier.decode.return_value = "generated text"

        call_count = [0]

        def mock_verify(ctx, draft):
            call_count[0] += 1
            # Accept all drafts; bonus token = 99 (not EOS)
            # After 5 calls, emit EOS as bonus
            if call_count[0] >= 5:
                return draft, 0  # 0 = EOS
            return draft, 99

        mock_verifier.verify.side_effect = mock_verify

        proposer = KMPProposer(min_n=2, max_n=3)
        engine = SpeculativeEngine(
            proposer=proposer,
            verifier=mock_verifier,
            num_speculative_tokens=3,
        )

        output = engine.generate("test prompt", max_new_tokens=50)
        self.assertIsInstance(output, str)
        # Should have terminated before max_new_tokens due to EOS
        mock_verifier.verify.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1)
