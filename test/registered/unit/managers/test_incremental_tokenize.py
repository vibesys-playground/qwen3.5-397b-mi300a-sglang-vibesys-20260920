"""Unit tests for incremental_tokenize.py: the suffix-only incremental
tokenization cache behind SGLANG_INCREMENTAL_TOKENIZE.

Uses a synthetic tokenizer (no torch/transformers dependency) so these tests
run on CPU with no model download. The synthetic tokenizer is intentionally
concatenative at the character level (encode(a) + encode(b, add_special_tokens=
False) == encode(a + b, add_special_tokens=False) for any a, b), which mirrors
the property Qwen3.5's own fast tokenizer has around a chat-template special-
token boundary such as <|im_start|> (an atomic added token that a BPE merge
cannot cross) -- the exact property incremental_tokenize.py relies on to
concatenate a cached prefix's ids with a freshly tokenized suffix.
"""

import unittest
from unittest.mock import call

from sglang.srt.managers import incremental_tokenize
from sglang.srt.managers.incremental_tokenize import (
    BOUNDARY_MARKER,
    IncrementalTokenizeCache,
    encode_chat_prompt,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeTokenizer:
    """Character-level synthetic tokenizer: each character maps to its
    ordinal. add_special_tokens=True (the default, mirroring a tokenizer
    that auto-adds specials) prepends a fixed BOS id; add_special_tokens=
    False does not. This makes the tokenizer trivially concatenative,
    which is what lets suffix-only tokenization be spliced onto a cached
    prefix and still match full tokenization exactly.
    """

    BOS_ID = -1

    def __init__(self):
        self.encode_calls = []

    def encode(self, text, add_special_tokens=True, **kwargs):
        self.encode_calls.append(call(text, add_special_tokens=add_special_tokens))
        ids = [ord(c) for c in text]
        if add_special_tokens:
            ids = [self.BOS_ID] + ids
        return ids


class IncrementalTokenizeCacheTestCase(unittest.TestCase):
    """Directly exercises the LRU cache and its boundary rule, independent
    of encode_chat_prompt."""

    def test_hit_requires_boundary_marker_immediately_after_the_match(self):
        cache = IncrementalTokenizeCache(capacity=8)
        history = f"<|im_start|>user\nhi<|im_end|>\n"
        cache.put(history, [1, 2, 3])

        # A new text that literally starts with `history` AND continues with
        # BOUNDARY_MARKER is a valid hit.
        good_text = history + BOUNDARY_MARKER + "assistant\nhello<|im_end|>\n"
        key, ids = cache.longest_prefix_entry(good_text)
        self.assertEqual(key, history)
        self.assertEqual(ids, [1, 2, 3])

    def test_string_prefix_without_boundary_marker_is_not_reused(self):
        """A coincidental string-prefix match whose continuation does NOT
        start with BOUNDARY_MARKER must be rejected, even though it is a
        literal prefix -- reusing it could silently splice ids across a
        point where a BPE merge might have applied differently had the
        text been tokenized as one string."""
        cache = IncrementalTokenizeCache(capacity=8)
        history = "<|im_start|>user\nhi<|im_end|>\n"
        cache.put(history, [1, 2, 3])

        # Continuation right after the cached prefix is plain text, not the
        # special-token boundary marker.
        bad_text = history + "not a boundary at all"
        key, ids = cache.longest_prefix_entry(bad_text)
        self.assertIsNone(key)
        self.assertIsNone(ids)

    def test_miss_when_no_cached_entry_is_a_prefix(self):
        cache = IncrementalTokenizeCache(capacity=8)
        cache.put("<|im_start|>user\nhi<|im_end|>\n", [1, 2, 3])
        key, ids = cache.longest_prefix_entry("completely different text")
        self.assertIsNone(key)
        self.assertIsNone(ids)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 0)

    def test_longest_matching_prefix_wins(self):
        cache = IncrementalTokenizeCache(capacity=8)
        short = "<|im_start|>user\nhi<|im_end|>\n"
        long_ = short + "<|im_start|>assistant\nyo<|im_end|>\n"
        cache.put(short, [1, 2, 3])
        cache.put(long_, [1, 2, 3, 4, 5, 6])

        text = long_ + BOUNDARY_MARKER + "user\nmore<|im_end|>\n"
        key, ids = cache.longest_prefix_entry(text)
        self.assertEqual(key, long_)
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6])

    def test_lru_eviction_drops_least_recently_used(self):
        cache = IncrementalTokenizeCache(capacity=2)
        cache.put("a<|im_start|>", [1])
        cache.put("b<|im_start|>", [2])
        cache.put("c<|im_start|>", [3])  # evicts "a<|im_start|>"

        key, ids = cache.longest_prefix_entry(
            "a<|im_start|>" + BOUNDARY_MARKER + "x"
        )
        self.assertIsNone(key)
        self.assertIsNone(ids)

        key, ids = cache.longest_prefix_entry(
            "b<|im_start|>" + BOUNDARY_MARKER + "x"
        )
        self.assertEqual(key, "b<|im_start|>")


class EncodeChatPromptTestCase(unittest.TestCase):
    """Exercises encode_chat_prompt's own decision logic (disabled/ineligible
    fallback, cache miss, cache hit, and the equals-full-tokenization
    exactness property), using the synthetic tokenizer above."""

    def setUp(self):
        self.tokenizer = _FakeTokenizer()
        self.cache = IncrementalTokenizeCache(capacity=8)
        self._orig_enabled = incremental_tokenize.ENABLED
        self.addCleanup(self._restore_enabled)

    def _restore_enabled(self):
        incremental_tokenize.ENABLED = self._orig_enabled

    def _full_tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def test_disabled_falls_back_to_plain_full_tokenize_and_skips_history_render(self):
        incremental_tokenize.ENABLED = False
        full_text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        history_called = []

        rendered, ids = encode_chat_prompt(
            self.tokenizer,
            is_eligible=True,
            render_full=lambda: full_text,
            render_history=lambda: history_called.append(1) or "should not run",
            encode_kwargs={},
            cache=self.cache,
        )
        self.assertEqual(rendered, full_text)
        self.assertEqual(ids, self._full_tokenize(full_text))
        self.assertEqual(history_called, [])  # render_history never invoked

    def test_ineligible_falls_back_to_plain_full_tokenize(self):
        incremental_tokenize.ENABLED = True
        full_text = "<|im_start|>user\n<image>\n<|im_end|>\n<|im_start|>assistant\n"

        rendered, ids = encode_chat_prompt(
            self.tokenizer,
            is_eligible=False,  # e.g. request carries image content
            render_full=lambda: full_text,
            render_history=lambda: "should not run",
            encode_kwargs={},
            cache=self.cache,
        )
        self.assertEqual(rendered, full_text)
        self.assertEqual(ids, self._full_tokenize(full_text))

    def test_cache_miss_first_turn_falls_back_and_seeds_the_cache(self):
        incremental_tokenize.ENABLED = True
        history_text = "<|im_start|>user\nhi<|im_end|>\n"
        full_text = history_text + "<|im_start|>assistant\n"

        rendered, ids = encode_chat_prompt(
            self.tokenizer,
            is_eligible=True,
            render_full=lambda: full_text,
            render_history=lambda: history_text,
            encode_kwargs={},
            cache=self.cache,
        )
        self.assertEqual(rendered, full_text)
        self.assertEqual(ids, self._full_tokenize(full_text))
        # The cache should now hold the history-only tokenization, seeded
        # for a later turn.
        key, cached_ids = self.cache.longest_prefix_entry(
            history_text + BOUNDARY_MARKER + "x"
        )
        self.assertEqual(key, history_text)
        self.assertEqual(cached_ids, self._full_tokenize(history_text))

    def test_cache_hit_tokenizes_only_the_suffix_and_matches_full_tokenization(self):
        incremental_tokenize.ENABLED = True
        turn1_history = "<|im_start|>user\nhi<|im_end|>\n"
        turn1_full = turn1_history + "<|im_start|>assistant\n"

        # Turn 1: cold cache, seeds the entry.
        encode_chat_prompt(
            self.tokenizer,
            is_eligible=True,
            render_full=lambda: turn1_full,
            render_history=lambda: turn1_history,
            encode_kwargs={},
            cache=self.cache,
        )

        # Turn 2: history now includes turn 1's full exchange plus a new
        # user message; the previous history text is an exact prefix
        # ending right at a boundary marker.
        turn2_history = (
            turn1_history + "<|im_start|>assistant\nyo<|im_end|>\n"
            "<|im_start|>user\nmore<|im_end|>\n"
        )
        turn2_full = turn2_history + "<|im_start|>assistant\n"

        self.tokenizer.encode_calls.clear()
        rendered, ids = encode_chat_prompt(
            self.tokenizer,
            is_eligible=True,
            render_full=lambda: turn2_full,
            render_history=lambda: turn2_history,
            encode_kwargs={},
            cache=self.cache,
        )

        # Capture what encode_chat_prompt itself asked the tokenizer to
        # encode before any further (test-only) calls touch the same
        # tokenizer's call log below.
        encoded_texts = [c.args[0] for c in self.tokenizer.encode_calls]

        # Exactness: incremental result must equal plain full tokenization.
        self.assertEqual(rendered, turn2_full)
        self.assertEqual(ids, self._full_tokenize(turn2_full))

        # Economy: on a hit, the tokenizer must never be asked to encode
        # the full text or the full history again -- only the new suffix
        # of the history and the small generation-prompt tail.
        self.assertNotIn(turn2_full, encoded_texts)
        self.assertNotIn(turn2_history, encoded_texts)
        suffix = turn2_history[len(turn1_history) :]
        self.assertIn(suffix, encoded_texts)

    def test_full_text_not_starting_with_history_falls_back_completely(self):
        """Defensive: if render_full/render_history ever diverge outside a
        shared trailing tail (should not happen given a real chat
        template's structure), there is no safe split point and this must
        fall back to full tokenization rather than risk an incorrect
        splice."""
        incremental_tokenize.ENABLED = True
        full_text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        unrelated_history = "totally different text"

        rendered, ids = encode_chat_prompt(
            self.tokenizer,
            is_eligible=True,
            render_full=lambda: full_text,
            render_history=lambda: unrelated_history,
            encode_kwargs={},
            cache=self.cache,
        )
        self.assertEqual(rendered, full_text)
        self.assertEqual(ids, self._full_tokenize(full_text))


if __name__ == "__main__":
    unittest.main()
