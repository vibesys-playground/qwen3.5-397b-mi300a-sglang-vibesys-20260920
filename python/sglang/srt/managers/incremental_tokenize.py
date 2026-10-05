"""Incremental (suffix-only) tokenization of the chat-template-rendered
prompt, gated on SGLANG_INCREMENTAL_TOKENIZE (default off: ``encode_chat_prompt``
below falls back to plain full-text tokenization, with the exact same
``tokenizer.encode(rendered_prompt, **encode_kwargs)`` call the accepted
stack already makes, whenever the switch is unset -- behavior and cost are
byte-identical to the accepted stack in that case).

Background (measured during the multi-turn serving campaign): at
every turn, the server re-renders and re-tokenizes the ENTIRE rendered
conversation from scratch, even though every earlier turn's own text was
already tokenized on a previous request. ``template_rendered -> tokenized``
(the fast-tokenizer encode step alone) was measured as the single largest
TokenizerManager-side cost: 4.17/5.03/6.87ms medians at 256/512/1024 extend
tokens (turn-2 prompts of roughly 1200-2200 total tokens), about
0.003ms/total-prompt-token, and 4.95/7.96ms p50/p95 at 48 concurrent
sessions. The campaign ledger under .vibesys/tasks/multiturn/ carries the
full write-up and pre-registered prediction.

Design: a small process-wide LRU cache maps a rendered "history text" (the
Jinja chat-template render of a message list with ``add_generation_prompt=
False``, i.e. through the end of the last complete message and nothing
past it) to the token ids for that exact text. On a new request:

  1. Render the request's own history text the same way (add_generation_
     prompt=False, otherwise identical kwargs to the real generation-
     prompt render).
  2. Look for the longest cached history text that is a literal string
     prefix of this request's history text, AND whose end lands
     immediately before a chat-template special-token boundary (e.g.
     ``<|im_start|>``) in this request's own text -- so no BPE merge can
     span the split point and the two tokenizations can be concatenated
     losslessly.
  3. On a hit, tokenize only the new suffix (``add_special_tokens=False``,
     since this text is never the start of a sequence) and concatenate it
     onto the cached ids, instead of tokenizing the whole conversation.
     On a miss (first turn of a conversation, a cold/evicted cache, or the
     previous turn's rendered text turning out not to be a usable prefix
     -- e.g. the chat template stripped a `<think>` block, changed a
     system prompt, or added the generation prompt somewhere other than
     the very end), tokenize the whole text, exactly like the switch-off
     path, and separately cache the history-text tokenization so a LATER
     turn of the same conversation can still hit.
  4. The generation-prompt tail (the part of the full rendered prompt past
     the history text, e.g. ``<|im_start|>assistant\\n...``) is always
     tokenized separately with ``add_special_tokens=False`` and appended;
     it is tiny and constant-ish in length, not conversation-length-
     dependent, so it is never worth caching.

The cache is keyed on rendered TEXT, not on any session id or client-
supplied identifier: this server is stateless across HTTP requests (each
turn resends the full message history), so a cache hit only requires that
some earlier request's own rendered history happens to be a literal
prefix of this one's, regardless of which conversation produced it. A
bounded LRU (default 256 entries) keeps memory flat regardless of how many
distinct conversations are in flight; an evicted or never-cached
conversation simply falls back to full tokenization for its next turn,
never to an incorrect result -- this module makes no attempt to repair a
mismatch, it only ever chooses between "safe to reuse" and "start over".

An offline exactness check (part of the campaign's evaluation scripts)
verifies, against the real checkpoint tokenizer and every turn of the
benchmark's own sessions, that ``encode_chat_prompt``'s output is
byte-identical to plain full-text tokenization for every turn, and reports
the cache hit rate, before this switch is used in any timed comparison.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

_TRUTHY_FALSE = {"0", "false", "no", ""}


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in _TRUTHY_FALSE


ENABLED = _env_truthy("SGLANG_INCREMENTAL_TOKENIZE")

# The exact Qwen3.5 chat template (this checkpoint's tokenizer_config.json,
# read on the cluster) renders every message as
# "<|im_start|><role>\n...<|im_end|>\n" and appends the generation prompt,
# when requested, as "<|im_start|>assistant\n..." -- i.e. any text that
# follows a complete rendered history always begins with this literal
# special-token string, and BPE cannot merge tokens across it (it is a
# registered added/special token, encoded atomically by the fast
# tokenizer). This is the one boundary marker this module trusts; a cache
# candidate whose split point in the new text does not start with it is
# never used, regardless of how much of a string-prefix match it is.
BOUNDARY_MARKER = "<|im_start|>"

_DEFAULT_CAPACITY = 256


class IncrementalTokenizeCache:
    """Bounded LRU: rendered history text -> its token ids.

    Not keyed by session; any two requests whose own rendered history
    happens to share a literal prefix ending at ``BOUNDARY_MARKER`` can
    reuse each other's cached ids. Thread-safe (a lock around the whole
    lookup/insert), since some deployments run the chat-completion
    encode path off the main asyncio thread.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._data: "OrderedDict[str, List[int]]" = OrderedDict()
        self._lock = threading.Lock()
        # Diagnostics only (read by the offline exactness check and eval scripts);
        # never affect behavior.
        self.hits = 0
        self.misses = 0

    def longest_prefix_entry(self, text: str) -> Tuple[Optional[str], Optional[List[int]]]:
        """Longest cached key that is a literal prefix of ``text`` and whose
        split point in ``text`` starts with BOUNDARY_MARKER, or (None, None).
        """
        best_key: Optional[str] = None
        best_ids: Optional[List[int]] = None
        best_len = -1
        with self._lock:
            for key, ids in self._data.items():
                klen = len(key)
                if klen <= best_len or klen == 0 or klen > len(text):
                    continue
                if not text.startswith(key):
                    continue
                if not text[klen:].startswith(BOUNDARY_MARKER):
                    continue
                best_key, best_ids, best_len = key, ids, klen
            if best_key is not None:
                self._data.move_to_end(best_key)
                self.hits += 1
            else:
                self.misses += 1
        return best_key, best_ids

    def put(self, key: str, ids: List[int]) -> None:
        if not key:
            return
        with self._lock:
            self._data[key] = ids
            self._data.move_to_end(key)
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0


# Process-wide cache used by the real server. The offline exactness check and any
# other standalone harness should construct their own IncrementalTokenizeCache
# instead of sharing this one, so repeated runs in the same process don't
# see stale state from an earlier run.
_default_cache = IncrementalTokenizeCache()


def encode_chat_prompt(
    tokenizer: Any,
    *,
    is_eligible: bool,
    render_full: Callable[[], str],
    render_history: Callable[[], str],
    encode_kwargs: Dict[str, Any],
    cache: Optional[IncrementalTokenizeCache] = None,
    on_template_rendered: Optional[Callable[[], None]] = None,
) -> Tuple[str, List[int]]:
    """Returns ``(full_rendered_prompt, prompt_ids)``.

    ``render_full`` and ``render_history`` must call
    ``tokenizer.apply_chat_template`` with identical arguments except for
    ``add_generation_prompt`` (True / False respectively) -- same messages,
    tools, and extra template kwargs -- so that ``render_full()`` is
    guaranteed, by the chat template's own structure, to start with
    ``render_history()``'s output.

    ``is_eligible`` gates this whole module: pass False for multimodal
    input, non-string message content, or anything else the caller has
    already decided must go through plain full tokenization. When False,
    or when SGLANG_INCREMENTAL_TOKENIZE is unset, this reduces to exactly
    the pre-existing call: ``render_full()`` then
    ``tokenizer.encode(full_text, **encode_kwargs)``.
    """
    if cache is None:
        cache = _default_cache

    if not ENABLED or not is_eligible:
        full_text = render_full()
        if on_template_rendered is not None:
            on_template_rendered()
        ids = tokenizer.encode(full_text, **encode_kwargs)
        return full_text, ids

    full_text = render_full()
    try:
        history_text = render_history()
    except Exception:
        # Defensive: a second, generation-prompt-free render should never
        # fail when the first one succeeded, but if it ever does, fall
        # back to the exact switch-off behavior for this request rather
        # than risk any incorrectness.
        if on_template_rendered is not None:
            on_template_rendered()
        ids = tokenizer.encode(full_text, **encode_kwargs)
        return full_text, ids

    if on_template_rendered is not None:
        on_template_rendered()

    if not full_text.startswith(history_text):
        # Should not happen given this template's structure (the
        # generation prompt is only ever appended at the very end), but
        # if the two renders ever diverge anywhere but a shared trailing
        # tail, there is no safe split point: fall back completely.
        ids = tokenizer.encode(full_text, **encode_kwargs)
        try:
            history_ids = tokenizer.encode(history_text, **encode_kwargs)
            cache.put(history_text, history_ids)
        except Exception:
            pass
        return full_text, ids

    tail_text = full_text[len(history_text) :]

    cached_key, cached_ids = cache.longest_prefix_entry(history_text)
    if cached_key is None:
        # Miss: tokenize the whole thing (same cost as the switch-off
        # path for this one request), and separately cache the
        # history-only tokenization -- using the SAME encode_kwargs as
        # the full path, since this is, like the full text, the start of
        # a fresh sequence and must pick up any leading special token the
        # tokenizer would otherwise add.
        ids = tokenizer.encode(full_text, **encode_kwargs)
        history_ids = tokenizer.encode(history_text, **encode_kwargs)
        cache.put(history_text, history_ids)
        return full_text, ids

    suffix_text = history_text[len(cached_key) :]
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False) if suffix_text else []
    history_ids = cached_ids + suffix_ids
    cache.put(history_text, history_ids)

    tail_ids = tokenizer.encode(tail_text, add_special_tokens=False) if tail_text else []
    ids = history_ids + tail_ids
    return full_text, ids
