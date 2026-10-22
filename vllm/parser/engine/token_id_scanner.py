# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scan delta token IDs for special tokens and split the stream into
pre-lexed terminals and plain text chunks."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import tokenizers
from transformers import TokenizersBackend

from vllm.tokenizers.streaming import (
    _PartialPromptDecodeStream,
    create_decode_stream,
)

DROP_TERMINAL = "__DROP__"


@dataclass(slots=True)
class TextChunk:
    text: str
    token_texts: tuple[str, ...] = ()
    token_count: int = 0


@dataclass(slots=True)
class PreLexedTerminal:
    terminal: str
    token_id: int
    text: str


LexerInput = TextChunk | PreLexedTerminal


class _TokenTextPositions:
    """Locate control IDs with the same native decoder as the output processor.

    Token and text deltas need not end at the same position: UTF-8 decoding and
    stop-string buffering can delay text. Keep absolute decoded positions, not
    textual searches, so an ordinary spelling of a marker is never a boundary.
    Only a partial visible control marker is buffered; ordinary text is emitted
    from the supplied delta, never reconstructed or normalized here.
    """

    def __init__(self, backend, terminal_names: dict[int, str], decode_token):
        self.backend = backend
        self.terminal_names = terminal_names
        self.decode_token = decode_token
        self.reset()

    def reset(self) -> None:
        self.stream: (
            tokenizers.decoders.DecodeStream | _PartialPromptDecodeStream | None
        ) = None
        self.prompt_token_ids: Sequence[int] | None = None
        self.decoded_chars = 0
        self.emitted_chars = 0
        self.pending_text = ""
        self.terminals: deque[tuple[int, int, PreLexedTerminal]] = deque()

    def set_prompt(self, token_ids: Sequence[int]) -> None:
        if self.stream is not None:
            raise RuntimeError("Cannot change parser prompt after decoding starts")
        self.prompt_token_ids = token_ids

    def scan(self, text: str, token_ids: Sequence[int]) -> list[LexerInput]:
        if self.stream is None:
            if not token_ids:
                # Text-only parser APIs still use their lexical grammar.
                return [TextChunk(text)] if text else []
            self.stream = create_decode_stream(
                self.backend, self.prompt_token_ids, skip_special_tokens=False
            )
            self.prompt_token_ids = None
        for token_id in token_ids:
            try:
                piece = self.stream.step(self.backend, token_id) or ""
            except Exception as exc:
                # Match FastIncrementalDetokenizer's native UTF-8 prefix
                # recovery; do not guess positions from visible marker text.
                if not str(exc).startswith("Invalid prefix encountered"):
                    raise
                self.stream = tokenizers.decoders.DecodeStream(
                    skip_special_tokens=False
                )
                piece = self.stream.step(self.backend, token_id) or ""
            self.decoded_chars += len(piece)
            name = self.terminal_names.get(token_id)
            if name is not None:
                marker = self.decode_token(token_id)
                if not marker or not piece.endswith(marker):
                    raise ValueError("Control token has no exact decoded boundary")
                end = self.decoded_chars
                self.terminals.append(
                    (end - len(marker), end, PreLexedTerminal(name, token_id, marker))
                )

        visible = self.pending_text + text
        cursor = 0
        base = self.emitted_chars
        limit = base + len(visible)
        items: list[LexerInput] = []
        while self.terminals:
            start, end, terminal = self.terminals[0]
            if start >= limit:
                break
            if start > base + cursor:
                items.append(TextChunk(visible[cursor : start - base]))
                cursor = start - base
            if end > limit:
                break
            if visible[start - base : end - base] != terminal.text:
                raise ValueError("Token and text streams disagree at a control ID")
            items.append(terminal)
            cursor = end - base
            self.terminals.popleft()
        # A marker partially held by the output processor must not leak into
        # the lexer. Everything before the next known marker is ordinary text.
        safe_end = len(visible)
        if self.terminals:
            safe_end = min(safe_end, max(cursor, self.terminals[0][0] - base))
        if safe_end > cursor:
            items.append(TextChunk(visible[cursor:safe_end]))
        self.emitted_chars += safe_end
        self.pending_text = visible[safe_end:]
        return items

    def finish(self) -> list[LexerInput]:
        # A stop may remove a suffix or the final EOS token. Do not resurrect
        # undisplayed text or fire transitions beyond the delivered prefix.
        self.pending_text = ""
        self.terminals.clear()
        return []


class TokenIDScanner:
    """Maps special token IDs in the delta to terminals.

    Before text-based lexing happens, the scanner checks each token ID
    in the delta against a mapping of ``{token_id: terminal_name}``.
    Matched tokens are emitted as :class:`PreLexedTerminal` items;
    everything else is grouped into :class:`TextChunk` items for the
    incremental lexer to process.

    When a terminal's text is not yet in ``delta_text`` (held back by
    the detokenizer), the terminal is deferred until the text arrives
    in a subsequent delta.
    """

    def __init__(
        self,
        token_id_to_terminal: dict[int, str],
        tokenizer,
        *,
        track_token_positions: bool = False,
    ) -> None:
        self.token_id_to_terminal = token_id_to_terminal
        self.tokenizer = tokenizer
        self._token_text_cache: dict[int, str] = {}
        self._deferred_terminals: list[PreLexedTerminal] = []
        self._deferred_prefix_token_counts: list[int] = []
        self._deferred_prefix_texts: list[str] = []
        self._deferred_trailing_token_count = 0
        self._deferred_post_text: str = ""
        backend = getattr(tokenizer, "_tokenizer", None)
        # Only model classes that opt in use native position tracking.
        # Text-only/mocked tokenizers retain the existing lexical interface.
        self._positions = (
            _TokenTextPositions(backend, token_id_to_terminal, self._decode_token)
            if track_token_positions and isinstance(tokenizer, TokenizersBackend)
            else None
        )

    @property
    def tracks_token_positions(self) -> bool:
        return self._positions is not None

    def set_prompt_token_ids(self, token_ids: Sequence[int]) -> None:
        if self._positions is not None:
            self._positions.set_prompt(token_ids)

    def reset(self) -> None:
        """Clear mutable state for reuse. Preserves the token text cache."""
        self._deferred_terminals.clear()
        self._deferred_prefix_token_counts.clear()
        self._deferred_prefix_texts.clear()
        self._deferred_trailing_token_count = 0
        self._deferred_post_text = ""
        if self._positions is not None:
            self._positions.reset()

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self._token_text_cache:
            token_text = self.tokenizer.decode([token_id])
            self._token_text_cache[token_id] = (
                token_text if isinstance(token_text, str) else ""
            )
        return self._token_text_cache[token_id]

    def _decode_tokens(self, token_ids: Sequence[int]) -> tuple[str, ...]:
        if self.tokenizer is None:
            return ()
        return tuple(self._decode_token(tid) for tid in token_ids)

    _EMPTY: tuple[LexerInput, ...] = ()

    def scan(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> Sequence[LexerInput]:
        if self._positions is not None:
            return self._positions.scan(delta_text, delta_token_ids)
        prefix_items: list[LexerInput] = []
        effective_text = delta_text
        deferred_trailing_count = 0

        if self._deferred_terminals:
            prefix_items, effective_text, deferred_trailing_count = (
                self._resolve_deferred(delta_text, len(delta_token_ids))
            )

        if not self.token_id_to_terminal:
            if effective_text:
                token_texts = self._decode_tokens(delta_token_ids)
                prefix_items.append(
                    TextChunk(
                        effective_text,
                        token_texts,
                        deferred_trailing_count + len(token_texts),
                    )
                )
            return prefix_items

        has_special = False
        token_id_to_terminal = self.token_id_to_terminal
        for tid in delta_token_ids:
            if tid in token_id_to_terminal:
                has_special = True
                break

        if not has_special:
            if effective_text:
                if not prefix_items:
                    token_texts = self._decode_tokens(delta_token_ids)
                    return [TextChunk(effective_text, token_texts, len(token_texts))]
                token_texts = self._decode_tokens(delta_token_ids)
                prefix_items.append(
                    TextChunk(
                        effective_text,
                        token_texts,
                        deferred_trailing_count + len(token_texts),
                    )
                )
            return prefix_items or self._EMPTY

        decoded_token_texts = [self._decode_token(tid) for tid in delta_token_ids]

        results: list[LexerInput] = []
        group_start = 0

        def append_text_group(end: int) -> None:
            if group_start == end:
                return
            group_ids = delta_token_ids[group_start:end]
            # Individual UTF-8 byte tokens can decode as replacement characters.
            # Decode the whole ordinary span so those bytes form the real text.
            group_text = self.tokenizer.decode(list(group_ids))
            results.append(
                TextChunk(
                    group_text,
                    tuple(decoded_token_texts[group_start:end]),
                    end - group_start,
                )
            )

        for idx, tid in enumerate(delta_token_ids):
            terminal = self.token_id_to_terminal.get(tid)
            if terminal is not None:
                append_text_group(idx)
                results.append(
                    PreLexedTerminal(terminal, tid, decoded_token_texts[idx])
                )
                group_start = idx + 1

        append_text_group(len(delta_token_ids))

        if deferred_trailing_count:
            results.insert(0, TextChunk("", token_count=deferred_trailing_count))

        if effective_text:
            results = self._recover_holdback_text(effective_text, results)
        else:
            # No detokenizer text to validate against — individually-decoded
            # TextChunks are unreliable (context-dependent decoding).
            # Defer PreLexedTerminals so the state machine doesn't
            # transition before the preceding text has arrived.  The
            # deferred terminals will be resolved against the actual
            # delta_text in a subsequent scan() or flushed by finish().
            prefix_token_count = 0
            prefix_text = ""
            for r in results:
                if isinstance(r, TextChunk):
                    prefix_token_count += r.token_count
                    prefix_text += r.text
                if isinstance(r, PreLexedTerminal):
                    self._deferred_terminals.append(r)
                    self._deferred_prefix_token_counts.append(prefix_token_count)
                    self._deferred_prefix_texts.append(prefix_text)
                    prefix_token_count = 0
                    prefix_text = ""
            self._deferred_trailing_token_count += prefix_token_count
            results = []

        return prefix_items + results

    def flush_pending(self) -> list[LexerInput]:
        if self._positions is not None:
            return self._positions.finish()
        if not self._deferred_terminals and not self._deferred_post_text:
            return []
        results: list[LexerInput] = []
        if self._deferred_post_text:
            prefix_count = (
                self._deferred_prefix_token_counts[0]
                if self._deferred_prefix_token_counts
                else 0
            )
            results.append(
                TextChunk(self._deferred_post_text, token_count=prefix_count)
            )
            self._deferred_post_text = ""
        results.extend(self._deferred_terminals)
        if self._deferred_trailing_token_count:
            results.append(
                TextChunk("", token_count=self._deferred_trailing_token_count)
            )
        self._deferred_terminals.clear()
        self._deferred_prefix_token_counts.clear()
        self._deferred_prefix_texts.clear()
        self._deferred_trailing_token_count = 0
        return results

    def _resolve_deferred(
        self,
        delta_text: str,
        current_token_count: int = 0,
    ) -> tuple[list[LexerInput], str, int]:
        """Resolve deferred terminals against new delta_text.

        When a previous ``scan()`` deferred a terminal (its text hadn't
        arrived yet), the next delta's text should contain that terminal's
        text.  Split delta_text at the terminal boundary: text before
        belongs to the previous parser state, the terminal triggers the
        state transition, and text after belongs to the new state.

        Returns ``(prefix_items, remaining_text)`` where prefix_items
        are the resolved deferred terminals (with any preceding text)
        and remaining_text is the unconsumed portion of delta_text that
        should be scanned with the current delta's token IDs.
        """
        deferred = self._deferred_terminals
        prefix_token_counts = self._deferred_prefix_token_counts
        prefix_texts = self._deferred_prefix_texts
        trailing_token_count = self._deferred_trailing_token_count
        self._deferred_terminals = []
        self._deferred_prefix_token_counts = []
        self._deferred_prefix_texts = []
        self._deferred_trailing_token_count = 0

        results: list[LexerInput] = []
        remaining = delta_text

        if self._deferred_post_text:
            remaining = self._deferred_post_text + remaining
            self._deferred_post_text = ""

        for idx, terminal in enumerate(deferred):
            prefix_token_count = prefix_token_counts[idx]
            prefix_text = prefix_texts[idx]
            # Keep the ordinary-token span with its deferred control ID.
            # A spelling in earlier literal text is not that control token.
            pos = remaining.find(prefix_text + terminal.text)
            if pos >= 0:
                pos += len(prefix_text)
            elif (prefix_text + terminal.text).startswith(remaining):
                pos = -1  # The exact token/text prefix is still held back.
            else:
                pos = remaining.find(terminal.text)
            if pos > 0:
                results.append(
                    TextChunk(remaining[:pos], token_count=prefix_token_count)
                )
                results.append(terminal)
                remaining = remaining[pos + len(terminal.text) :]
            elif pos == 0:
                results.append(terminal)
                remaining = remaining[len(terminal.text) :]
            else:
                # Accumulate text until terminal text arrives —
                # only the terminal provides a reliable split point.
                if remaining:
                    self._deferred_post_text += remaining
                    prefix_token_count += current_token_count
                    remaining = ""
                self._deferred_terminals.append(terminal)
                self._deferred_prefix_token_counts.append(prefix_token_count)
                self._deferred_prefix_texts.append(prefix_text)

        if self._deferred_terminals:
            self._deferred_trailing_token_count = trailing_token_count
            trailing_token_count = 0

        return results, remaining, trailing_token_count

    def _recover_holdback_text(
        self,
        delta_text: str,
        results: list[LexerInput],
    ) -> list[LexerInput]:
        """Recover detokenizer hold-back text not in delta_token_ids.

        The detokenizer may flush previously held-back text in
        ``delta_text`` that has no corresponding token ID in
        ``delta_token_ids``.  This hold-back text always appears as a
        prefix of ``delta_text``.
        """
        if not results:
            return [TextChunk(delta_text)]

        reconstructed = self._join_decoded_text(results)

        if not reconstructed:
            return [TextChunk(delta_text)] + results

        pos = delta_text.find(reconstructed)
        if pos > 0:
            return [TextChunk(delta_text[:pos])] + results
        if pos == 0:
            return results

        # Fallback: SentencePiece context-dependent decoding mismatch.
        # Rebuild from delta_text using PreLexedTerminals as split anchors.
        return self._rebuild_from_anchors(delta_text, results)

    def _join_decoded_text(self, results: list[LexerInput]) -> str:
        """Join TextChunk and PreLexedTerminal text into one string."""
        parts: list[str] = []
        for item in results:
            if isinstance(item, (TextChunk, PreLexedTerminal)):
                parts.append(item.text)
        return "".join(parts)

    def _rebuild_from_anchors(
        self,
        delta_text: str,
        results: list[LexerInput],
    ) -> list[LexerInput]:
        """Rebuild results from delta_text using terminals as anchors.

        When context-dependent decoding creates a mismatch between
        individually-decoded tokens and delta_text, use
        PreLexedTerminals as split points and reallocate text from
        delta_text.  If a terminal's text is not found in delta_text,
        it is deferred to the next scan() call.

        Anchor positions first use adjacent decoded ordinary-token spans.
        This distinguishes real control IDs from identical literal text before
        or after them. When neither adjacent span matches (for example, a
        tokenizer with context-dependent whitespace), retain the existing
        right-to-left textual recovery for the remaining anchors.
        """
        anchors = [item for item in results if isinstance(item, PreLexedTerminal)]
        if not anchors:
            return [TextChunk(delta_text)]

        token_groups: list[list[str]] = [[] for _ in range(len(anchors) + 1)]
        count_groups = [0] * (len(anchors) + 1)
        text_groups = [""] * (len(anchors) + 1)
        group_idx = 0
        for item in results:
            if isinstance(item, PreLexedTerminal):
                group_idx += 1
            else:
                token_groups[group_idx].extend(item.token_texts)
                count_groups[group_idx] += item.token_count
                text_groups[group_idx] += item.text

        # Resolve positions right-to-left: each anchor gets the
        # rightmost occurrence that is still before the next anchor.
        positions: list[int] = [-1] * len(anchors)
        reconstructed = self._join_decoded_text(results)
        exact_prefix = reconstructed.startswith(delta_text)
        if exact_prefix:
            # Stop-string buffering can deliver only a prefix of the tokens.
            # Use their exact positions; a later control ID may not be visible.
            offset = 0
            anchor_idx = 0
            for item in results:
                if isinstance(item, PreLexedTerminal):
                    if offset + len(item.text) <= len(delta_text):
                        positions[anchor_idx] = offset
                    anchor_idx += 1
                offset += len(item.text)
        search_end = len(delta_text)
        for i in range(len(anchors) - 1, -1, -1):
            if exact_prefix:
                break
            # Retain the token-derived following span when locating a marker.
            # A literal lookalike after the real marker is part of that span,
            # not another candidate for the control-token boundary.
            following_text = text_groups[i + 1]
            pos = (
                delta_text.rfind(anchors[i].text + following_text, 0, search_end)
                if following_text
                else -1
            )
            if pos < 0 and text_groups[i]:
                # The following span may end in an unfinished Unicode byte
                # sequence held back by the detokenizer. In that case the
                # preceding token-derived span still identifies this boundary.
                preceding_text = text_groups[i]
                prefix_pos = delta_text.rfind(
                    preceding_text + anchors[i].text, 0, search_end
                )
                if prefix_pos >= 0:
                    pos = prefix_pos + len(preceding_text)
            if pos < 0:
                pos = delta_text.rfind(anchors[i].text, 0, search_end)
            if pos >= 0:
                positions[i] = pos
                search_end = pos

        # Build results left-to-right using the resolved positions.
        new_results: list[LexerInput] = []
        consumed = 0
        for i, anchor in enumerate(anchors):
            pos = positions[i]
            if pos >= consumed:
                if pos > consumed:
                    new_results.append(
                        TextChunk(
                            delta_text[consumed:pos],
                            tuple(token_groups[i]),
                            count_groups[i],
                        )
                    )
                new_results.append(anchor)
                consumed = pos + len(anchor.text)
            else:
                has_later_valid = any(p >= 0 for p in positions[i + 1 :])
                # DROP anchors (EOS, etc.) may have text that never
                # arrives in delta_text (stripped by detokenizer).
                # Don't defer remaining content waiting for text
                # that will never come.
                if (
                    not has_later_valid
                    and consumed < len(delta_text)
                    and anchor.terminal != DROP_TERMINAL
                ):
                    self._deferred_post_text += delta_text[consumed:]
                    consumed = len(delta_text)
                self._deferred_terminals.append(anchor)
                self._deferred_prefix_token_counts.append(count_groups[i])
                self._deferred_prefix_texts.append(text_groups[i])
        if consumed < len(delta_text):
            new_results.append(
                TextChunk(
                    delta_text[consumed:],
                    tuple(token_groups[-1]),
                    count_groups[-1],
                )
            )
        return new_results
