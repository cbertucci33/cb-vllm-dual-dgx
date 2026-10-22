# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TokenIDScanner."""

from unittest.mock import MagicMock

import pytest

from vllm.parser.engine.events import EventType
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.engine.token_id_scanner import (
    PreLexedTerminal,
    TextChunk,
    TokenIDScanner,
)
from vllm.parser.gemma4 import gemma4_config

CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
CHANNEL_START_ID = 100
CHANNEL_END_ID = 101
REGULAR_TOKEN_ID = 200
TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
TOOL_START_ID = 110
TOOL_END_ID = 111


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.get_vocab.return_value = {
        CHANNEL_START: CHANNEL_START_ID,
        CHANNEL_END: CHANNEL_END_ID,
    }
    tok.decode.side_effect = lambda ids: {
        CHANNEL_START_ID: CHANNEL_START,
        CHANNEL_END_ID: CHANNEL_END,
        REGULAR_TOKEN_ID: "regular",
    }.get(ids[0], f"<unk:{ids[0]}>")
    return tok


@pytest.fixture
def scanner(tokenizer):
    return TokenIDScanner(
        token_id_to_terminal={
            CHANNEL_START_ID: "THINK_START",
            CHANNEL_END_ID: "THINK_END",
        },
        tokenizer=tokenizer,
    )


class TestJoinDecodedTextReturnsStr:
    """_join_decoded_text always returns str."""

    @pytest.fixture
    def bare_scanner(self):
        return TokenIDScanner({}, tokenizer=None)

    def test_mixed_items(self, bare_scanner):
        items = [
            TextChunk("hello "),
            PreLexedTerminal("TOOL_START", 42, "<tool_call>"),
            TextChunk(" world"),
        ]
        result = bare_scanner._join_decoded_text(items)
        assert isinstance(result, str)
        assert result == "hello <tool_call> world"

    def test_empty_list(self, bare_scanner):
        result = bare_scanner._join_decoded_text([])
        assert isinstance(result, str)
        assert result == ""

    def test_only_text_chunks(self, bare_scanner):
        result = bare_scanner._join_decoded_text([TextChunk("abc"), TextChunk("def")])
        assert result == "abcdef"


class TestHoldbackTextRecovery:
    def test_holdback_text_with_special_token_text_absent(self, scanner):
        """Terminal deferred when its text is absent from delta_text."""
        result = scanner.scan(
            delta_text="processed is appropriate.",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 0

        result2 = scanner.scan(
            delta_text="<channel|>Understood.",
            delta_token_ids=[20, 21],
        )
        pre_lexed = [r for r in result2 if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"
        texts = [r.text for r in result2 if isinstance(r, TextChunk)]
        combined = "".join(texts)
        assert "processed is appropriate." in combined
        assert "Understood." in combined

    def test_holdback_text_with_special_token_text_present(self, scanner):
        """Hold-back text + special token text both in delta_text."""
        result = scanner.scan(
            delta_text="holdback text<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "holdback text"
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_END"

    def test_no_holdback_text(self, scanner):
        """delta_text is exactly the special token text."""
        result = scanner.scan(
            delta_text="<channel|>",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 1
        assert isinstance(result[0], PreLexedTerminal)
        assert result[0].terminal == "THINK_END"

    def test_empty_delta_text(self, scanner):
        """Empty delta_text defers the terminal until text arrives."""
        result = scanner.scan(
            delta_text="",
            delta_token_ids=[CHANNEL_END_ID],
        )

        assert len(result) == 0

        flushed = scanner.flush_pending()
        assert len(flushed) == 1
        assert isinstance(flushed[0], PreLexedTerminal)
        assert flushed[0].terminal == "THINK_END"

    def test_empty_delta_text_drops_individual_decode_text(self, tokenizer):
        """Empty delta_text with multiple tokens: all results deferred."""
        tool_start_id = 400
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tool_start_id: "<|tool_call>",
            tok_a: "call:",
            tok_b: "get_weather",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={tool_start_id: "TOOL_START"},
            tokenizer=tokenizer,
        )

        result = scanner.scan(
            delta_text="",
            delta_token_ids=[tool_start_id, tok_a, tok_b],
        )

        assert len(result) == 0

        flushed = scanner.flush_pending()
        assert len(flushed) == 2
        assert isinstance(flushed[0], PreLexedTerminal)
        assert flushed[0].terminal == "TOOL_START"
        assert isinstance(flushed[1], TextChunk)
        assert flushed[1].text == ""
        assert flushed[1].token_count == 2

    def test_deferred_terminal_preserves_trailing_token_count(self, tokenizer):
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            201: "alpha",
            202: "beta",
        }[ids[0]]
        scanner = TokenIDScanner({CHANNEL_START_ID: "THINK_START"}, tokenizer)

        assert scanner.scan("", [CHANNEL_START_ID, 201, 202]) == []

        result = scanner.scan(f"{CHANNEL_START}alphabeta", [])
        assert isinstance(result[0], PreLexedTerminal)
        assert isinstance(result[1], TextChunk)
        assert result[1].text == "alphabeta"
        assert result[1].token_count == 2

    def test_holdback_before_start_tag(self, scanner):
        result = scanner.scan(
            delta_text="prefix text<|channel>",
            delta_token_ids=[CHANNEL_START_ID],
        )

        assert len(result) == 2
        assert isinstance(result[0], TextChunk)
        assert result[0].text == "prefix text"
        assert isinstance(result[1], PreLexedTerminal)
        assert result[1].terminal == "THINK_START"

    def test_multi_token_batch_special_in_middle(self, scanner, tokenizer):
        """Multi-token batch with special token in the middle."""
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tok_a: "wordA",
            tok_b: "wordB",
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], "?")

        scanner_multi = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner_multi.scan(
            delta_text="holdback wordA<channel|> wordB",
            delta_token_ids=[tok_a, CHANNEL_END_ID, tok_b],
        )

        texts = [r.text for r in result if isinstance(r, TextChunk)]
        terminals = [r.terminal for r in result if isinstance(r, PreLexedTerminal)]
        assert "THINK_END" in terminals
        assert "holdback wordA" in "".join(texts)

    def test_multi_token_batch_special_token_text_absent(self, scanner, tokenizer):
        """Multi-token batch where special token text is absent."""
        tok_a = 201
        tok_b = 202
        tokenizer.decode.side_effect = lambda ids: {
            tok_a: "alpha",
            tok_b: "beta",
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], "?")

        scanner_multi = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner_multi.scan(
            delta_text="holdback alpha",
            delta_token_ids=[tok_a, CHANNEL_END_ID, tok_b],
        )

        assert len(result) == 0

        result2 = scanner_multi.scan(
            delta_text="<channel|> more text",
            delta_token_ids=[300],
        )
        pre_lexed = [r for r in result2 if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"
        text_chunks = [r for r in result2 if isinstance(r, TextChunk)]
        combined = "".join(t.text for t in text_chunks)
        assert "holdback alpha" in combined
        assert "more text" in combined

    def test_holdback_with_content_after_special_token(self, tokenizer):
        """Hold-back + special token + content after in one delta."""
        tok_content = 210
        tokenizer.decode.side_effect = lambda ids: {
            CHANNEL_END_ID: CHANNEL_END,
            tok_content: "content start",
        }.get(ids[0], "?")

        scanner = TokenIDScanner(
            token_id_to_terminal={CHANNEL_END_ID: "THINK_END"},
            tokenizer=tokenizer,
        )

        result = scanner.scan(
            delta_text="reasoning end.<channel|>content start",
            delta_token_ids=[CHANNEL_END_ID, tok_content],
        )

        pre_lexed = [r for r in result if isinstance(r, PreLexedTerminal)]
        assert len(pre_lexed) == 1
        assert pre_lexed[0].terminal == "THINK_END"

        text_chunks = [r for r in result if isinstance(r, TextChunk)]
        combined = "".join(t.text for t in text_chunks)
        assert "reasoning end." in combined


class TestEndToEndReasoningHoldback:
    """End-to-end engine tests with detokenizer hold-back."""

    def test_reasoning_content_not_truncated(self):
        config = gemma4_config()
        tok = MagicMock()
        vocab = {
            CHANNEL_START: CHANNEL_START_ID,
            CHANNEL_END: CHANNEL_END_ID,
        }
        tok.get_vocab.return_value = vocab
        tok.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], f"tok{ids[0]}")

        engine = StreamingParserEngine(config, tok)
        all_events = []

        all_events.extend(engine.feed(CHANNEL_START, [CHANNEL_START_ID]))
        all_events.extend(
            engine.feed(
                "thought\nThe request was received and ",
                [10, 11, 12, 13, 14],
            )
        )
        # CHANNEL_END token arrives but its text is held back.
        all_events.extend(
            engine.feed(
                "processed is appropriate.",
                [CHANNEL_END_ID],
            )
        )
        # Detokenizer flushes the held-back text.
        all_events.extend(
            engine.feed(
                "<channel|>Understood.",
                [20, 21],
            )
        )

        all_events.extend(engine.finish())

        reasoning_text = "".join(
            e.value for e in all_events if e.type == EventType.REASONING_CHUNK
        )
        content_text = "".join(
            e.value for e in all_events if e.type == EventType.TEXT_CHUNK
        )

        assert "processed is appropriate." in reasoning_text
        assert "Understood." in content_text

    def test_backtick_content_not_truncated(self):
        config = gemma4_config()
        tok = MagicMock()
        vocab = {
            CHANNEL_START: CHANNEL_START_ID,
            CHANNEL_END: CHANNEL_END_ID,
        }
        tok.get_vocab.return_value = vocab
        tok.decode.side_effect = lambda ids: {
            CHANNEL_START_ID: CHANNEL_START,
            CHANNEL_END_ID: CHANNEL_END,
        }.get(ids[0], f"tok{ids[0]}")

        engine = StreamingParserEngine(config, tok)
        all_events = []

        all_events.extend(engine.feed(CHANNEL_START, [CHANNEL_START_ID]))
        all_events.extend(
            engine.feed(
                "thought\n1/10 completed. Next: ",
                [10, 11, 12, 13],
            )
        )
        all_events.extend(
            engine.feed(
                "`hostname`.\n",
                [CHANNEL_END_ID],
            )
        )
        all_events.extend(
            engine.feed(
                "<channel|>tool output",
                [20, 21],
            )
        )

        all_events.extend(engine.finish())

        reasoning_text = "".join(
            e.value for e in all_events if e.type == EventType.REASONING_CHUNK
        )

        assert "`hostname`." in reasoning_text


_CHANNEL_START_TAG = "<|channel>"
_CHANNEL_END_TAG = "<channel|>"
_TOOL_START_TAG = "<|tool_call>"
_TOOL_END_TAG = "<tool_call|>"
_QUOTE_TAG = '<|"|>'

_CHANNEL_START_TID = 100
_CHANNEL_END_TID = 101
_TOOL_START_TID = 102
_TOOL_END_TID = 103
_QUOTE_TID = 104
_TOK = list(range(200, 215))


def _gemma4_vocab() -> dict[str, int]:
    return {
        _CHANNEL_START_TAG: _CHANNEL_START_TID,
        _CHANNEL_END_TAG: _CHANNEL_END_TID,
        _TOOL_START_TAG: _TOOL_START_TID,
        _TOOL_END_TAG: _TOOL_END_TID,
        _QUOTE_TAG: _QUOTE_TID,
    }


def _make_gemma4_tokenizer(
    extra_decode: dict[int, str] | None = None,
) -> MagicMock:
    special = {
        _CHANNEL_START_TID: _CHANNEL_START_TAG,
        _CHANNEL_END_TID: _CHANNEL_END_TAG,
        _TOOL_START_TID: _TOOL_START_TAG,
        _TOOL_END_TID: _TOOL_END_TAG,
        _QUOTE_TID: _QUOTE_TAG,
    }
    decode_map = {**special, **(extra_decode or {})}

    tok = MagicMock()
    tok.get_vocab.return_value = _gemma4_vocab()
    tok.decode.side_effect = lambda ids: decode_map.get(ids[0], f"tok{ids[0]}")
    return tok


def _collect_events(engine, deltas):
    from vllm.parser.engine.events import SemanticEvent

    all_events: list[SemanticEvent] = []
    for delta_text, delta_token_ids in deltas:
        all_events.extend(engine.feed(delta_text, delta_token_ids))
    all_events.extend(engine.finish())
    return all_events


def _reasoning_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.REASONING_CHUNK)


def _content_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.TEXT_CHUNK)


def _arg_text(events) -> str:
    return "".join(e.value for e in events if e.type == EventType.ARG_VALUE_CHUNK)


def _has_event(events, event_type) -> bool:
    return any(e.type == event_type for e in events)


class TestMultiTokenBoundaryPreservation:
    """No text lost at state boundaries with multi-token deltas."""

    def test_empty_delta_text_at_channel_end_unified(self):
        """Empty delta_text when CHANNEL_END arrives; text comes later."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                ("", [_CHANNEL_START_TID]),
                ("<|channel>thought\nSome reasoning.", [_TOK[0], _TOK[1]]),
                ("", [_CHANNEL_END_TID]),
                ("<channel|>Final answer.", [_TOK[2], _TOK[3]]),
            ],
        )

        reasoning = _reasoning_text(events)
        content = _content_text(events)
        assert "Some reasoning." in reasoning
        assert "Final answer." in content
        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)

    def test_deferred_channel_end_flushed_at_finish_unified(self):
        """Deferred CHANNEL_END flushed at end-of-stream."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nReasoning text.", [_TOK[0]]),
                (" Final thought.", [_CHANNEL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        assert "Reasoning text. Final thought." in reasoning
        assert _has_event(events, EventType.REASONING_END)

    def test_reasoning_to_tool_call_handoff_unified(self):
        """Full reasoning -> content -> tool call flow."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nI need to check the weather.", [_TOK[0], _TOK[1], _TOK[2]]),
                (_CHANNEL_END_TAG, [_CHANNEL_END_TID]),
                ("Let me call a tool.", [_TOK[3], _TOK[4]]),
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[5], _TOK[6]]),
                ('<|"|>SF<|"|>}', [_QUOTE_TID, _TOK[7], _QUOTE_TID, _TOK[8]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        assert "I need to check the weather." in reasoning
        assert "Let me call a tool." in content
        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)
        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "SF" in _arg_text(events)

    def test_multiple_tool_calls_rapid_transitions_unified(self):
        """Two back-to-back tool calls with correct tool_index tracking."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[0], _TOK[1]]),
                ('<|"|>NYC<|"|>}', [_QUOTE_TID, _TOK[2], _QUOTE_TID, _TOK[3]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_time{tz:", [_TOK[4], _TOK[5]]),
                ('<|"|>EST<|"|>}', [_QUOTE_TID, _TOK[6], _QUOTE_TID, _TOK[7]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
        ends = [e for e in events if e.type == EventType.TOOL_CALL_END]
        assert len(starts) == 2
        assert len(ends) == 2
        assert starts[0].tool_index == 0
        assert starts[1].tool_index == 1

        names = "".join(e.value for e in events if e.type == EventType.TOOL_NAME)
        assert "get_weather" in names
        assert "get_time" in names

    def test_deferred_channel_end_before_tool_call_unified(self):
        """Deferred CHANNEL_END followed by a tool call."""
        tok = _make_gemma4_tokenizer()
        engine = StreamingParserEngine(gemma4_config(), tok)

        events = _collect_events(
            engine,
            [
                (_CHANNEL_START_TAG, [_CHANNEL_START_TID]),
                ("thought\nNeed to call a tool.", [_TOK[0], _TOK[1]]),
                (" Let me proceed.", [_CHANNEL_END_TID]),
                (_CHANNEL_END_TAG, [_TOK[2]]),
                (_TOOL_START_TAG, [_TOOL_START_TID]),
                ("call:get_weather{city:", [_TOK[3], _TOK[4]]),
                ('<|"|>Tokyo<|"|>}', [_QUOTE_TID, _TOK[5], _QUOTE_TID, _TOK[6]]),
                (_TOOL_END_TAG, [_TOOL_END_TID]),
            ],
        )

        reasoning = _reasoning_text(events)
        assert "Need to call a tool. Let me proceed." in reasoning
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)
        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "Tokyo" in _arg_text(events)


class TestStreamInterval10:
    """Tests with stream_interval=10 (large multi-token batches)."""

    def test_channel_end_mid_batch_text_present(self):
        """<channel|> mid-batch with its text present in delta_text."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"word{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        events.extend(
            engine.feed(
                "<|channel>thought\nword0 word1 word2 word3 word4 "
                "word5 word6 word7 word8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        events.extend(
            engine.feed(
                "word9 word10 word11 <channel|>word12 word13 word14 word0 word1 word2 ",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _CHANNEL_END_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        for w in ("word9", "word10", "word11"):
            assert w in reasoning, f"{w!r} missing from reasoning"

        for w in ("word12", "word13", "word14"):
            assert w in content, f"{w!r} missing from content"

        assert _has_event(events, EventType.REASONING_END)

    def test_channel_end_and_tool_start_same_batch_unified(self):
        """Both <channel|> and <|tool_call> in a single batch."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []

        events.extend(
            engine.feed(
                "<|channel>thought\nw0 w1 w2 w3 w4 w5 w6 w7 w8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        events.extend(
            engine.feed(
                "w9 w10 <channel|>w11 <|tool_call>",
                [
                    _TOK[9],
                    _TOK[10],
                    _CHANNEL_END_TID,
                    _TOK[11],
                    _TOOL_START_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                ],
            )
        )
        events.extend(engine.finish())

        reasoning = _reasoning_text(events)

        assert "w9" in reasoning
        assert "w10" in reasoning
        assert _has_event(events, EventType.REASONING_END)
        assert _has_event(events, EventType.TOOL_CALL_START)

    def test_channel_end_mid_batch_text_absent(self):
        """<channel|> mid-batch with its text absent from delta_text."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"word{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        events.extend(
            engine.feed(
                "<|channel>thought\nword0 word1 word2 word3 word4 "
                "word5 word6 word7 word8 ",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        events.extend(
            engine.feed(
                "word9 word10 word11 ",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _CHANNEL_END_TID,
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        events.extend(
            engine.feed(
                "<channel|>word12 word13 word14 word0 word1 word2 ",
                [_TOK[3], _TOK[4], _TOK[5]],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        for w in ("word9", "word10", "word11"):
            assert w in reasoning, f"{w!r} missing from reasoning"

        for w in ("word12", "word13", "word14"):
            assert w in content, f"{w!r} missing from content"

        assert _has_event(events, EventType.REASONING_END)

    def test_tool_end_mid_batch_text_absent_unified(self):
        """<tool_call|> mid-batch with text absent."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i}" for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []
        events.extend(
            engine.feed(
                _CHANNEL_START_TAG,
                [_CHANNEL_START_TID],
            )
        )
        events.extend(
            engine.feed(
                "thought\nNeed a tool.",
                [_TOK[0], _TOK[1]],
            )
        )
        events.extend(
            engine.feed(
                _TOOL_START_TAG,
                [_TOOL_START_TID],
            )
        )
        events.extend(
            engine.feed(
                "call:get_weather{city:",
                [_TOK[2], _TOK[3], _TOK[4]],
            )
        )

        events.extend(
            engine.feed(
                '<|"|>San Francisco<|"|>}',
                [
                    _QUOTE_TID,
                    _TOK[5],
                    _TOK[6],
                    _QUOTE_TID,
                    _TOK[7],
                    _TOOL_END_TID,
                    _TOK[8],
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                ],
            )
        )

        events.extend(
            engine.feed(
                "<tool_call|>w8w9w10w11w12",
                [_TOK[12], _TOK[13]],
            )
        )

        events.extend(engine.finish())

        assert _has_event(events, EventType.TOOL_CALL_END)
        assert "San Francisco" in _arg_text(events)

    def test_large_batch_holdback_spans_two_batches(self):
        """Holdback text spanning two batches with <channel|> in the second."""
        tok = _make_gemma4_tokenizer({_TOK[i]: f"w{i} " for i in range(15)})
        engine = StreamingParserEngine(gemma4_config(), tok)

        events: list = []

        events.extend(
            engine.feed(
                "<|channel>thought\nThe user asked about machine learning "
                "and I need to think about the best approach to",
                [
                    _CHANNEL_START_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                ],
            )
        )

        events.extend(
            engine.feed(
                " explain this complex topic. Let me organize my thoughts.",
                [
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _TOK[12],
                    _TOK[13],
                    _TOK[14],
                    _CHANNEL_END_TID,
                    _TOK[0],
                    _TOK[1],
                    _TOK[2],
                ],
            )
        )

        events.extend(
            engine.feed(
                "<channel|>w0 w1 w2 Here is what I recommend: start with "
                "the fundamentals and build up from there.",
                [
                    _TOK[3],
                    _TOK[4],
                    _TOK[5],
                    _TOK[6],
                    _TOK[7],
                    _TOK[8],
                    _TOK[9],
                    _TOK[10],
                    _TOK[11],
                    _TOK[12],
                ],
            )
        )

        events.extend(engine.finish())

        reasoning = _reasoning_text(events)
        content = _content_text(events)

        assert "organize my thoughts." in reasoning
        assert "explain" in reasoning
        assert "recommend" in content

        assert _has_event(events, EventType.REASONING_START)
        assert _has_event(events, EventType.REASONING_END)


class TestRebuildFromAnchorsLiteralLookalike:
    """Literal token text in prose must not be consumed as an anchor."""

    @pytest.fixture
    def tool_scanner(self):
        tok = MagicMock()
        tok.get_vocab.return_value = {
            TOOL_START: TOOL_START_ID,
            TOOL_END: TOOL_END_ID,
        }
        tok.decode.side_effect = lambda ids: {
            TOOL_START_ID: TOOL_START,
            TOOL_END_ID: TOOL_END,
        }.get(ids[0], f"t{ids[0]}")
        return TokenIDScanner(
            {TOOL_START_ID: "TOOL_START", TOOL_END_ID: "TOOL_END"},
            tok,
        )

    def test_literal_before_real_anchor(self, tool_scanner):
        delta_text = 'Use <tool_call> like this: <tool_call>{"name":"f"}</tool_call>'
        delta_token_ids = [1, 2, 3, 4, 5, TOOL_START_ID, 6, 7, TOOL_END_ID]
        items = tool_scanner.scan(delta_text, delta_token_ids)

        text_parts = [it.text for it in items if isinstance(it, TextChunk)]
        terminals = [it for it in items if isinstance(it, PreLexedTerminal)]

        assert len(terminals) == 2
        assert terminals[0].terminal == "TOOL_START"
        assert terminals[1].terminal == "TOOL_END"

        joined_text = "".join(text_parts)
        assert "<tool_call>" in joined_text
        assert '{"name":"f"}' in joined_text

    def test_multiple_tool_calls_with_literal_between(self, tool_scanner):
        delta_text = (
            '<tool_call>{"name":"a"}</tool_call>'
            " see <tool_call> syntax "
            '<tool_call>{"name":"b"}</tool_call>'
        )
        delta_token_ids = [
            TOOL_START_ID,
            1,
            TOOL_END_ID,
            2,
            3,
            4,
            TOOL_START_ID,
            5,
            TOOL_END_ID,
        ]
        items = tool_scanner.scan(delta_text, delta_token_ids)

        terminals = [it for it in items if isinstance(it, PreLexedTerminal)]
        assert len(terminals) == 4

        text_parts = [it.text for it in items if isinstance(it, TextChunk)]
        joined_text = "".join(text_parts)
        assert "<tool_call> syntax" in joined_text


class TestRebuildFromAnchorsCascadingDeferral:
    """Missing middle anchor defers only itself, not subsequent ones."""

    @pytest.fixture
    def bare_scanner(self):
        tok = MagicMock()
        tok.decode.side_effect = lambda ids: f"t{ids[0]}"
        return TokenIDScanner({}, tok)

    def test_middle_anchor_missing_does_not_cascade(self, bare_scanner):
        a = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        b = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        c = PreLexedTerminal("TOOL_END", TOOL_END_ID, TOOL_END)
        delta_text = f"prefix{TOOL_START}middle{TOOL_END}suffix"
        results = [a, b, c]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        texts = [r for r in rebuilt if isinstance(r, TextChunk)]
        joined = "".join(t.text for t in texts)

        assert len(terminals) == 2
        assert terminals[0].terminal == "TOOL_START"
        assert terminals[1].terminal == "TOOL_END"
        assert "prefix" in joined
        assert "middle" in joined
        assert "suffix" in joined
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"
        assert bare_scanner._deferred_post_text == ""

    def test_first_anchor_missing_rest_still_emitted(self, bare_scanner):
        a = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        b = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        delta_text = f"text{TOOL_START}more"
        results = [a, b]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        assert len(terminals) == 1
        assert terminals[0].terminal == "TOOL_START"
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"

    def test_last_anchor_missing_preceding_still_emitted(self, bare_scanner):
        a = PreLexedTerminal("TOOL_START", TOOL_START_ID, TOOL_START)
        b = PreLexedTerminal("THINK_END", CHANNEL_END_ID, CHANNEL_END)
        delta_text = f"text{TOOL_START}more"
        results = [a, b]

        rebuilt = bare_scanner._rebuild_from_anchors(delta_text, results)

        terminals = [r for r in rebuilt if isinstance(r, PreLexedTerminal)]
        assert len(terminals) == 1
        assert terminals[0].terminal == "TOOL_START"
        texts = [r for r in rebuilt if isinstance(r, TextChunk)]
        joined = "".join(t.text for t in texts)
        assert "text" in joined
        assert bare_scanner._deferred_post_text == "more"
        assert len(bare_scanner._deferred_terminals) == 1
        assert bare_scanner._deferred_terminals[0].terminal == "THINK_END"


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize("glyph", ["漢", "𐐀"])
@pytest.mark.parametrize("schedule", ["full", "whole", "two_chunks"])
def test_actual_tokenizer_literal_after_real_reasoning_end(profile, glyph, schedule):
    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, cls = load_local_model_parser(profile)
    reasoning = "reason" + glyph
    content = 'Print literal "</think>" exactly.'
    prefix = tokenizer.encode(
        "<think>" + reasoning + "</think>", add_special_tokens=False
    )
    # Literal markup is ordinary token IDs, not a reasoning control token.
    ordinary = [
        tid
        for char in content
        for tid in tokenizer.encode(char, add_special_tokens=False)
    ]
    assert tokenizer.get_vocab()["</think>"] not in ordinary
    ids = prefix + ordinary
    text = tokenizer.decode(ids)
    request = ResponsesRequest(model="test", input="hi", include_reasoning=True)
    if schedule == "full":
        assert cls(tokenizer).parse(text, request, True, ids)[:2] == (
            reasoning,
            content,
        )
        return
    schedules = (
        [[len(ids)]]
        if schedule == "whole"
        else [[split, len(ids)] for split in range(1, len(ids))]
    )
    for ends in schedules:
        parser = cls(tokenizer)
        deltas = []
        start, previous = 0, ""
        for end in ends:
            decoded = tokenizer.decode(ids[:end]).rstrip("\ufffd")
            assert decoded.startswith(previous)
            deltas.extend(
                parser.parse_deltas(
                    decoded[len(previous) :],
                    ids[start:end],
                    request,
                    finished=end == len(ids),
                )
            )
            start, previous = end, decoded
        assert "".join(d.reasoning or "" for d in deltas) == reasoning, ends
        assert "".join(d.content or "" for d in deltas) == content, ends


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize("detokenizer_kind", ["fast", "slow"])
@pytest.mark.parametrize("holdback", [False, True])
@pytest.mark.parametrize("literal_reasoning", [False, True])
def test_real_detokenizer_anchor_alignment(
    profile, detokenizer_kind, holdback, literal_reasoning
):
    from types import SimpleNamespace

    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.detokenizer import (
        FastIncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )

    tokenizer, parser_cls = load_local_model_parser(profile)
    reasoning = (
        "reason𐐀 literal </think> before the end." if literal_reasoning else "reason𐐀"
    )
    content = 'Print literal "</think>" exactly.𐐀 END.'

    def ordinary(text):
        return [
            tid
            for char in text
            for tid in tokenizer.encode(char, add_special_tokens=False)
        ]

    ids = (
        tokenizer.encode("<think>", add_special_tokens=False)
        + ordinary(reasoning)
        + tokenizer.encode("</think>", add_special_tokens=False)
        + ordinary(content)
    )
    detokenizer_cls = (
        FastIncrementalDetokenizer
        if detokenizer_kind == "fast"
        else SlowIncrementalDetokenizer
    )
    schedules = [[len(ids)], list(range(1, len(ids) + 1))]
    schedules.extend([split, len(ids)] for split in range(1, len(ids)))
    for ends in schedules:
        sampling = SamplingParams(
            skip_special_tokens=False,
            stop=["unused_stop_marker"] if holdback else None,
        )
        core_request = SimpleNamespace(
            sampling_params=sampling,
            prompt_token_ids=[],
            prompt_embeds=None,
            request_id="scanner-boundary",
        )
        detokenizer = detokenizer_cls(tokenizer, core_request)
        parser = parser_cls(tokenizer)
        request = ResponsesRequest(model="test", input="hi", include_reasoning=True)
        deltas = []
        start = 0
        for end in ends:
            detokenizer.update(ids[start:end], stop_terminated=False)
            finished = end == len(ids)
            text = detokenizer.get_next_output_text(finished, delta=True)
            deltas.extend(
                parser.parse_deltas(text, ids[start:end], request, finished=finished)
            )
            expected_count = min(max(end - 1, 0), len(ordinary(reasoning)))
            assert parser.count_reasoning_tokens(ids[:end]) == expected_count, ends
            start = end
        assert "".join(d.reasoning or "" for d in deltas) == reasoning, ends
        assert "".join(d.content or "" for d in deltas) == content, ends


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize("detokenizer_kind", ["fast", "slow"])
@pytest.mark.parametrize("holdback", [False, True])
@pytest.mark.parametrize("prompt_kind", ["empty", "thinking", "partial_utf8"])
@pytest.mark.parametrize("spaces_between_special_tokens", [False, True])
def test_native_positions_three_chunk_unicode(
    profile, detokenizer_kind, holdback, prompt_kind, spaces_between_special_tokens
):
    from types import SimpleNamespace

    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.detokenizer import (
        FastIncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )

    tok, cls = load_local_model_parser(profile)
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    ordinary = lambda text: [tid for char in text for tid in enc(char)]
    glyph = enc("𐐀")
    prefix = enc("<think>")
    prompt = []
    first_glyph = glyph
    if prompt_kind != "empty":
        prompt = enc("user input\n") + prefix
        prefix = []
        if prompt_kind == "partial_utf8":
            prompt += glyph[:1]
            first_glyph = glyph[1:]
    middle = enc("</think>") + ordinary("x</think>")
    ids = prefix + first_glyph + middle + glyph + enc(" end")
    second_start = len(prefix) + len(first_glyph) + len(middle)
    schedules = [
        [len(prefix) + a, second_start + b, len(ids)]
        for a in range(1, len(first_glyph))
        for b in range(1, len(glyph))
    ]
    detok_cls = (
        FastIncrementalDetokenizer
        if detokenizer_kind == "fast"
        else SlowIncrementalDetokenizer
    )
    for ends in schedules:
        sampling = SamplingParams(
            skip_special_tokens=False,
            spaces_between_special_tokens=spaces_between_special_tokens,
            stop=["unused_stop_marker"] if holdback else None,
        )
        detok = detok_cls(
            tok,
            SimpleNamespace(
                sampling_params=sampling,
                prompt_token_ids=prompt,
                prompt_embeds=None,
                request_id="three-chunk-regression",
            ),
        )
        parser = cls(tok)
        engine = parser._reasoning_parser._parser_engine
        assert engine._engine._scanner.tracks_token_positions
        request = ResponsesRequest(model="test", input="hi", include_reasoning=True)
        deltas = []
        start = 0
        for end in ends:
            detok.update(ids[start:end], stop_terminated=False)
            finished = end == len(ids)
            text = detok.get_next_output_text(finished, delta=True)
            deltas.extend(
                parser.parse_deltas(
                    text, ids[start:end], request, prompt, finished=finished
                )
            )
            start = end
        assert "".join(d.reasoning or "" for d in deltas) == "𐐀", ends
        assert "".join(d.content or "" for d in deltas) == "x</think>𐐀 end", ends
        assert parser.count_reasoning_tokens(ids) == len(first_glyph)
        positions = parser._reasoning_parser._parser_engine._engine._scanner._positions
        assert not positions.pending_text and not positions.terminals
        parser._reasoning_parser._parser_engine._reset()
        assert positions.stream is None and positions.emitted_chars == 0


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize("detokenizer_kind", ["fast", "slow"])
@pytest.mark.parametrize("stop_kind", ["ordinary", "control", "partial_control", "eos"])
@pytest.mark.parametrize("include_stop", [False, True])
def test_native_positions_stop_truncation(
    profile, detokenizer_kind, stop_kind, include_stop
):
    from types import SimpleNamespace

    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.detokenizer import (
        FastIncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )

    tok, cls = load_local_model_parser(profile)
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    ordinary = lambda text: [tid for char in text for tid in enc(char)]
    ids = enc("<think>𐐀</think>") + ordinary("safe </think> STOP hidden")
    stop = {
        "ordinary": ["STOP"],
        "control": ["</think>"],
        "partial_control": ["</think>"[:5]],
        "eos": None,
    }[stop_kind]
    if stop_kind == "eos":
        ids = enc("<think>𐐀</think>") + ordinary("safe </think>")
        ids += [tok.eos_token_id]
    else:
        ids += enc("<think>not delivered</think>") + ordinary("nor this")
    sampling = SamplingParams(
        skip_special_tokens=False,
        stop=stop,
        include_stop_str_in_output=include_stop,
    )
    detok_cls = (
        FastIncrementalDetokenizer
        if detokenizer_kind == "fast"
        else SlowIncrementalDetokenizer
    )
    detok = detok_cls(
        tok,
        SimpleNamespace(
            sampling_params=sampling,
            prompt_token_ids=[],
            prompt_embeds=None,
            request_id="stop-position-regression",
        ),
    )
    detok.update(ids, stop_terminated=stop_kind == "eos")
    text = detok.get_next_output_text(finished=True, delta=True)
    parser = cls(tok)
    request = ResponsesRequest(model="test", input="hi", include_reasoning=True)
    deltas = parser.parse_deltas(text, ids, request, finished=True)
    assert "".join(d.reasoning or "" for d in deltas) == "𐐀"
    expected = (
        "safe </think> " + ("STOP" if include_stop else "")
        if stop_kind == "ordinary"
        else "safe </think>"
        if stop_kind == "eos"
        else ""
    )
    assert "".join(d.content or "" for d in deltas) == expected
    positions = parser._reasoning_parser._parser_engine._engine._scanner._positions
    assert not positions.pending_text
    assert not positions.terminals


@pytest.mark.parametrize("split", [1, 2, 3])
@pytest.mark.parametrize("fused_prefix", [False, True])
@pytest.mark.parametrize("literal_replacement", [False, True])
def test_native_partial_prefill_owns_only_prompt_text(
    split, fused_prefix, literal_replacement
):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel
    from tokenizers.models import WordLevel

    from vllm.tokenizers.streaming import create_decode_stream

    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_map = {value: chr(value) for value in visible}
    for value in range(256):
        if value not in byte_map:
            byte_map[value] = chr(256 + len(byte_map) - len(visible))

    def raw(data):
        return "".join(byte_map[value] for value in data)

    prefix = "prompt \ufffd" if literal_replacement else "prompt "
    pieces = [raw(prefix.encode()), *[raw(bytes([b])) for b in "𐐀".encode()]]
    if fused_prefix:
        pieces[1] = pieces[0] + pieces[1]
    pieces += [raw(" suffix\ufffd end".encode())]
    vocab = {piece: idx for idx, piece in enumerate(dict.fromkeys(pieces))}
    piece_ids = [vocab[piece] for piece in pieces]
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[unk]"))
    tokenizer.decoder = ByteLevel()
    prompt = ([piece_ids[0]] if not fused_prefix else []) + piece_ids[1 : split + 1]
    output = piece_ids[split + 1 :]
    assert tokenizer.decode(prompt + output) == prefix + "𐐀 suffix\ufffd end"
    stream = create_decode_stream(tokenizer, prompt)
    assert "".join(stream.step(tokenizer, tid) or "" for tid in output) == (
        "𐐀 suffix\ufffd end"
    )

    # A genuine replacement character at the end of a prompt is owned by the
    # prompt; an identical character in generated output must remain visible.
    replacement = raw("\ufffd".encode())
    tokenizer.add_tokens([replacement])
    replacement_id = tokenizer.token_to_id(replacement)
    stream = create_decode_stream(tokenizer, [piece_ids[0], replacement_id])
    assert stream.step(tokenizer, piece_ids[-1]) == " suffix\ufffd end"
