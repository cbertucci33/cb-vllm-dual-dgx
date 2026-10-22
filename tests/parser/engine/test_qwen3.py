# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Qwen3 tool call parser.

These validate that the engine-driven parser correctly handles
Qwen3 XML-style tool calls.
"""

import dataclasses
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
    simulate_tool_streaming,
)
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.qwen3 import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    qwen3_config,
)


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(
        {
            TOOL_CALL_START: 100,
            TOOL_CALL_END: 101,
        }
    )


@pytest.fixture
def parser(mock_tokenizer):
    return ParserEngine(
        mock_tokenizer,
        parser_engine_config=qwen3_config(thinking=False),
    )


class TestNonStreaming:
    def test_no_tool_calls(self, parser, mock_request):
        result = parser.extract_tool_calls(
            "This is a regular response without any tool calls.",
            mock_request,
        )
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == ("This is a regular response without any tool calls.")

    def test_single_tool_call(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Tokyo"}

    def test_parallel_tool_calls(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
            "<tool_call>\n"
            "<function=get_time>\n"
            "<parameter=timezone>Asia/Tokyo</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == {"city": "Tokyo"}
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"timezone": "Asia/Tokyo"}

    def test_various_data_types(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=test_function>\n"
            "<parameter=string_field>hello</parameter>\n"
            "<parameter=int_field>42</parameter>\n"
            "<parameter=float_field>3.14</parameter>\n"
            "<parameter=bool_field>true</parameter>\n"
            "<parameter=null_field>null</parameter>\n"
            '<parameter=array_field>["a", "b", "c"]</parameter>\n'
            '<parameter=object_field>{"nested": "value"}</parameter>\n'
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["string_field"] == "hello"
        assert args["int_field"] == "42"
        assert args["float_field"] == "3.14"
        assert args["bool_field"] == "true"
        assert args["null_field"] == "null"
        assert args["array_field"] == '["a", "b", "c"]'
        assert args["object_field"] == '{"nested": "value"}'

    def test_empty_arguments(self, parser, mock_request):
        text = "<tool_call>\n<function=refresh>\n</function>\n</tool_call>"
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "refresh"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}

    def test_surrounding_text(self, parser, mock_request):
        text = (
            "Let me check the weather for you.\n\n"
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n</tool_call>\n\n"
            "I will get that information."
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert result.content is not None
        assert "Let me check the weather" in result.content
        assert result.tool_calls[0].function.name == "get_weather"

    def test_escaped_strings(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=test_function>\n"
            '<parameter=quoted>He said "hello"</parameter>\n'
            "<parameter=path>C:\\Users\\file.txt</parameter>\n"
            "<parameter=newline>line1\nline2</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["quoted"] == 'He said "hello"'
        assert args["path"] == "C:\\Users\\file.txt"
        assert args["newline"] == "line1\nline2"

    def test_multiple_parameters(self, parser, mock_request):
        text = (
            "<tool_call>\n<function=search>\n"
            "<parameter=query>vllm parsing</parameter>\n"
            "<parameter=limit>10</parameter>\n"
            "<parameter=exact_match>false</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {
            "query": "vllm parsing",
            "limit": "10",
            "exact_match": "false",
        }

    def test_multiline_param_values(self, parser, mock_request):
        """Parameter values spanning multiple lines."""
        text = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "ls -la /tmp\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "List files in /tmp directory\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "Bash"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["command"] == "ls -la /tmp"
        assert args["description"] == "List files in /tmp directory"

    def test_multiline_two_tool_calls(self, parser, mock_request):
        """Two tool calls with multi-line parameter values (bug report)."""
        text = (
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "find /workspace -name '*.py' | head -20\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "Find Python files\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
            "<tool_call>\n"
            "<function=Read>\n"
            "<parameter=file_path>\n"
            "/workspace/main.py\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "Bash"
        assert result.tool_calls[1].function.name == "Read"
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert "find /workspace" in args0["command"]
        assert "Find Python files" in args0["description"]
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert "/workspace/main.py" in args1["file_path"]

    def test_consecutive_tool_calls_without_tool_end(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "</function>\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Paris</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0].function.arguments)
        assert args0 == {"city": "Tokyo"}
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args1 == {"city": "Paris"}

    def test_nested_json_array_parameter(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=AskUserQuestion>\n"
            "<parameter=questions>"
            '[{"question": "Pick a color",'
            ' "multiSelect": false, "answer": null}]'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)

        assert result.tools_called is True
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {
            "questions": '[{"question": "Pick a color",'
            ' "multiSelect": false, "answer": null}]',
        }


class TestStreaming:
    def test_basic_streaming(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo"}

    def test_streaming_multi_param(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "<parameter=unit>celsius</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "get_weather"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"city": "Tokyo", "unit": "celsius"}

    def test_streaming_args_arrive_incrementally(self, parser, mock_request):
        """Arguments must stream as intermediate deltas, not batch at
        tool-end."""
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "<parameter=unit>celsius</parameter>\n",
            "<parameter=days>5</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        arg_deltas: list[str] = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        arg_deltas.append(tc.function.arguments)

        assert len(arg_deltas) > 1, (
            f"Expected arguments across multiple deltas, got {len(arg_deltas)}: "
            f"{arg_deltas}"
        )
        concatenated = "".join(arg_deltas)
        parsed = json.loads(concatenated)
        assert parsed == {"city": "Tokyo", "unit": "celsius", "days": "5"}

    def test_streaming_long_string_arg_before_parameter_end(self, parser, mock_request):
        """Long string arguments should stream before the closing parameter tag."""
        chunks = [
            "<tool_call>\n",
            "<function=write_report>\n",
            "<parameter=content>",
            "Artificial intelligence has rapidly transformed the way ",
            "developers build dynamic applications with external tools.",
            "</param",
            "eter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        pre_close_arg_deltas: list[str] = []
        all_arg_deltas: list[str] = []
        for idx, (delta, _) in enumerate(results):
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        all_arg_deltas.append(tc.function.arguments)
                        if idx < 5:
                            pre_close_arg_deltas.append(tc.function.arguments)

        assert len(pre_close_arg_deltas) > 1, (
            "Expected long string arguments to stream incrementally before "
            f"</parameter>, got {pre_close_arg_deltas}"
        )
        partial_args = "".join(pre_close_arg_deltas)
        assert partial_args.startswith('{"content": "Artificial intelligence')
        assert partial_args.endswith("external tools.")
        assert not partial_args.endswith('"}')

        all_args = "".join(all_arg_deltas)
        assert json.loads(all_args) == {
            "content": (
                "Artificial intelligence has rapidly transformed the way "
                "developers build dynamic applications with external tools."
            )
        }

    def test_streaming_text_before_tool(self, parser, mock_request):
        chunks = [
            "Let me check ",
            "the weather. ",
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        assert collect_content(results).strip().startswith("Let me check")

    def test_streaming_empty_args(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=refresh>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "refresh"

    def test_streaming_split_parameter_tag(self, parser, mock_request):
        """Parameter tag split across chunks."""
        chunks = [
            "<tool_call>\n",
            "<function=test>\n",
            "<parameter=",
            "name>Alice",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "test"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["name"] == "Alice"

    def test_streaming_split_next_parameter_tag_is_buffered(self, parser, mock_request):
        """A split opening parameter tag must not leak into previous value."""
        chunks = [
            "<tool_call>\n",
            "<function=search>\n",
            "<parameter=query>hello ",
            "<param",
            "eter=limit>10</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        args_after_partial_tag = collect_tool_arguments(results[:4])
        assert "<param" not in args_after_partial_tag
        assert args_after_partial_tag == '{"query": "hello '

        args_text = collect_tool_arguments(results)
        assert json.loads(args_text) == {"query": "hello ", "limit": "10"}

    def test_streaming_numeric_values(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=set_config>\n",
            "<parameter=count>42</parameter>\n",
            "<parameter=active>true</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)
        args_text = collect_tool_arguments(results)
        if args_text:
            parsed = json.loads(args_text)
            assert parsed["count"] == "42"
            assert parsed["active"] == "true"

    def test_streaming_parallel_calls(self, parser, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n",
            "</function>\n",
            "</tool_call>",
            "<tool_call>\n",
            "<function=get_time>\n",
            "<parameter=tz>JST</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)

        assert "get_weather" in names
        assert "get_time" in names

    def test_streaming_value_split_across_chunks(self, parser, mock_request):
        """Parameter value split across multiple chunks."""
        chunks = [
            "<tool_call>\n",
            "<function=search>\n",
            "<parameter=query>hello ",
            "world",
            " test</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        results = simulate_tool_streaming(parser, mock_request, chunks)

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["query"] == "hello world test"

    def test_streaming_split_tool_call_tag(self, parser, mock_request):
        """<tool_call> arrives as a single special token; the rest of
        the content is split into fine-grained chunks."""
        chunks = [
            "<tool_call>\n",
            "<function=test>\n",
            "<parameter=x>1",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "test"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed["x"] == "1"

    def test_char_by_char_streaming(self, mock_request):
        """Feed text character-by-character to test lexer robustness.

        Uses a tokenizer without special token IDs because char-by-char
        delivery only occurs when the tokenizer splits the tag across
        multiple sub-word tokens (i.e., no dedicated special token).
        """
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.get_vocab.return_value = {}
        tokenizer.decode.side_effect = lambda ids: "".join(
            chr(i) if i < 128 else f"<{i}>" for i in ids
        )
        no_tid_parser = ParserEngine(
            tokenizer, parser_engine_config=qwen3_config(thinking=False)
        )

        full_text = (
            "<tool_call>\n"
            "<function=echo>\n"
            "<parameter=msg>hi</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        chunks = list(full_text)
        results = simulate_tool_streaming(no_tid_parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "echo"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert parsed == {"msg": "hi"}

    def test_streaming_multiline_param_values(self, parser, mock_request):
        """Multi-line parameter values in streaming mode."""
        chunks = [
            "<tool_call>\n",
            "<function=Bash>\n",
            "<parameter=command>\n",
            "ls -la /tmp\n",
            "</parameter>\n",
            "<parameter=description>\n",
            "List files\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        name = collect_function_name(results)
        assert name == "Bash"

        args_text = collect_tool_arguments(results)
        assert args_text
        parsed = json.loads(args_text)
        assert "ls -la /tmp" in parsed["command"]
        assert "List files" in parsed["description"]

    def test_streaming_multiline_two_tool_calls(self, parser, mock_request):
        """Two tool calls with multi-line values — matches bug report."""
        chunks = [
            "<tool_call>\n",
            "<function=Bash>\n",
            "<parameter=command>\n",
            "find /workspace -name '*.py' | head -20\n",
            "</parameter>\n",
            "<parameter=description>\n",
            "Find Python files\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
            "<tool_call>\n",
            "<function=Read>\n",
            "<parameter=file_path>\n",
            "/workspace/main.py\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser, mock_request, chunks)

        names = []
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        names.append(tc.function.name)

        assert "Bash" in names
        assert "Read" in names


class TestArgConverter:
    """Direct tests for the Qwen3 arg_converter with multi-line values."""

    def test_multiline_param_values(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = (
            "<parameter=command>\n"
            "ls -la /tmp\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "List files\n"
            "</parameter>\n"
        )
        result = json.loads(_qwen3_arg_converter(raw, partial=False))
        assert result["command"] == "ls -la /tmp"
        assert result["description"] == "List files"

    def test_two_multiline_params(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = (
            "<parameter=a>\nfoo\nbar\n</parameter>\n"
            "<parameter=b>\nbaz\nqux\n</parameter>\n"
        )
        result = json.loads(_qwen3_arg_converter(raw, partial=False))
        assert result["a"] == "foo\nbar"
        assert result["b"] == "baz\nqux"

    def test_partial_multiline(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = "<parameter=command>\nls -la</parameter>\n<parameter=desc>\npartial value"
        result = json.loads(_qwen3_arg_converter(raw, partial=True))
        assert result["command"] == "ls -la"
        assert result["desc"] == "partial value"

    def test_partial_value_with_angle_bracket(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = "<parameter=expr>x<5"
        result = json.loads(_qwen3_arg_converter(raw, partial=True))
        assert result == {"expr": "x<5"}

    def test_partial_value_with_angle_bracket_and_complete_param(self):
        from vllm.parser.qwen3 import (
            _qwen3_arg_converter,
        )

        raw = "<parameter=city>Tokyo</parameter>\n<parameter=expr>x<5"
        result = json.loads(_qwen3_arg_converter(raw, partial=True))
        assert result == {"city": "Tokyo", "expr": "x<5"}


WRITE_CALL_HEAD = (
    "<tool_call>\n<function=write_file>\n"
    "<parameter=path>\n/repo/app.py\n</parameter>\n"
    "<parameter=mode>\n644\n</parameter>\n"
    "<parameter=content>\n"
)
WRITE_CALL_TAIL = "\n</parameter>\n</function>\n</tool_call>"
# Escapes, non-ASCII, blank lines and tag-like text that must stay inside
# the value, including prefixes of the closing and next-parameter tags.
CONTENT_LINE = 'if a < b > c: print("x\\y")  # </p> </param <parameter\té\n\n'


class TestArgValueExtender:
    """The open parameter value streams without re-running the converter."""

    @pytest.fixture
    def write_tool(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return ChatCompletionToolsParam(
            type="function",
            function={
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "mode": {"type": "integer"},
                        "content": {"type": "string"},
                    },
                },
            },
        )

    @pytest.mark.parametrize("use_schema", [False, True], ids=["no_schema", "schema"])
    @pytest.mark.parametrize("parser_name", ["qwen3", "mimo"])
    def test_long_call_streams_non_streaming_arguments(
        self, mock_tokenizer, mock_request, write_tool, use_schema, parser_name
    ):
        from vllm.parser.mimo import MiMoParser
        from vllm.parser.qwen3 import Qwen3Parser

        cls = MiMoParser if parser_name == "mimo" else Qwen3Parser
        mock_request.tools = [write_tool] if use_schema else []
        text = WRITE_CALL_HEAD + CONTENT_LINE * 200 + WRITE_CALL_TAIL

        def make_parser():
            return cls(mock_tokenizer, chat_template_kwargs={"enable_thinking": False})

        results = simulate_tool_streaming(
            make_parser(),
            mock_request,
            [text[i : i + 7] for i in range(0, len(text), 7)],
        )
        result = make_parser().extract_tool_calls(text, mock_request)

        assert collect_tool_arguments(results) == (
            result.tool_calls[0].function.arguments
        )
        # MiMo keeps the newlines around the value; the schema coerces both.
        mode = '"\\n644\\n"' if parser_name == "mimo" else '"644"'
        if use_schema:
            mode = "644"
        assert f'"mode": {mode},' in collect_tool_arguments(results)

    def test_conversions_do_not_grow_with_argument_length(
        self, mock_tokenizer, mock_request, write_tool
    ):
        mock_request.tools = [write_tool]
        base = qwen3_config(thinking=False)
        converter = base.arg_converter
        assert converter is not None

        def streaming_conversions(lines: int) -> int:
            calls = 0

            def counting_converter(raw_args: str, partial: bool) -> str:
                nonlocal calls
                calls += partial
                return converter(raw_args, partial)

            parser = ParserEngine(
                mock_tokenizer,
                parser_engine_config=dataclasses.replace(
                    base, arg_converter=counting_converter
                ),
            )
            text = WRITE_CALL_HEAD + CONTENT_LINE * lines + WRITE_CALL_TAIL
            simulate_tool_streaming(
                parser,
                mock_request,
                [text[i : i + 4] for i in range(0, len(text), 4)],
            )
            return calls

        assert streaming_conversions(lines=20) == streaming_conversions(lines=200)

    def test_extender_follows_newline_trimming_and_split_end(self):
        from vllm.parser.qwen3 import _qwen3_arg_value_extender

        extend = _qwen3_arg_value_extender("<parameter=k>")
        assert extend is not None
        deltas = ["\n", 'a"\n', "b", "\n", "</param", "eter>"]
        expected = ["", 'a\\"', "\\nb", "", "\\n</param", None]
        assert [extend(d) for d in deltas] == expected

    def test_extender_stops_at_split_next_parameter(self):
        from vllm.parser.qwen3 import _qwen3_arg_value_extender

        extend = _qwen3_arg_value_extender("<parameter=k>x")
        assert extend is not None
        assert [extend(d) for d in ["< para", "meter ="]] == ["< para", None]

    @pytest.mark.parametrize(
        "raw_args",
        [
            "<parameter=k>x</parameter>",
            "<parameter=k",
            "<parameter=k>1</parameter><parameter=j>2</parameter><parameter=k>3",
        ],
        ids=["closed_value", "open_name", "repeated_key_not_last"],
    )
    def test_no_extender_unless_open_value_is_last(self, raw_args):
        from vllm.parser.qwen3 import _qwen3_arg_value_extender

        assert _qwen3_arg_value_extender(raw_args) is None


class TestSchemaAwareTypeCoercion:
    """Verify that _fix_arg_types corrects miscoerced values using the
    tool schema."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "TaskUpdate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "taskId": {"type": "string"},
                            "count": {"type": "integer"},
                            "ratio": {"type": "number"},
                            "flag": {"type": "string"},
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_tools(self, mock_tokenizer, tools):
        return ParserEngine(
            mock_tokenizer,
            tools=tools,
            parser_engine_config=qwen3_config(thinking=False),
        )

    def test_string_param_not_coerced_to_int(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=taskId>1</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["taskId"] == "1"
        assert isinstance(args["taskId"], str)

    def test_string_param_not_coerced_to_bool(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=flag>true</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["flag"] == "true"
        assert isinstance(args["flag"], str)

    def test_int_param_still_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=count>42</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["count"] == 42
        assert isinstance(args["count"], int)

    def test_no_tools_keeps_strings(self, parser, mock_request):
        text = (
            "<tool_call>\n"
            "<function=TaskUpdate>\n"
            "<parameter=taskId>1</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["taskId"] == "1"
        assert isinstance(args["taskId"], str)

    def test_streaming_string_param_not_coerced(self, parser_with_tools, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=TaskUpdate>\n",
            "<parameter=taskId>1</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_str = collect_tool_arguments(results)
        args = json.loads(args_str)
        assert args["taskId"] == "1"
        assert isinstance(args["taskId"], str)


class TestAnyOfTypeCoercion:
    """Verify that _fix_arg_types handles union types (anyOf/oneOf)."""

    @pytest.fixture
    def tools_with_anyof(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "set_config",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "port": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"},
                                ],
                            },
                            "count": {"type": "integer"},
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_anyof(self, mock_tokenizer, tools_with_anyof):
        return ParserEngine(
            mock_tokenizer,
            tools=tools_with_anyof,
            parser_engine_config=qwen3_config(thinking=False),
        )

    def test_anyof_string_param_not_coerced(self, parser_with_anyof, mock_request):
        """A param with anyOf including 'string' must not be coerced
        to integer."""
        text = (
            "<tool_call>\n"
            "<function=set_config>\n"
            "<parameter=port>8080</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_anyof.extract_tool_calls(text, mock_request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["port"] == "8080"


class TestSchemaCoercionBoolNumberNull:
    """Verify that _fix_arg_types coerces string values to non-string
    schema types using coerce_to_schema_type."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "configure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "ratio": {"type": "number"},
                            "count": {"type": "integer"},
                            "value": {"type": ["integer", "null"]},
                            "label": {"type": "string"},
                        },
                    },
                },
            )
        ]

    @pytest.fixture
    def parser_with_tools(self, mock_tokenizer, tools):
        return ParserEngine(
            mock_tokenizer,
            tools=tools,
            parser_engine_config=qwen3_config(thinking=False),
        )

    def test_bool_param_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=enabled>true</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["enabled"] is True
        assert isinstance(args["enabled"], bool)

    def test_number_param_whole_normalized(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=ratio>5.0</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["ratio"] == 5
        assert isinstance(args["ratio"], int)

    def test_number_param_fractional(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=ratio>3.14</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["ratio"] == pytest.approx(3.14)
        assert isinstance(args["ratio"], float)

    def test_null_coerced_when_in_schema(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=value>null</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["value"] is None

    def test_null_stays_string_without_null_schema(
        self, parser_with_tools, mock_request
    ):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=label>null</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["label"] == "null"
        assert isinstance(args["label"], str)

    def test_streaming_bool_param_coerced(self, parser_with_tools, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=configure>\n",
            "<parameter=enabled>true</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_str = collect_tool_arguments(results)
        args = json.loads(args_str)
        assert args["enabled"] is True
        assert isinstance(args["enabled"], bool)

    def test_streaming_number_param_coerced(self, parser_with_tools, mock_request):
        chunks = [
            "<tool_call>\n",
            "<function=configure>\n",
            "<parameter=ratio>3.14</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_str = collect_tool_arguments(results)
        args = json.loads(args_str)
        assert args["ratio"] == pytest.approx(3.14)
        assert isinstance(args["ratio"], float)

    def test_streaming_matches_non_streaming_comprehensive(
        self, parser_with_tools, mock_request
    ):
        text = (
            "<tool_call>\n"
            "<function=configure>\n"
            "<parameter=enabled>true</parameter>\n"
            "<parameter=ratio>5.0</parameter>\n"
            "<parameter=count>42</parameter>\n"
            "<parameter=value>null</parameter>\n"
            "<parameter=label>hello</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        non_stream = parser_with_tools.extract_tool_calls(text, mock_request)
        ns_args = json.loads(non_stream.tool_calls[0].function.arguments)

        chunks = [line + "\n" for line in text.split("\n") if line]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        s_args = json.loads(collect_tool_arguments(results))

        assert s_args == ns_args
        assert ns_args == {
            "enabled": True,
            "ratio": 5,
            "count": 42,
            "value": None,
            "label": "hello",
        }


class TestNestedSchemaCoercion:
    """Verify that _fix_arg_types recurses into nested objects and arrays."""

    @pytest.fixture
    def tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
        )

        return [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "filters": {
                                "type": "object",
                                "properties": {
                                    "language": {"type": "string"},
                                    "min_stars": {"type": "integer"},
                                },
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "limits": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "verbose": {"type": "boolean"},
                        },
                    },
                },
            ),
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "AskUserQuestion",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "multiSelect": {
                                            "type": "boolean",
                                        },
                                        "answer": {
                                            "type": ["string", "null"],
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            ),
        ]

    @pytest.fixture
    def parser_with_tools(self, mock_tokenizer, tools):
        return ParserEngine(
            mock_tokenizer,
            tools=tools,
            parser_engine_config=qwen3_config(thinking=False),
        )

    def test_nested_object_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            '<parameter=filters>{"language": "python",'
            ' "min_stars": 100}</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["filters"] == {"language": "python", "min_stars": 100}
        assert isinstance(args["filters"]["min_stars"], int)

    def test_nested_array_items_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=limits>[10, 20, 30]</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["limits"] == [10, 20, 30]
        assert all(isinstance(v, int) for v in args["limits"])

    def test_nested_string_array_not_coerced(self, parser_with_tools, mock_request):
        text = (
            "<tool_call>\n"
            "<function=search>\n"
            '<parameter=tags>["ml", "42"]</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["tags"] == ["ml", "42"]
        assert all(isinstance(v, str) for v in args["tags"])

    def test_array_of_objects_with_bool_and_null_coerced(
        self, parser_with_tools, mock_request
    ):
        text = (
            "<tool_call>\n"
            "<function=AskUserQuestion>\n"
            "<parameter=questions>"
            '[{"question": "Pick a color",'
            ' "multiSelect": false, "answer": null}]'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parser_with_tools.extract_tool_calls(text, mock_request)
        args = json.loads(result.tool_calls[0].function.arguments)
        questions = args["questions"]
        assert isinstance(questions, list)
        assert len(questions) == 1
        assert questions[0]["question"] == "Pick a color"
        assert questions[0]["multiSelect"] is False
        assert questions[0]["answer"] is None

    def test_streaming_array_of_objects_with_bool_and_null_coerced(
        self, parser_with_tools, mock_request
    ):
        chunks = [
            "<tool_call>\n",
            "<function=AskUserQuestion>\n",
            '<parameter=questions>[{"question": "Pick a color",',
            ' "multiSelect": false, "answer": null}]',
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        results = simulate_tool_streaming(parser_with_tools, mock_request, chunks)
        args_str = collect_tool_arguments(results)
        args = json.loads(args_str)
        questions = args["questions"]
        assert isinstance(questions, list)
        assert len(questions) == 1
        assert questions[0]["question"] == "Pick a color"
        assert questions[0]["multiSelect"] is False
        assert questions[0]["answer"] is None


@pytest.mark.parametrize("value", ["\n北京 &amp;\n", "null", "42", "", "a\nb"])
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_mimo_preserves_verbatim_parameter_values(
    mock_tokenizer, mock_request, value, chunk_size
):
    from vllm.parser.mimo import MiMoParser

    text = (
        f"<tool_call><function=run><parameter=text>{value}</parameter>"
        "</function></tool_call>"
    )
    parser = MiMoParser(mock_tokenizer, chat_template_kwargs={"enable_thinking": False})
    result = parser.extract_tool_calls(text, mock_request)
    assert json.loads(result.tool_calls[0].function.arguments) == {"text": value}
    parser = MiMoParser(mock_tokenizer, chat_template_kwargs={"enable_thinking": False})
    results = simulate_tool_streaming(
        parser,
        mock_request,
        [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)],
    )
    assert collect_function_name(results) == "run"
    assert json.loads(collect_tool_arguments(results)) == {"text": value}


@pytest.mark.parametrize(
    "profile,tools_enabled,closed",
    [
        ("mimo", False, False),
        ("mimo", False, True),
        ("mimo", True, False),
        ("mimo", True, True),
        pytest.param(
            "glm",
            True,
            True,
        ),
        pytest.param(
            "glm",
            True,
            False,
        ),
        pytest.param(
            "glm",
            False,
            False,
        ),
        pytest.param(
            "glm",
            False,
            True,
        ),
    ],
)
def test_selected_real_parser_reasoning_reentry(profile, tools_enabled, closed):
    from tests.parser.engine.replay_harness import (
        load_local_model_parser,
        replay_real_tokens,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.parser import ParserManager
    from vllm.parser.abstract_parser import DelegatingParser

    tokenizer, selected_cls = load_local_model_parser(profile)
    assert issubclass(selected_cls, DelegatingParser)
    if profile == "glm":
        from vllm.parser.glm47_moe import Glm47MoeServingParser

        assert issubclass(selected_cls, Glm47MoeServingParser)
    reason, tool = ("mimo", "mimo") if profile == "mimo" else ("glm45", "glm47")
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name=reason,
        tool_parser_name=tool,
        enable_auto_tools=tools_enabled,
    )
    request = ResponsesRequest(model="test", input="hi", include_reasoning=True)
    text = "<think>first</think>visible<think>second"
    if closed:
        text += "</think>final"
    ids = tokenizer.encode(text, add_special_tokens=False)
    expected_content = "visiblefinal" if closed else "visible"
    full_parser = parser_cls(tokenizer)
    full = full_parser.parse(text, request, tools_enabled, ids)
    full_reasoning_tokens = full_parser.count_reasoning_tokens(ids)
    assert full_reasoning_tokens > 0
    assert full[:2] == ("firstsecond", expected_content)
    for split in range(len(ids) + 1):
        parser = parser_cls(tokenizer)
        deltas = replay_real_tokens(parser, tokenizer, ids, request, split)
        assert "".join(d.reasoning or "" for d in deltas) == "firstsecond", split
        assert "".join(d.content or "" for d in deltas) == expected_content, split
        assert parser.count_reasoning_tokens(ids) == full_reasoning_tokens, split
    # The same actual adapter used by structured output recognizes the latest
    # genuine marker, rather than latching the first reasoning end in history.
    adapter = parser_cls(tokenizer).reasoning_parser
    assert adapter is not None
    opener = tokenizer.encode("<think>", add_special_tokens=False)
    ended = tokenizer.encode("<think>x</think>visible", add_special_tokens=False)
    assert adapter.is_reasoning_end(ended)
    assert not adapter.is_reasoning_end(ended + opener)
    assert not adapter.is_reasoning_end_streaming(ended + opener, opener)
    closer = tokenizer.encode("</think>", add_special_tokens=False)
    assert adapter.is_reasoning_end_streaming(ended + opener + closer, closer)


@pytest.mark.parametrize(
    "profile",
    [
        "mimo",
        pytest.param(
            "glm",
        ),
    ],
)
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
def test_selected_real_parser_tools_reentry_and_literal_bytes(profile, choice):
    from tests.parser.engine.replay_harness import (
        load_local_model_parser,
        replay_real_tokens,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, parser_cls = load_local_model_parser(profile)
    tools = [
        {
            "type": "function",
            "name": "echo",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        }
    ]
    request = ResponsesRequest(
        model="test",
        input="hi",
        tools=tools,
        tool_choice={"type": "function", "name": "echo"}
        if choice == "named"
        else choice,
        include_reasoning=True,
    )
    value = '\n  "<think>quoted</think>"\n漢字\n'
    if profile == "mimo":
        call = (
            "<tool_call><function=echo><parameter=text>"
            + value
            + "</parameter></function></tool_call>"
        )
    else:
        call = (
            "<tool_call>echo<arg_key>text</arg_key><arg_value>"
            + value
            + "</arg_value></tool_call>"
        )
    text = "<think>first</think>visible<think>second</think>" + call + call
    ids = tokenizer.encode(text, add_special_tokens=False)
    parser = parser_cls(tokenizer, request.tools)
    if (
        profile == "mimo"
        and choice != "auto"
        and parser.tool_parser.supports_required_and_named
    ):
        pytest.skip(
            "XML fixture requires strict=1; strict=0 required/named JSON has "
            "separate exact-byte and chunk-invariance fixtures"
        )
    # Keep adapter-based request/strict-tag selection in the selected path.
    assert parser.reasoning_parser is not None and parser.tool_parser is not None
    parser.adjust_request(request)
    assert request.skip_special_tokens is False
    full = parser.parse(text, request, True, ids)
    assert full[:2] == ("firstsecond", "visible")
    assert len(full[2]) == 2
    assert [json.loads(call.arguments) for call in full[2]] == [{"text": value}] * 2
    for split in range(len(ids) + 1):
        parser = parser_cls(tokenizer, request.tools)
        deltas = replay_real_tokens(parser, tokenizer, ids, request, split)
        assert "".join(d.reasoning or "" for d in deltas) == "firstsecond", split
        assert "".join(d.content or "" for d in deltas) == "visible"
        calls: dict[int, str] = {}
        for delta in deltas:
            for call in delta.tool_calls or []:
                if call.function:
                    calls[call.index] = calls.get(call.index, "") + (
                        call.function.arguments or ""
                    )
        assert [json.loads(calls[i]) for i in sorted(calls)] == [{"text": value}] * 2


def test_mimo_literal_think_text_is_not_a_control_token():
    from tests.parser.engine.replay_harness import (
        load_local_model_parser,
        replay_real_tokens,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, cls = load_local_model_parser("mimo")
    request = ResponsesRequest(model="test", input="hi", include_reasoning=True)
    prefix = tokenizer.encode("<think>first</think>", add_special_tokens=False)
    literal = 'Use "<think>quoted</think>" exactly.'
    # Encode each character through the real tokenizer so the quoted spelling
    # contains ordinary text tokens, not the model control-token IDs.
    ordinary = [
        tid
        for char in literal
        for tid in tokenizer.encode(char, add_special_tokens=False)
    ]
    assert tokenizer.get_vocab()["<think>"] not in ordinary
    ids = prefix + ordinary
    full = cls(tokenizer).parse(tokenizer.decode(ids), request, True, ids)
    assert full[:2] == ("first", literal)
    for split in range(len(ids) + 1):
        deltas = replay_real_tokens(cls(tokenizer), tokenizer, ids, request, split)
        assert "".join(d.content or "" for d in deltas) == literal
        assert "".join(d.reasoning or "" for d in deltas) == "first"


@pytest.mark.parametrize("thinking", [False, True])
def test_mimo_selected_parser_respects_prompt_reasoning_state(thinking):
    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, cls = load_local_model_parser("mimo")
    request = ResponsesRequest(model="test", input="hi", include_reasoning=True)
    parser = cls(tokenizer, chat_template_kwargs={"enable_thinking": thinking})
    prompt = tokenizer.encode(
        "<|im_start|>assistant\n<think>already done</think>",
        add_special_tokens=False,
    )
    ids = tokenizer.encode("visible", add_special_tokens=False)
    delta = parser.parse_delta("visible", ids, request, prompt, finished=True)
    assert delta.content == "visible"
    assert delta.reasoning is None


def test_glm_deferred_literal_control_token_in_tool_argument():
    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, cls = load_local_model_parser("glm")
    value = "<think>quoted</think>"
    text = (
        "<think>reason</think><tool_call>echo<arg_key>text</arg_key>"
        "<arg_value>" + value + "</arg_value></tool_call>"
    )
    request = ResponsesRequest(
        model="test",
        input="hi",
        include_reasoning=True,
        tools=[
            {
                "type": "function",
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }
        ],
    )
    result = cls(tokenizer).parse(
        text, request, True, tokenizer.encode(text, add_special_tokens=False)
    )
    assert result[0] == "reason"
    assert json.loads(result[2][0].arguments) == {"text": value}


@pytest.mark.parametrize("profile", ["mimo", "glm"])
@pytest.mark.parametrize("choice", ["required", "named"])
def test_selected_legacy_json_tool_contract(choice, profile):
    from tests.parser.engine.replay_harness import (
        load_local_model_parser,
        replay_real_tokens,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, cls = load_local_model_parser(profile)
    parser = cls(tokenizer)
    # This is the existing legacy capability flag selected when structural
    # tags are disabled, not a replacement for parsing or tokenization.
    import os

    if os.environ.get("VLLM_ENFORCE_STRICT_TOOL_CALLING") != "0":
        pytest.skip("legacy JSON requires a fresh strict=0 process")
    request = ResponsesRequest(
        model="test",
        input="hi",
        include_reasoning=True,
        tools=[
            {
                "type": "function",
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }
        ],
        tool_choice={"type": "function", "name": "echo"}
        if choice == "named"
        else choice,
    )
    request = parser.adjust_request(request)
    assert request.text.format.type == "json_schema"
    arguments = {"text": '  <think>quoted</think>\n"value"'}
    body = (
        json.dumps(arguments)
        if choice == "named"
        else json.dumps([{"name": "echo", "parameters": arguments}])
    )
    text = "<think>first</think><think>second</think>" + body
    ids = tokenizer.encode(text, add_special_tokens=False)
    result = parser.parse(text, request, True, ids)
    assert result[0] == "firstsecond"
    assert result[2][0].name == "echo"
    assert json.loads(result[2][0].arguments) == arguments
    full_reasoning_tokens = parser.count_reasoning_tokens(ids)
    assert full_reasoning_tokens > 0
    for split in range(len(ids) + 1):
        stream_parser = cls(tokenizer)
        deltas = replay_real_tokens(stream_parser, tokenizer, ids, request, split)
        assert "".join(delta.reasoning or "" for delta in deltas) == "firstsecond"
        streamed = "".join(
            call.function.arguments or ""
            for delta in deltas
            for call in delta.tool_calls or []
            if call.function
        )
        assert json.loads(streamed) == arguments
        assert stream_parser.count_reasoning_tokens(ids) == full_reasoning_tokens, split


def test_mimo_selected_hides_reasoning_without_changing_content():
    from tests.entrypoints.openai.responses.test_message_phases import serving
    from tests.parser.engine.replay_harness import (
        load_local_model_parser,
        replay_real_tokens,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.outputs import CompletionOutput

    tokenizer, cls = load_local_model_parser("mimo")
    request = ResponsesRequest(model="test", input="hi", include_reasoning=False)
    text = "<think>first</think>visible<think>second</think>final"
    ids = tokenizer.encode(text, add_special_tokens=False)
    for split in range(len(ids) + 1):
        deltas = replay_real_tokens(cls(tokenizer), tokenizer, ids, request, split)
        assert "".join(delta.content or "" for delta in deltas) == "visiblefinal"
        assert not any(delta.reasoning for delta in deltas)
    items = serving()._make_response_output_items(
        request,
        CompletionOutput(
            index=0,
            text=text,
            token_ids=ids,
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
        ),
        tokenizer,
        parser=cls(tokenizer),
    )
    assert [item.type for item in items] == ["message"]
    assert items[0].content[0].text == "visiblefinal"


@pytest.mark.parametrize("implicit", [False, True])
def test_mimo_selected_tool_transition_and_duplicate_reasoning_end(implicit):
    from tests.parser.engine.replay_harness import (
        load_local_model_parser,
        replay_real_tokens,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, cls = load_local_model_parser("mimo")
    request = ResponsesRequest(
        model="test",
        input="hi",
        include_reasoning=True,
        tools=[
            {
                "type": "function",
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }
        ],
    )
    prefix = "<think>first"
    if not implicit:
        prefix += "</think></think><think>second</think>"
    text = prefix + (
        "<tool_call><function=echo><parameter=text>  value\n"
        "</parameter></function></tool_call>"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    expected_reasoning = "first" if implicit else "firstsecond"
    full = cls(tokenizer, request.tools).parse(text, request, True, ids)
    assert full[0] == expected_reasoning
    assert json.loads(full[2][0].arguments) == {"text": "  value\n"}
    for split in range(len(ids) + 1):
        deltas = replay_real_tokens(
            cls(tokenizer, request.tools), tokenizer, ids, request, split
        )
        assert (
            "".join(delta.reasoning or "" for delta in deltas) == expected_reasoning
        ), split
        arguments = "".join(
            call.function.arguments or ""
            for delta in deltas
            for call in delta.tool_calls or []
            if call.function
        )
        assert json.loads(arguments) == {"text": "  value\n"}, split


def test_mimo_selected_tool_choice_none_does_not_emit_calls():
    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, cls = load_local_model_parser("mimo")
    request = ResponsesRequest(
        model="test",
        input="hi",
        tool_choice="none",
        tools=[
            {
                "type": "function",
                "name": "echo",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    text = "<think>reason</think><tool_call><function=echo></function></tool_call>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    full = cls(tokenizer, request.tools).parse(text, request, True, ids)
    assert not full[2]
    delta = cls(tokenizer, request.tools).parse_delta(text, ids, request, finished=True)
    assert delta is None or not delta.tool_calls


@pytest.mark.parametrize("profile", ["mimo", "glm"])
@pytest.mark.parametrize("replay", ["whole", "two_chunks", "single_token"])
def test_selected_required_json_parallel_chunk_invariance(replay, profile):
    import os

    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    if os.environ.get("VLLM_ENFORCE_STRICT_TOOL_CALLING") != "0":
        pytest.skip("fresh strict=0 process exercises actual legacy capability")
    tokenizer, cls = load_local_model_parser(profile)
    first = {
        "text": '中文 <think>literal</think>\n"quote"\\tail',
        "nested": [None, True, False, 1.25, {"items": [1, "x"]}],
    }
    second = {"number": -2.5e8, "empty": {}, "list": ["}", "]", "\\", "\n"]}
    arguments = [
        json.dumps(first, ensure_ascii=False),
        (
            '{"number":-2.50e+8,"empty":{},"list":["}","]","\\\\","\\n"],'
            '"escaped":"\\u4e2d\\ud83d\\ude00"}'
        ),
    ]
    second["escaped"] = "中😀"
    # Escaped property/name spellings and reversed field order must not
    # confuse envelope/argument boundaries.
    body = (
        '[{"na\\u006de":"first","parameters":'
        + arguments[0]
        + '},{"parameters":'
        + arguments[1]
        + ',"name":"second"}]'
    )
    prefix = "<think>first</think><think>second</think>"
    text = prefix + body
    ids = tokenizer.encode(text, add_special_tokens=False)
    request = ResponsesRequest(
        model="test",
        input="hi",
        include_reasoning=True,
        tool_choice="required",
        tools=[
            {"type": "function", "name": name, "parameters": {"type": "object"}}
            for name in ("first", "second")
        ],
    )
    full_parser = cls(tokenizer, request.tools)
    request = full_parser.adjust_request(request)
    assert request.text.format.type == "json_schema"
    assert full_parser.tool_parser.supports_required_and_named is True
    full = full_parser.parse(text, request, True, ids)
    assert full[0] == "firstsecond"
    assert [call.name for call in full[2]] == ["first", "second"]
    assert [json.loads(call.arguments) for call in full[2]] == [first, second]
    expected_count = full_parser.count_reasoning_tokens(ids)
    schedules = (
        [[len(ids)]]
        if replay == "whole"
        else [[split, len(ids)] for split in range(len(ids) + 1)]
        if replay == "two_chunks"
        else [list(range(1, len(ids) + 1))]
    )
    for ends in schedules:
        parser = cls(tokenizer, request.tools)
        deltas = []
        previous = ""
        start = 0
        first_call_emitted_at = None
        for end in ends:
            decoded = tokenizer.decode(ids[:end]).rstrip("\ufffd")
            assert decoded.startswith(previous)
            delta = parser.parse_delta(
                decoded[len(previous) :], ids[start:end], request, finished=False
            )
            if delta is not None:
                deltas.append(delta)
                if first_call_emitted_at is None and any(
                    c.id and c.index == 0 for c in delta.tool_calls or []
                ):
                    first_call_emitted_at = len(decoded)
            previous, start = decoded, end
        terminal = parser.parse_delta("", [], request, finished=True)
        if terminal is not None:
            deltas.append(terminal)
        assert "".join(d.reasoning or "" for d in deltas) == "firstsecond", ends
        assert not any(d.content for d in deltas), ends
        by_index: dict[int, dict[str, Any]] = {}
        for delta in deltas:
            for call in delta.tool_calls or []:
                info = by_index.setdefault(
                    call.index, {"ids": [], "name": "", "args": ""}
                )
                if call.id:
                    info["ids"].append(call.id)
                if call.function:
                    info["name"] += call.function.name or ""
                    info["args"] += call.function.arguments or ""
        assert sorted(by_index) == [0, 1], (ends, by_index)
        assert [by_index[i]["name"] for i in (0, 1)] == ["first", "second"], ends
        assert [by_index[i]["args"] for i in (0, 1)] == arguments, ends
        assert all(len(by_index[i]["ids"]) == 1 for i in (0, 1)), ends
        assert by_index[0]["ids"][0] != by_index[1]["ids"][0], ends
        assert parser.count_reasoning_tokens(ids) == expected_count, ends
        if replay == "single_token":
            assert first_call_emitted_at is not None
            assert first_call_emitted_at < text.index('},{"parameters"'), ends


@pytest.mark.parametrize("tail", ['{"x":-2.50e+', '{"x":"\\u4e', '{"x":"value\\'])
def test_mimo_required_json_final_flush_preserves_unfinished_argument(tail):
    import os

    from tests.parser.engine.replay_harness import (
        load_local_model_parser,
        replay_real_tokens,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    if os.environ.get("VLLM_ENFORCE_STRICT_TOOL_CALLING") != "0":
        pytest.skip("fresh strict=0 process exercises actual legacy capability")
    tokenizer, cls = load_local_model_parser("mimo")
    request = ResponsesRequest(
        model="test",
        input="hi",
        include_reasoning=True,
        tool_choice="required",
        tools=[{"type": "function", "name": "first", "parameters": {"type": "object"}}],
    )
    text = (
        '<think>first</think><think>second</think>[{"name":"first","parameters":' + tail
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    for split in range(len(ids) + 1):
        parser = cls(tokenizer, request.tools)
        assert parser.tool_parser.supports_required_and_named is True
        deltas = replay_real_tokens(parser, tokenizer, ids, request, split)
        assert "".join(delta.reasoning or "" for delta in deltas) == "firstsecond"
        calls = [c for delta in deltas for c in delta.tool_calls or []]
        assert {call.index for call in calls} == {0}, split
        assert len([call.id for call in calls if call.id]) == 1, split
        assert "".join(c.function.name or "" for c in calls if c.function) == "first"
        assert "".join(c.function.arguments or "" for c in calls if c.function) == tail
        assert not any(delta.content for delta in deltas), split


@pytest.mark.parametrize("name_first", [False, True])
def test_mimo_required_json_stream_releases_emitted_prefixes(name_first):
    from vllm.parser.mimo import _RequiredJSONStream

    raw = json.dumps(
        {"body": '中文 <think>literal</think> "quotes" \\ line\n' * 2048},
        ensure_ascii=False,
    )
    body = (
        '[{"name":"echo","parameters":' + raw + "}]"
        if name_first
        else '[{"parameters":' + raw + ',"name":"echo"}]'
    )
    parser = _RequiredJSONStream()
    emitted = []
    for char in body:
        emitted.extend(parser.feed(char))
        if parser.name is not None:
            assert not parser.pending
        assert not hasattr(parser, "previous_text")
    assert "".join(args for _, _, args, _ in emitted) == raw
    assert all(index == 0 and name == "echo" for index, name, _, _ in emitted)
    assert sum(first for _, _, _, first in emitted) == 1
    assert parser.pending == parser.capture == parser.stack == []
    assert parser.feed("") == []


@pytest.mark.parametrize(
    "raw", ["null", "true", "false", "-2.50e+8", '"literal \u4e2d"', "[]", "{}"]
)
def test_mimo_required_json_incremental_scalar_arguments(raw):
    from vllm.parser.mimo import _RequiredJSONStream

    body = '[{"name":"echo","parameters":' + raw + "}]"
    for split in range(len(body) + 1):
        parser = _RequiredJSONStream()
        updates = parser.feed(body[:split]) + parser.feed(body[split:])
        assert "".join(args for _, _, args, _ in updates) == raw
        assert sum(first for _, _, _, first in updates) == 1


@pytest.mark.parametrize("choice", ["required", "named"])
def test_mimo_legacy_json_without_thinking_preserves_argument_markers(choice):
    import os

    from tests.parser.engine.replay_harness import (
        load_local_model_parser,
        replay_real_tokens,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    if os.environ.get("VLLM_ENFORCE_STRICT_TOOL_CALLING") != "0":
        pytest.skip("legacy JSON requires a fresh strict=0 process")
    tokenizer, cls = load_local_model_parser("mimo")
    request = ResponsesRequest(
        model="test",
        input="hi",
        tools=[{"type": "function", "name": "echo", "parameters": {"type": "object"}}],
        tool_choice={"type": "function", "name": "echo"}
        if choice == "named"
        else choice,
    )
    args = {"text": "<think>quoted</think> 中文"}
    text = json.dumps(
        args if choice == "named" else [{"name": "echo", "parameters": args}],
        ensure_ascii=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)

    def parser():
        return cls(
            tokenizer, request.tools, chat_template_kwargs={"enable_thinking": False}
        )

    full = parser().parse(text, request, True, ids)
    assert not full[0] and not full[1]
    assert json.loads(full[2][0].arguments) == args
    p = parser()
    deltas = replay_real_tokens(p, tokenizer, ids, request, len(ids) // 2)
    assert not any(d.reasoning or d.content for d in deltas)
    assert (
        json.loads(
            "".join(
                c.function.arguments or ""
                for d in deltas
                for c in d.tool_calls or []
                if c.function
            )
        )
        == args
    )
    assert p.count_reasoning_tokens(ids) == 0


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize(
    "mode", ["implicit_start", "tool_span", "reentry", "prompt_end"]
)
def test_selected_reasoning_count_uses_tokens_and_model_states(profile, mode):
    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    tokenizer, parser_cls = load_local_model_parser(profile)
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    request = ResponsesRequest(
        model="test",
        input="hi",
        include_reasoning=True,
        tools=[
            {
                "type": "function",
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }
        ],
        tool_choice="auto",
    )
    reasoning = "reason𐐀"
    reasoning_ids = enc(reasoning)
    prefix = [] if mode == "implicit_start" else enc("<think>")
    ids = prefix + reasoning_ids
    expected_positions = set(range(len(prefix), len(ids)))
    prompt = []
    if mode == "prompt_end":
        ids = enc("answer𐐀")
        expected_positions.clear()
        prompt = enc("<think></think>")
    elif mode == "tool_span":
        # Tool start implicitly ends reasoning; special IDs inside arguments
        # must not start another reasoning interval.
        tool = (
            "<tool_call>echo<arg_key>text</arg_key><arg_value>"
            "<think>argument𐐀</think></arg_value></tool_call>"
            if profile == "glm"
            else "<tool_call><function=echo><parameter=text>"
            "<think>argument𐐀</think></parameter></function></tool_call>"
        )
        ids += enc(tool) + enc("answer")
    else:
        ids += enc("</think>answer")
        if mode == "reentry":
            ids += enc("<think>")
            expected_positions.update(range(len(ids), len(ids) + len(reasoning_ids)))
            ids += reasoning_ids + enc("</think>answer")

    for step in (1, len(ids)):
        parser = parser_cls(tokenizer, request.tools)
        # Hold all text back: accounting must still follow generated tokens.
        for start in range(0, len(ids), step):
            end = min(start + step, len(ids))
            parser.parse_deltas(
                "",
                ids[start:end],
                request,
                prompt_token_ids=prompt,
                finished=False,
            )
            assert parser.count_reasoning_tokens(ids[:end]) == sum(
                pos < end for pos in expected_positions
            )
        parser.parse_deltas(tokenizer.decode(ids), [], request, finished=True)
        assert parser.count_reasoning_tokens(ids) == len(expected_positions)
        # The underlying engine's non-streaming reset must not retain counts.
        engine = parser._mimo_parser if profile == "mimo" else parser._glm_parser
        engine.parse("answer", request, True, enc("answer"))
        assert engine.count_reasoning_tokens(enc("answer")) == len(enc("answer"))


def test_token_reasoning_accounting_checks_reachable_states(mock_tokenizer):
    from vllm.parser.engine.parser_engine_config import ParserState
    from vllm.parser.engine.streaming_parser_engine import TokenReasoningCounter

    # This tokenizer supplies tool IDs only. Disabled reasoning is unreachable,
    # so tool-only token accounting is valid and exactly zero.
    config = qwen3_config(thinking=False)
    counter = TokenReasoningCounter(config, mock_tokenizer)
    counter.feed_token_ids([100, 101])
    assert counter.reasoning_token_count == 0
    # Resetting into reasoning changes that contract: these missing boundaries
    # cannot be counted from IDs. Do not silently select text-based accounting.
    with pytest.raises(ValueError, match="ID-only reasoning"):
        counter.reset(initial_state=ParserState.REASONING)
    counter.reset(initial_state=ParserState.CONTENT)
    counter.feed_token_ids([100, 101])
    assert counter.reasoning_token_count == 0
