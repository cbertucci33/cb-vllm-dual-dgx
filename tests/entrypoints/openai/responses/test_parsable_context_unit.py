# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ParsableContext's parsing behavior.

These tests verify that ParsableContext correctly delegates to the unified
Parser (via parse) and properly builds response output items.
"""

import json
from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.responses import ResponseFunctionToolCall

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.mcp.tool import HarmonyPythonTool
from vllm.entrypoints.openai.responses.context import ParsableContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import DelegatingParser

pytestmark = pytest.mark.skip_global_cleanup


# ---------------------------------------------------------------------------
# Test parser stubs
# ---------------------------------------------------------------------------


class _NoOpParser(DelegatingParser):
    """Parser that extracts no reasoning and no tool calls."""

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def extract_reasoning(self, model_output, request):
        return None, model_output

    def extract_reasoning_streaming(self, *args, **kwargs):
        return None

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def parse_delta(self, *args, **kwargs) -> DeltaMessage | None:
        return None


class _ReasoningOnlyParser(DelegatingParser):
    """Parser that extracts reasoning but no tool calls."""

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def extract_reasoning(self, model_output, request):
        if "<think>" in model_output and "</think>" in model_output:
            start = model_output.index("<think>") + len("<think>")
            end = model_output.index("</think>")
            reasoning = model_output[start:end]
            content = model_output[end + len("</think>") :]
            return reasoning, content.strip() or None
        return None, model_output

    def extract_reasoning_streaming(self, *args, **kwargs):
        return None

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def parse_delta(self, *args, **kwargs) -> DeltaMessage | None:
        return None


class _CountingParser(_NoOpParser):
    """Parser that records the ids it counts; token id 7 is reasoning."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer)
        self.counted_ids: list[list[int]] = []

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        self.counted_ids.append(list(token_ids))
        return sum(token_id == 7 for token_id in token_ids)


class _StubToolParser:
    """Minimal tool parser stub that always returns a hardcoded tool call."""

    supports_required_and_named = False

    def __init__(self, tokenizer=None, tools=None):
        pass

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    id="call_123",
                    type="function",
                    function=FunctionCall(
                        name="get_weather",
                        arguments='{"location": "Paris"}',
                    ),
                )
            ],
            content=None,
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def adjust_request(self, request):
        return request


class _ToolCallingParser(DelegatingParser):
    """Parser that extracts a hardcoded tool call from any input."""

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer)
        self._tool_parser = _StubToolParser()

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def extract_reasoning(self, model_output, request):
        return None, model_output

    def extract_reasoning_streaming(self, *args, **kwargs):
        return None

    def extract_tool_calls_streaming(self, *args, **kwargs):
        return None

    def parse_delta(self, *args, **kwargs) -> DeltaMessage | None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(**overrides) -> ResponsesRequest:
    defaults = {"model": "test-model", "input": "test"}
    defaults.update(overrides)
    return ResponsesRequest.model_validate(defaults)


def _make_request_output(
    text: str = "Hello, world!",
    token_ids: Sequence[int] = (1, 2, 3),
    finish_reason: str = "stop",
    num_cache_creation_tokens: int | None = None,
) -> RequestOutput:
    return RequestOutput(
        request_id="test",
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=list(token_ids),
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=finish_reason,
            )
        ],
        finished=True,
        num_cache_creation_tokens=num_cache_creation_tokens,
    )


def _make_context(parser_cls, **overrides):
    # ParsableContext no longer lazily builds a parser from ``parser_cls``;
    # the caller (here, the serving layer in production) must supply one.
    request = overrides.get("request", _make_request())
    response_parser = overrides.pop("response_parser", None)
    if response_parser is None and parser_cls is not None:
        response_parser = parser_cls(MagicMock(), request.tools)

    defaults = dict(
        tokenizer=MagicMock(),
        parser_cls=parser_cls,
        response_parser=response_parser,
        response_messages=[],
        request=request,
        available_tools=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    defaults.update(overrides)
    return ParsableContext(**defaults)


# ---------------------------------------------------------------------------
# Tests: basic text passthrough
# ---------------------------------------------------------------------------


def test_process_text_with_parser():
    """Parser with no reasoning/tools returns a single message item."""
    ctx = _make_context(_NoOpParser)
    ctx.append_output(_make_request_output(text="Hello!"))

    assert len(ctx.response_messages) == 1
    msg = ctx.response_messages[0]
    assert msg.type == "message"
    assert msg.content[0].text == "Hello!"


def test_process_text_without_parser():
    """parser_cls=None falls back to plain text wrapping."""
    ctx = _make_context(None)
    ctx.append_output(_make_request_output(text="Hello!"))

    assert len(ctx.response_messages) == 1
    msg = ctx.response_messages[0]
    assert msg.type == "message"
    assert msg.content[0].text == "Hello!"


# ---------------------------------------------------------------------------
# Tests: empty / whitespace output
# ---------------------------------------------------------------------------


def test_process_empty_text_without_parser():
    """Empty text with no parser produces no output items."""
    ctx = _make_context(None)
    ctx.append_output(_make_request_output(text=""))

    assert len(ctx.response_messages) == 0


def test_process_empty_text_with_parser():
    """Empty text with parser produces no output items."""
    ctx = _make_context(_NoOpParser)
    ctx.append_output(_make_request_output(text=""))

    assert len(ctx.response_messages) == 0


# ---------------------------------------------------------------------------
# Tests: reasoning extraction
# ---------------------------------------------------------------------------


def test_process_extracts_reasoning():
    """Parser that finds reasoning produces both reasoning and message items."""
    ctx = _make_context(_ReasoningOnlyParser)
    ctx.append_output(
        _make_request_output(text="<think>Let me check</think>The answer is 42")
    )

    types = [m.type for m in ctx.response_messages]
    assert "reasoning" in types
    assert "message" in types

    reasoning_item = next(m for m in ctx.response_messages if m.type == "reasoning")
    assert reasoning_item.content[0].text == "Let me check"

    message_item = next(m for m in ctx.response_messages if m.type == "message")
    assert message_item.content[0].text == "The answer is 42"


def test_process_reasoning_only_no_content():
    """When reasoning consumes all text, only a reasoning item is produced."""
    ctx = _make_context(_ReasoningOnlyParser)
    ctx.append_output(_make_request_output(text="<think>Just thinking</think>"))

    types = [m.type for m in ctx.response_messages]
    assert "reasoning" in types
    assert "message" not in types


# ---------------------------------------------------------------------------
# Tests: tool call extraction
# ---------------------------------------------------------------------------


def test_process_extracts_tool_calls():
    """Parser that finds tool calls produces function_call items."""
    request = _make_request(
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    ctx = _make_context(_ToolCallingParser, request=request, enable_auto_tools=True)
    ctx.append_output(_make_request_output(text="calling tool"))

    types = [m.type for m in ctx.response_messages]
    assert "function_call" in types

    tool_item = next(m for m in ctx.response_messages if m.type == "function_call")
    assert tool_item.name == "get_weather"
    assert tool_item.arguments == '{"location": "Paris"}'
    assert tool_item.status == "completed"


@pytest.mark.parametrize(
    ("tool_name", "session_name", "dispatched_name", "arguments"),
    [
        ("code_interpreter", "python", "python", {"code": "print(42)"}),
        ("web_search_preview", "browser", "search", {"query": "vLLM"}),
        ("container.exec", "container", "exec", {"cmd": ["pwd"]}),
    ],
)
@pytest.mark.asyncio
async def test_builtin_tool_output_preserves_function_call_id(
    tool_name, session_name, dispatched_name, arguments
):
    """Built-in tool outputs remain correlated with their originating call."""
    tool_session = MagicMock()
    tool_session.call_tool = AsyncMock(
        return_value=SimpleNamespace(content=[SimpleNamespace(text="result")])
    )
    context = _make_context(None, available_tools=[session_name])
    tool_call = ResponseFunctionToolCall(
        id=f"fc_{session_name}",
        call_id=f"call_{session_name}",
        type="function_call",
        name=tool_name,
        arguments=json.dumps(arguments),
    )
    context.response_messages.append(tool_call)
    context._tool_sessions[session_name] = tool_session

    output = await context.call_tool()

    tool_session.call_tool.assert_awaited_once_with(dispatched_name, arguments)
    assert output[0].call_id == tool_call.call_id


@pytest.mark.asyncio
async def test_local_python_tool_output_preserves_function_call_id():
    """Local code-interpreter output remains correlated with its call."""

    async def process(_):
        yield SimpleNamespace(content=[SimpleNamespace(text="result")])

    python_tool = object.__new__(HarmonyPythonTool)
    python_tool.python_tool = MagicMock(process=process)
    context = _make_context(None, available_tools=["python"])
    tool_call = ResponseFunctionToolCall(
        id="fc_python",
        call_id="call_python",
        type="function_call",
        name="code_interpreter",
        arguments='{"code": "print(42)"}',
    )
    context.response_messages.append(tool_call)
    context._tool_sessions["python"] = python_tool

    output = await context.call_tool()

    assert output[0].call_id == tool_call.call_id


# ---------------------------------------------------------------------------
# Tests: finish_reason tracking
# ---------------------------------------------------------------------------


def test_finish_reason_tracked():
    """finish_reason from CompletionOutput is stored on the context."""
    ctx = _make_context(_NoOpParser)
    assert ctx.finish_reason is None

    ctx.append_output(_make_request_output(finish_reason="stop"))
    assert ctx.finish_reason == "stop"

    ctx.append_output(_make_request_output(finish_reason="length"))
    assert ctx.finish_reason == "length"


# ---------------------------------------------------------------------------
# Tests: multi-turn accumulation
# ---------------------------------------------------------------------------


def test_multi_turn_accumulation():
    """Multiple append_output() calls accumulate response_messages."""
    ctx = _make_context(_NoOpParser)

    ctx.append_output(_make_request_output(text="First turn"))
    ctx.append_output(_make_request_output(text="Second turn"))

    assert len(ctx.response_messages) == 2
    texts = [m.content[0].text for m in ctx.response_messages]
    assert texts == ["First turn", "Second turn"]


def test_reasoning_tokens_counted_per_round():
    """Each round is counted on its own ids, not the concatenated rounds."""
    ctx = _make_context(_CountingParser)

    ctx.append_output(_make_request_output(token_ids=[7, 7, 1]))
    ctx.append_output(_make_request_output(token_ids=[7, 2]))

    assert ctx.response_parser.counted_ids == [[7, 7, 1], [7, 2]]
    assert ctx.num_reasoning_tokens == 3


def test_cache_creation_tokens_counted_per_round():
    ctx = _make_context(None)

    for count, expected in [(None, 0), (0, 0), (3, 3), (4, 7)]:
        ctx.append_output(_make_request_output(num_cache_creation_tokens=count))
        assert ctx.num_cache_creation_tokens == expected


def test_reasoning_tokens_zero_without_parser():
    ctx = _make_context(None)

    ctx.append_output(_make_request_output(token_ids=[7, 7, 1]))

    assert ctx.num_reasoning_tokens == 0


def test_num_init_messages_offset():
    """Initial messages are preserved and offset works correctly."""
    init_messages = [MagicMock(type="message")]
    ctx = _make_context(_NoOpParser, response_messages=init_messages)

    assert ctx.num_init_messages == 1

    ctx.append_output(_make_request_output(text="New output"))

    assert len(ctx.response_messages) == 2
    items = ctx.make_response_output_items()
    assert len(items) == 1
    assert items[0].type == "message"


@pytest.mark.parametrize("profile", ["mimo", "glm"])
@pytest.mark.parametrize(
    "tool_name,method",
    [
        ("code_interpreter", "call_python_tool"),
        ("web_search_preview", "call_search_tool"),
        ("container.exec", "call_container_tool"),
    ],
)
def test_real_parser_builtin_tool_result_retains_call_id(profile, tool_name, method):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.utils import (
        construct_chat_messages_with_tool_call,
    )

    tokenizer, cls = load_local_model_parser(profile)
    request = ResponsesRequest(
        model="test",
        input="hi",
        include_reasoning=True,
        tools=[
            {
                "type": "function",
                "name": tool_name,
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                },
            }
        ],
    )
    if profile == "mimo":
        text = (
            "<think>reason</think><tool_call><function="
            + tool_name
            + "><parameter=code>pass</parameter></function></tool_call>"
        )
    else:
        text = (
            "<think>reason</think><tool_call>"
            + tool_name
            + "<arg_key>code</arg_key><arg_value>pass</arg_value></tool_call>"
        )
    context = ParsableContext(
        response_messages=[],
        tokenizer=tokenizer,
        parser_cls=cls,
        request=request,
        available_tools=[],
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
    )
    context.append_output(
        RequestOutput(
            request_id="test",
            prompt="hi",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=text,
                    token_ids=tokenizer.encode(text, add_special_tokens=False),
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
        )
    )
    call = context.response_messages[-1]
    assert call.type == "function_call"
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(text="recorded result")]
            )
        )
    )
    result = asyncio.run(getattr(context, method)(session, call))
    assert result[0].call_id == call.call_id
    messages = construct_chat_messages_with_tool_call([call, result[0]])
    assert messages[0]["tool_calls"][0]["id"] == messages[1]["tool_call_id"]
    assert messages[1]["content"] == "recorded result"


@pytest.mark.parametrize(
    "method", ["call_python_tool", "call_search_tool", "call_container_tool"]
)
@pytest.mark.parametrize("retry", [False, True])
def test_parsable_malformed_json_retry_returns_matching_response_item(
    method, retry, monkeypatch
):
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from openai.types.responses import ResponseFunctionToolCall

    monkeypatch.setenv("VLLM_TOOL_JSON_ERROR_AUTOMATIC_RETRY", "1" if retry else "0")
    context = object.__new__(ParsableContext)
    context.called_tools = set()
    call = ResponseFunctionToolCall(
        id="fc_source",
        call_id="call_source",
        type="function_call",
        name="web_search_preview" if method == "call_search_tool" else "container.exec",
        arguments="{",
    )
    session = SimpleNamespace(call_tool=AsyncMock())
    if not retry:
        with pytest.raises(json.JSONDecodeError):
            asyncio.run(getattr(context, method)(session, call))
    else:
        result = asyncio.run(getattr(context, method)(session, call))
        assert result[0].type == "function_call_output"
        assert result[0].call_id == "call_source"
        assert "Error parsing tool arguments as JSON:" in result[0].output
    session.call_tool.assert_not_called()


def test_parsable_dispatches_all_builtin_calls_by_id_in_model_order():
    """Parallel call envelopes must not drop earlier calls or pair by adjacency."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from openai.types.responses import ResponseFunctionToolCall

    ctx = _make_context(None, available_tools=["python", "browser"])
    calls = [
        ResponseFunctionToolCall(
            id=f"fc_{i}",
            call_id=f"call_{i}",
            type="function_call",
            name=name,
            arguments='{"code":"pass","query":"x"}',
        )
        for i, name in enumerate(["python", "web_search_preview", "code_interpreter"])
    ]
    foreign = ResponseFunctionToolCall(
        id="fc_external",
        call_id="call_external",
        type="function_call",
        name="caller_owned",
        arguments="{}",
    )
    ctx.response_messages.extend([*calls, foreign])
    recorded = []

    async def invoke(name, args):
        recorded.append(name)
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"result-{len(recorded)}")]
        )

    ctx._tool_sessions = {
        name: SimpleNamespace(call_tool=AsyncMock(side_effect=invoke))
        for name in ("python", "browser")
    }
    before = [call.model_dump() for call in calls]
    assert ctx.need_builtin_tool_call()
    outputs = asyncio.run(ctx.call_tool())
    assert recorded == ["python", "search", "python"]
    assert [item.call_id for item in outputs] == [call.call_id for call in calls]
    assert [call.model_dump() for call in calls] == before
    assert asyncio.run(ctx.call_tool()) == []
    ctx.append_tool_output(list(reversed(outputs)))
    assert not ctx.need_builtin_tool_call()
    rendered = ctx.make_response_output_items()
    assert [item.output for item in rendered[:3]] == [
        "result-1",
        "result-2",
        "result-3",
    ]
    assert rendered[3] is foreign


def test_parsable_real_python_tool_adapter_preserves_selected_call_id():
    """Exercise the real Tool adapter with a no-side-effect backend iterator."""
    import asyncio

    from openai.types.responses import ResponseFunctionToolCall
    from openai_harmony import Author, Message, Role, TextContent

    from vllm.entrypoints.mcp.tool import HarmonyPythonTool

    class Backend:
        async def process(self, message):
            yield Message(
                author=Author(role=Role.TOOL, name="python"),
                content=[
                    TextContent(text=message.content[0].text),
                    TextContent(text=" tail"),
                ],
            )

    tool = object.__new__(HarmonyPythonTool)
    tool.python_tool = Backend()
    ctx = _make_context(None, available_tools=["python"])
    ctx._tool_sessions = {"python": tool}
    for i in range(2):
        ctx.response_messages.append(
            ResponseFunctionToolCall(
                id=f"fc_{i}",
                call_id=f"call_{i}",
                type="function_call",
                name="code_interpreter",
                arguments=f'{{"code":"value-{i}"}}',
            )
        )
    outputs = asyncio.run(ctx.call_tool())
    assert [(item.call_id, item.output) for item in outputs] == [
        ("call_0", "value-0 tail"),
        ("call_1", "value-1 tail"),
    ]
    assert ctx.active_tool_call is None


def test_parsable_cancelled_dispatch_is_not_replayed():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from openai.types.responses import ResponseFunctionToolCall

    ctx = _make_context(None, available_tools=["python"])
    call = ResponseFunctionToolCall(
        id="fc_cancel",
        call_id="call_cancel",
        type="function_call",
        name="python",
        arguments='{"code":"pass"}',
    )
    ctx.response_messages.append(call)
    session = SimpleNamespace(call_tool=AsyncMock(side_effect=asyncio.CancelledError))
    ctx._tool_sessions = {"python": session}
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ctx.call_tool())
    assert asyncio.run(ctx.call_tool()) == []
    assert session.call_tool.await_count == 1


def test_parsable_real_browser_tool_adapter_preserves_arguments_and_call_id():
    import asyncio

    from openai.types.responses import ResponseFunctionToolCall
    from openai_harmony import Author, Message, Role, TextContent

    from vllm.entrypoints.mcp.tool import HarmonyBrowserTool

    observed = []

    class Backend:
        async def process(self, message):
            observed.append(message)
            yield Message(
                author=Author(role=Role.TOOL, name="browser"),
                content=[TextContent(text="search"), TextContent(text=" result")],
            )

    tool = object.__new__(HarmonyBrowserTool)
    tool.browser_tool = Backend()
    ctx = _make_context(None, available_tools=["browser"])
    ctx._tool_sessions = {"browser": tool}
    call = ResponseFunctionToolCall(
        id="fc_search",
        call_id="call_search",
        type="function_call",
        name="web_search_preview",
        arguments='{"query":"exact query"}',
    )
    ctx.response_messages.append(call)
    outputs = asyncio.run(ctx.call_tool())
    assert outputs[0].call_id == call.call_id
    assert outputs[0].output == "search result"
    assert observed[0].recipient == "browser.search"
    assert observed[0].content[0].text == call.arguments
    assert ctx.active_tool_call is None


@pytest.mark.parametrize(
    "method",
    [
        "call_python_tool",
        "call_search_tool",
        "call_container_tool",
    ],
)
@pytest.mark.parametrize("parts", [[], ["first", " second", " 中文"]])
def test_parsable_builtin_preserves_all_text_result_parts(method, parts):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mcp.types import CallToolResult, TextContent
    from openai.types.responses import ResponseFunctionToolCall

    ctx = _make_context(None)
    call = ResponseFunctionToolCall(
        id="fc_parts",
        call_id="call_parts",
        name="echo",
        type="function_call",
        arguments='{"code":"pass"}',
    )
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text=part) for part in parts],
            )
        )
    )
    output = asyncio.run(getattr(ctx, method)(session, call))
    assert len(output) == 1
    assert output[0].call_id == "call_parts"
    assert output[0].output == "".join(parts)
    session.call_tool.assert_awaited_once()


@pytest.mark.parametrize(
    "method", ["call_python_tool", "call_search_tool", "call_container_tool"]
)
@pytest.mark.parametrize("kind", ["image", "structured", "resource"])
def test_parsable_builtin_preserves_rich_mcp_output(method, kind):
    """Valid MCP payloads must survive dispatch, continuation and delivery."""
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mcp.types import CallToolResult
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
    )

    from vllm.entrypoints.openai.responses.streaming_events import (
        ParsableBuiltinEventBridge,
    )
    from vllm.entrypoints.openai.responses.utils import (
        construct_chat_messages_with_tool_call,
    )
    from vllm.renderers.chat_utils import _parse_chat_message_content_mm_part

    payload: dict[str, object]
    if kind == "image":
        payload = {
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
                {"type": "text", "text": "after"},
            ]
        }
    elif kind == "structured":
        payload = {
            "content": [],
            "structuredContent": {"count": 3, "ok": True},
        }
    else:
        payload = {
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "uri": "test://report",
                        "mimeType": "text/plain",
                        "text": "report body",
                    },
                }
            ],
            "isError": True,
        }
    result = CallToolResult.model_validate(payload)
    ctx = _make_context(None)
    call = ResponseFunctionToolCall(
        id="fc_rich",
        call_id="call_rich",
        name="python",
        type="function_call",
        arguments='{"code":"pass"}',
    )
    ctx.response_messages.append(call)
    session = SimpleNamespace(call_tool=AsyncMock(return_value=result))
    outputs = asyncio.run(getattr(ctx, method)(session, call))
    ctx.append_tool_output(outputs)
    messages = construct_chat_messages_with_tool_call(outputs)
    assert messages[0]["tool_call_id"] == call.call_id
    content = messages[0]["content"]
    if kind == "image":
        assert [part["type"] for part in content] == [
            "input_text",
            "input_image",
            "input_text",
        ]
        assert content[0]["text"] == "before"
        assert content[2]["text"] == "after"
        assert _parse_chat_message_content_mm_part(content[1]) == (
            "input_image",
            "data:image/png;base64,aGVsbG8=",
        )
    else:
        assert isinstance(content, str)
        if kind == "structured":
            assert json.loads(content)["structuredContent"] == {"count": 3, "ok": True}
        else:
            assert "report body" in content
            assert "test://report" in content
            assert '"isError": true' in content

    bridge = ParsableBuiltinEventBridge(lambda name: "python")
    bridge.process(
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=0,
            item=call,
        )
    )
    bridge.process(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=0,
            item=call,
        )
    )
    done = bridge.complete(outputs)[-1].item
    final = ctx.make_response_output_items()[0]
    assert done.output == final.output
    assert done.id == final.id == "mcp_call_rich"
    if kind == "image":
        assert json.loads(done.output) == content
