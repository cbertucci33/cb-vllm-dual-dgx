# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7 parser for reasoning and tool calls.

GLM-4.7 uses XML-like tool calls::

    <tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>

The function name can be followed directly by the first ``<arg_key>`` tag,
and tool calls may have no arguments.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm import envs
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.json_tools import _json_tool_config, _RequiredJSONStream
from vllm.renderers.chat_utils import make_tool_call_id

logger = init_logger(__name__)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import FunctionCall
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_START = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_START = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

# Special tokens that delimit conversation turns in a rendered GLM prompt.
# Reasoning markers belonging to earlier turns must not be mistaken for the
# state of the turn currently being generated.
GLM_TURN_BOUNDARIES = frozenset(
    ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>")
)

_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)
_PARTIAL_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*)$",
    re.DOTALL,
)


def _glm47_parse_args(
    raw_args: str, partial: bool
) -> tuple[dict[str, object], str | None]:
    """Return the arguments and the key of the open trailing value, if any."""
    params: dict[str, object] = {}

    for match in _ARG_RE.finditer(raw_args):
        params[match.group("key").strip()] = match.group("value")

    open_key = None
    if partial:
        remaining = _ARG_RE.sub("", raw_args)
        match = _PARTIAL_ARG_RE.search(remaining)
        if match:
            key = match.group("key").strip()
            if key:
                params[key] = match.group("value")
                open_key = key

    return params, open_key


def _glm47_arg_converter(raw_args: str, partial: bool) -> str:
    return json.dumps(_glm47_parse_args(raw_args, partial)[0], ensure_ascii=False)


def _glm47_arg_value_extender(raw_args: str) -> ArgValueExtender | None:
    # The open value runs to the end of raw_args and is the last JSON value,
    # so a delta appends its JSON-escaped text unless it closes the value.
    params, open_key = _glm47_parse_args(raw_args, partial=True)
    if open_key is None or next(reversed(params)) != open_key:
        return None

    keep = len(ARG_VALUE_END) - 1
    tail = raw_args[-keep:]

    def extend(delta: str) -> str | None:
        nonlocal tail
        text = tail + delta
        if ARG_VALUE_END in text:
            return None
        tail = text[-keep:]
        return json.dumps(delta, ensure_ascii=False)[1:-1]

    return extend


def _glm53_always_thinks(tokenizer: TokenizerLike) -> bool:
    # GLM-5.3 keeps the GLM-4.x prompt and tool-call format but replaces the
    # enable_thinking toggle with an always-on "Reasoning Effort:" header, so
    # its generation prompt always opens <think> and the model output carries
    # no opening tag.
    template = getattr(tokenizer, "chat_template", None)
    return (
        isinstance(template, str)
        and "[gMASK]<sop>" in template
        and "Reasoning Effort:" in template
        and "enable_thinking" not in template
        and "<tool_call>" in template
        and "<arg_key>" in template
        and "<arg_value>" in template
    )


@functools.cache
def glm47_moe_config(thinking: bool = True) -> ParserEngineConfig:
    arg_tag_transitions = {
        (ParserState.TOOL_ARGS, terminal): Transition(
            ParserState.TOOL_ARGS,
            (EventType.ARG_VALUE_CHUNK,),
        )
        for terminal in (
            "ARG_KEY_START",
            "ARG_KEY_END",
            "ARG_VALUE_START",
            "ARG_VALUE_END",
        )
    }

    reasoning_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_token_id_terminals = (
        {
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        }
        if thinking
        else {}
    )
    reasoning_transitions = (
        {
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        }
        if thinking
        else {}
    )

    return ParserEngineConfig(
        name="glm47_moe",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            **reasoning_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
            "ARG_KEY_START": ARG_KEY_START,
            "ARG_KEY_END": ARG_KEY_END,
            "ARG_VALUE_START": ARG_VALUE_START,
            "ARG_VALUE_END": ARG_VALUE_END,
        },
        token_id_terminals={
            **reasoning_token_id_terminals,
            "TOOL_START": TOOL_CALL_START,
            "TOOL_END": TOOL_CALL_END,
        },
        transitions={
            **reasoning_transitions,
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "ARG_KEY_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.ARG_VALUE_CHUNK,),
            ),
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            **arg_tag_transitions,
        },
        turn_boundary_tokens=GLM_TURN_BOUNDARIES,
        arg_converter=_glm47_arg_converter,
        arg_value_extender=_glm47_arg_value_extender,
        stream_arg_deltas=True,
        tool_args_json=False,
        validate_tool_names=True,
    )


class Glm47MoeParser(ParserEngine):
    """GLM-4.7 parser backed by the deoperatortive parser engine."""

    token_id_reasoning_accounting = True
    token_id_text_alignment = True

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        if (thinking is False or enable_thinking is False) and _glm53_always_thinks(
            tokenizer
        ):
            logger.warning_once(
                "Ignoring enable_thinking/thinking: the GLM-5.3 chat template "
                "has no thinking switch, so reasoning is always on and "
                "disabling extraction would leak it into the content."
            )
            thinking = None
            enable_thinking = None
        self.thinking_enabled = (
            True
            if thinking is None and enable_thinking is None
            else bool(thinking) or bool(enable_thinking)
        )
        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        state = (
            ParserState.CONTENT
            if self.is_reasoning_end(prompt_token_ids)
            else self.parser_engine_config.initial_state
        )
        self.initialize_streaming(
            initial_state=state, prompt_token_ids=prompt_token_ids
        )

    def _emit_name_delta(self, idx: int, deltas, name: str | None) -> None:
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)


class Glm47MoeServingParser(DelegatingParser):
    """Retain adapter request contracts, but parse GLM in one state machine."""

    def __init__(
        self, tokenizer: TokenizerLike, tools: list[Tool] | None = None, **kwargs
    ) -> None:
        super().__init__(tokenizer, tools, **kwargs)
        # The adapter still owns metadata/structured-output hooks. Its exact
        # engine also owns generation parsing; there is no separate token pass.
        from vllm.parser.engine.adapters import ParserEngineReasoningAdapter

        assert isinstance(self._reasoning_parser, ParserEngineReasoningAdapter)
        engine = self._reasoning_parser._parser_engine
        assert isinstance(engine, Glm47MoeParser)
        if self._tool_parser is not None:
            # With tags disabled, adjust_request forces required/named JSON.
            # Native XML remains the auto and structural-tag protocol.
            self._tool_parser.supports_required_and_named = (
                not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING
            )
        self._glm_parser = engine
        self._glm_parser._tools = tools
        self._json_parser: Glm47MoeParser | None = None
        self._legacy_json_tools = False
        self._required_json_stream = _RequiredJSONStream()

    def _uses_json_tools(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> bool:
        return (
            self._tool_parser is not None
            and self._tool_parser.supports_required_and_named
            and request.tool_choice not in (None, "auto", "none")
        )

    def _generation_parser(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> Glm47MoeParser:
        self._legacy_json_tools = self._uses_json_tools(request)
        if not self._legacy_json_tools:
            self._glm_parser._suppress_tool_calls = request.tool_choice == "none"
            return self._glm_parser
        if self._json_parser is None:
            self._json_parser = Glm47MoeParser(
                self.model_tokenizer,
                self._glm_parser._tools,
                parser_engine_config=_json_tool_config(
                    self._glm_parser.parser_engine_config
                ),
            )
        return self._json_parser

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        self._initialize_history_tool_call_cnt(request)
        parser = self._generation_parser(request)
        parser.skip_tool_parsing = not self._legacy_json_tools and (
            self._tool_parser is None or not enable_auto_tools
        )
        reasoning, content, calls = parser.parse(
            model_output, request, enable_auto_tools, model_output_token_ids
        )
        if self._legacy_json_tools:
            calls, content = self._extract_tool_calls(
                content, request, enable_auto_tools
            )
        return reasoning, content, calls

    def parse_deltas(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> list[DeltaMessage]:
        parser = self._generation_parser(request)
        if self._legacy_json_tools:
            # JSON constraints still use the existing named/required adapter;
            # these envelopes cannot contain post-tool ordinary text.
            return super().parse_deltas(
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids,
                finished=finished,
            )
        parser.skip_tool_parsing = self._tool_parser is None
        return parser.parse_deltas(
            delta_text, delta_token_ids, request, prompt_token_ids, finished=finished
        )

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        parser = self._generation_parser(request)
        parser.skip_tool_parsing = (
            not self._legacy_json_tools and self._tool_parser is None
        )
        delta = parser.parse_delta(
            delta_text, delta_token_ids, request, prompt_token_ids, finished=finished
        )
        if not self._legacy_json_tools or delta is None:
            return delta
        self._initialize_history_tool_call_cnt(request)
        state = self._stream_state
        tool_delta = None
        if delta.content:
            if request.tool_choice == "required":
                tool_delta = self._required_json_delta(delta.content)
            else:
                tool_delta, state.function_name_returned = (
                    self._extract_tool_calls_streaming(
                        "",
                        delta.content,
                        delta.content,
                        [],
                        [],
                        [],
                        request,
                        tool_call_idx=state.history_tool_call_cnt,
                        tool_call_id_type=state.tool_call_id_type,
                        function_name_returned=state.function_name_returned,
                    )
                )
            state.history_tool_call_cnt += (
                sum(call.id is not None for call in (tool_delta.tool_calls or []))
                if tool_delta is not None
                else 0
            )
        if delta.reasoning:
            if tool_delta is None:
                return DeltaMessage(reasoning=delta.reasoning)
            tool_delta.reasoning = delta.reasoning
        return tool_delta

    def _required_json_delta(self, text: str) -> DeltaMessage | None:
        calls = []
        state = self._stream_state
        ids_created = 0
        for index, name, arguments, first in self._required_json_stream.feed(text):
            if first:
                call = DeltaToolCall(
                    index=index,
                    id=make_tool_call_id(
                        id_type=state.tool_call_id_type,
                        func_name=name,
                        idx=state.history_tool_call_cnt + ids_created,
                    ),
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=arguments),
                )
                ids_created += 1
            else:
                call = DeltaToolCall(
                    index=index,
                    function=DeltaFunctionCall(arguments=arguments),
                )
            calls.append(call)
        return DeltaMessage(tool_calls=calls) if calls else None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        parser = self._json_parser if self._legacy_json_tools else self._glm_parser
        assert parser is not None
        return parser.count_reasoning_tokens(token_ids)
