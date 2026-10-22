# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo's compact XML parameters contain verbatim string values."""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import (
    ParserState,
    Transition,
)
from vllm.parser.json_tools import _json_tool_config, _RequiredJSONStream
from vllm.parser.qwen3 import (
    _PARAM_RE,
    _PARTIAL_PARAM_RE,
    CHATML_TURN_BOUNDARIES,
    Qwen3Parser,
    _param_value_extender,
    qwen3_config,
)
from vllm.renderers.chat_utils import make_tool_call_id

if TYPE_CHECKING:
    from vllm.entrypoints.generate.base.protocol import FunctionCall
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


def _mimo_arg_converter(raw_args: str, partial: bool) -> str:
    params = {m.group(1): m.group(2) for m in _PARAM_RE.finditer(raw_args)}
    if partial:
        match = _PARTIAL_PARAM_RE.search(_PARAM_RE.sub("", raw_args))
        if match and match.group(1):
            params[match.group(1)] = match.group(2)
    return json.dumps(params, ensure_ascii=False)


class MiMoParser(Qwen3Parser):
    token_id_reasoning_accounting = True
    token_id_text_alignment = True

    def __init__(self, tokenizer, tools=None, **kwargs):
        thinking = (kwargs.get("chat_template_kwargs") or {}).get(
            "enable_thinking", True
        )
        config = qwen3_config(
            thinking=thinking,
            name="mimo",
            turn_boundary_tokens=CHATML_TURN_BOUNDARIES,
        )
        transitions = dict(config.transitions)
        transitions[ParserState.CONTENT, "THINK_START"] = Transition(
            ParserState.REASONING, (EventType.REASONING_START,)
        )
        kwargs.setdefault(
            "parser_engine_config",
            replace(
                config,
                transitions=transitions,
                arg_converter=_mimo_arg_converter,
                arg_value_extender=functools.partial(
                    _param_value_extender, trim_newlines=False
                ),
            ),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        state = (
            ParserState.CONTENT
            if self.is_reasoning_end(list(prompt_token_ids))
            else self.parser_engine_config.initial_state
        )
        self.initialize_streaming(
            initial_state=state, prompt_token_ids=prompt_token_ids
        )


class MiMoServingParser(DelegatingParser):
    """Retain adapter request contracts, but parse MiMo in one state machine."""

    def __init__(
        self, tokenizer: TokenizerLike, tools: list[Tool] | None = None, **kwargs
    ) -> None:
        super().__init__(tokenizer, tools, **kwargs)
        # The adapter still owns metadata/structured-output hooks. Its exact
        # engine also owns generation parsing; there is no separate token pass.
        from vllm.parser.engine.adapters import ParserEngineReasoningAdapter

        assert isinstance(self._reasoning_parser, ParserEngineReasoningAdapter)
        engine = self._reasoning_parser._parser_engine
        assert isinstance(engine, MiMoParser)
        self._mimo_parser = engine
        self._mimo_parser._tools = tools
        self._json_parser: MiMoParser | None = None
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
    ) -> MiMoParser:
        self._legacy_json_tools = self._uses_json_tools(request)
        if not self._legacy_json_tools:
            self._mimo_parser._suppress_tool_calls = request.tool_choice == "none"
            return self._mimo_parser
        if self._json_parser is None:
            self._json_parser = MiMoParser(
                self.model_tokenizer,
                self._mimo_parser._tools,
                parser_engine_config=_json_tool_config(
                    self._mimo_parser.parser_engine_config
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
        parser = self._json_parser if self._legacy_json_tools else self._mimo_parser
        assert parser is not None
        return parser.count_reasoning_tokens(token_ids)
