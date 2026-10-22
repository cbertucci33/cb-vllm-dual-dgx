# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual selected parsers must retain text on either side of tool calls."""

import json

import pytest

from tests.parser.engine.replay_harness import load_local_model_parser
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def _case(profile):
    tokenizer, cls = load_local_model_parser(profile)
    tool = (
        "<tool_call>echo<arg_key>text</arg_key><arg_value>x</arg_value></tool_call>"
        if profile == "glm"
        else "<tool_call><function=echo><parameter=text>x"
        "</parameter></function></tool_call>"
    )
    text = "<think>reason</think>before" + tool + "between" + tool + "after"
    ids = tokenizer.encode(text, add_special_tokens=False)
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
    return tokenizer, cls, request, text, ids


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize("mode", ["full", "terminal_chunk", "separate_finish"])
def test_selected_terminal_content_is_not_lost(profile, mode):
    tokenizer, cls, request, text, ids = _case(profile)
    parser = cls(tokenizer, request.tools)
    if mode == "full":
        reasoning, content, calls = parser.parse(text, request, True, ids)
        assert reasoning == "reason"
        assert content == "beforebetweenafter"
        assert [json.loads(c.arguments) for c in calls] == [{"text": "x"}] * 2
    else:
        deltas = [
            parser.parse_delta(text, ids, request, finished=mode == "terminal_chunk")
        ]
        if mode == "separate_finish":
            deltas.append(parser.parse_delta("", [], request, finished=True))
        assert "".join(d.content or "" for d in deltas if d) == "beforebetweenafter"
        calls = [c for d in deltas if d for c in d.tool_calls or []]
        assert [json.loads(c.function.arguments) for c in calls] == [{"text": "x"}] * 2


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize("schedule", ["whole", "single_token", "two_chunks"])
@pytest.mark.parametrize("visible_reasoning", [False, True])
def test_responses_preserves_semantic_item_order(profile, schedule, visible_reasoning):
    import asyncio

    from tests.entrypoints.openai.responses.test_message_phases import serving
    from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
    from vllm.entrypoints.openai.responses.context import SimpleContext
    from vllm.outputs import CompletionOutput, RequestOutput
    from vllm.sampling_params import SamplingParams

    tokenizer, cls, request, text, ids = _case(profile)
    request.include_reasoning = visible_reasoning
    schedules = (
        [[len(ids)]]
        if schedule == "whole"
        else [list(range(1, len(ids) + 1))]
        if schedule == "single_token"
        else [[split, len(ids)] for split in range(1, len(ids))]
    )

    async def run(ends):
        server = serving()
        server.tool_server = None
        server.response_store_lock = asyncio.Lock()
        server.response_store = {}
        server.msg_store = {}
        ctx = SimpleContext(response_parser=cls(tokenizer, request.tools))

        async def outputs():
            previous = ""
            start = 0
            for end in ends:
                decoded = tokenizer.decode(ids[:end]).rstrip("\ufffd")
                assert decoded.startswith(previous)
                final = end == len(ids)
                ctx.append_output(
                    RequestOutput(
                        request_id=request.request_id,
                        prompt="fixture",
                        prompt_token_ids=[],
                        prompt_logprobs=None,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text=decoded[len(previous) :],
                                token_ids=ids[start:end],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="stop" if final else None,
                            )
                        ],
                        finished=final,
                    )
                )
                previous, start = decoded, end
                yield ctx

        return [
            event
            async for event in server.responses_stream_generator(
                request,
                SamplingParams(max_tokens=128),
                outputs(),
                ctx,
                "test",
                tokenizer,
                RequestResponseMetadata(request_id=request.request_id),
                created_time=1,
            )
        ]

    for ends in schedules:
        events = asyncio.run(run(ends))
        done = [e for e in events if e.type == "response.output_item.done"]
        expected = ["message", "function_call", "message", "function_call", "message"]
        if visible_reasoning:
            expected.insert(0, "reasoning")
        assert [e.item.type for e in done] == expected, ends
        assert [e.output_index for e in done] == list(range(len(expected))), ends
        assert len({e.item.id for e in done}) == len(expected)
        assert [e.item.content[0].text for e in done if e.item.type == "message"] == [
            "before",
            "between",
            "after",
        ]
        terminal = events[-1]
        assert terminal.type == "response.completed"
        assert [item.model_dump() for item in terminal.response.output] == [
            e.item.model_dump() for e in done
        ]
        calls = [e.item for e in done if e.item.type == "function_call"]
        assert [c.name for c in calls] == ["echo", "echo"]
        assert [json.loads(c.arguments) for c in calls] == [{"text": "x"}] * 2


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize("choice", ["named", "required"])
@pytest.mark.parametrize("visible_reasoning", [False, True])
def test_ordered_api_retains_legacy_json_contract(profile, choice, visible_reasoning):
    import os

    if os.environ.get("VLLM_ENFORCE_STRICT_TOOL_CALLING") != "0":
        pytest.skip("legacy JSON uses a separate strict=0 process")
    tokenizer, cls, request, _, _ = _case(profile)
    payload = request.model_dump()
    payload["include_reasoning"] = visible_reasoning
    payload["tool_choice"] = (
        {"type": "function", "name": "echo"} if choice == "named" else "required"
    )
    request = ResponsesRequest.model_validate(payload)
    args = {"text": '<think>literal</think> "quoted"'}
    body = json.dumps(
        args if choice == "named" else [{"name": "echo", "parameters": args}]
    )
    text = "<think>reason</think>" + body
    ids = tokenizer.encode(text, add_special_tokens=False)
    parser = cls(tokenizer, request.tools)
    request = parser.adjust_request(request)
    deltas = parser.parse_deltas(text, ids, request, finished=True)
    assert "".join(d.reasoning or "" for d in deltas) == (
        "reason" if visible_reasoning else ""
    )
    assert not any(d.content for d in deltas)
    calls = [c for d in deltas for c in d.tool_calls or []]
    assert len(calls) == 1
    assert calls[0].function.name == "echo"
    assert json.loads(calls[0].function.arguments) == args
