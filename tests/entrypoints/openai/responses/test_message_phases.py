# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contracts for non-Harmony Responses item lifecycle.

Non-Harmony text has no model-provided channel: tool presence and finish reason
must not synthesize commentary/final-answer phases.

These exercise the production serving generator, with parsed deltas as the
input boundary. They do not generate tokens or test reasoning extraction.
Run directly with unittest to avoid the model-server pytest fixtures.
"""

import asyncio
import unittest

import pytest

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.context import ParsableContext, SimpleContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.exceptions import GenerationError
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import Parser
from vllm.sampling_params import SamplingParams


def tool(index=0, name="exec"):
    return DeltaToolCall(
        index=index,
        id=f"call_fixture_{index}",
        function=DeltaFunctionCall(name=name, arguments="{}"),
    )


class ParsedFixture(Parser):
    """Inject the established parser output boundary, not a parser replacement."""

    def __init__(self, deltas):
        self.deltas = iter(deltas)
        self.all_deltas = deltas

    def parse_delta(self, **kwargs):
        return next(self.deltas)

    def parse(self, *args, **kwargs):
        reasoning = "".join(d.reasoning or "" for d in self.all_deltas)
        content = "".join(d.content or "" for d in self.all_deltas)
        calls = [
            FunctionCall(id=t.id, name=t.function.name, arguments=t.function.arguments)
            for d in self.all_deltas
            for t in (d.tool_calls or [])
        ]
        return reasoning, content, calls

    def count_reasoning_tokens(self, token_ids):
        return 0


def serving():
    server = object.__new__(OpenAIServingResponses)
    server.use_harmony = False
    server.parser = object()
    server.enable_auto_tools = True
    server.enable_log_outputs = False
    server.enable_per_request_metrics = False
    return server


def messages(events):
    return [
        e.item
        for e in events
        if e.type == "response.output_item.done" and e.item.type == "message"
    ]


async def stream(deltas, finish_reason="stop", on_event=None):
    server = serving()
    request = ResponsesRequest(
        model="phase-fixture", input="fixture", store=False, include_reasoning=True
    )
    context = SimpleContext(response_parser=ParsedFixture(deltas))

    async def outputs():
        for index, _ in enumerate(deltas):
            finish = finish_reason if index == len(deltas) - 1 else None
            context.append_output(
                RequestOutput(
                    request_id=request.request_id,
                    prompt="fixture",
                    prompt_token_ids=[0],
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="fixture",
                            token_ids=[1],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason=finish,
                        )
                    ],
                    finished=finish is not None,
                )
            )
            yield context

    events = []
    async for event in server.responses_stream_generator(
        request,
        SamplingParams(max_tokens=32),
        outputs(),
        context,
        "phase-fixture",
        None,
        RequestResponseMetadata(request_id=request.request_id),
        created_time=1,
    ):
        if on_event is not None:
            on_event(event, context)
        events.append(event)
    return events


class TestMessageLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_pre_tool_text_keeps_phase_unset(self):
        events = await stream(
            [
                DeltaMessage(reasoning="private reasoning remains private"),
                DeltaMessage(content="First, create the test auth file."),
                DeltaMessage(tool_calls=[tool()]),
            ]
        )
        self.assertEqual(messages(events)[0].phase, None)
        self.assertEqual(
            messages(events)[0].content[0].text, "First, create the test auth file."
        )
        reasoning = [
            e.item
            for e in events
            if e.type == "response.output_item.done" and e.item.type == "reasoning"
        ]
        self.assertEqual(
            reasoning[0].content[0].text, "private reasoning remains private"
        )
        done = [e.item for e in events if e.type == "response.output_item.done"]
        terminal = events[-1].response.output
        self.assertEqual(
            [i.model_dump() for i in done], [i.model_dump() for i in terminal]
        )
        self.assertTrue(all(a is b for a, b in zip(done, terminal)))
        added = [e.item for e in events if e.type == "response.output_item.added"]
        self.assertEqual([i.id for i in added], [i.id for i in done])
        self.assertEqual(done[-1].call_id, "call_fixture_0")
        self.assertEqual([e.sequence_number for e in events], list(range(len(events))))

    async def test_normal_answer_keeps_phase_unset(self):
        events = await stream(
            [
                DeltaMessage(reasoning="private"),
                DeltaMessage(content="Done."),
            ]
        )
        self.assertEqual(messages(events)[0].phase, None)
        added = [
            e.item
            for e in events
            if e.type == "response.output_item.added" and e.item.type == "message"
        ]
        self.assertIsNone(added[0].phase)

    async def test_visible_deltas_do_not_wait_for_terminal_phase(self):
        observed = []

        def on_event(event, context):
            if event.type == "response.output_text.delta":
                observed.append((event.delta, context.num_output_tokens))

        await stream(
            [
                DeltaMessage(content="first"),
                DeltaMessage(content="second"),
                DeltaMessage(tool_calls=[tool()]),
            ],
            on_event=on_event,
        )
        self.assertEqual(observed, [("first", 1), ("second", 2)])

    async def test_parallel_tools_keep_names_ids_and_arguments(self):
        events = await stream(
            [
                DeltaMessage(content="Checking both."),
                DeltaMessage(tool_calls=[tool(0, "first"), tool(1, "second")]),
            ]
        )
        self.assertEqual(messages(events)[0].phase, None)
        calls = [i for i in events[-1].response.output if i.type == "function_call"]
        self.assertEqual([i.name for i in calls], ["first", "second"])
        self.assertEqual(
            [i.call_id for i in calls], ["call_fixture_0", "call_fixture_1"]
        )
        self.assertEqual([i.arguments for i in calls], ["{}", "{}"])

    async def test_content_reasoning_tool_interleaving_preserves_order(self):
        events = await stream(
            [
                DeltaMessage(content="Checking."),
                DeltaMessage(reasoning="more private reasoning"),
                DeltaMessage(tool_calls=[tool()]),
            ]
        )
        self.assertEqual(messages(events)[0].phase, None)
        done = [e.item.type for e in events if e.type == "response.output_item.done"]
        self.assertEqual(done, ["message", "reasoning", "function_call"])
        done_indexes = [
            e.output_index for e in events if e.type == "response.output_item.done"
        ]
        self.assertEqual(done_indexes, [0, 1, 2])

    async def test_multiple_visible_segments_only_last_is_final(self):
        events = await stream(
            [
                DeltaMessage(content="Working."),
                DeltaMessage(reasoning="private"),
                DeltaMessage(content="Done."),
            ]
        )
        self.assertEqual(
            [m.phase for m in messages(events)],
            [None for _ in [m.phase for m in messages(events)]],
        )

    async def test_tool_only_yield_does_not_invent_an_answer(self):
        events = await stream([DeltaMessage(tool_calls=[tool(0, "sessions_yield")])])
        self.assertEqual(messages(events), [])
        self.assertEqual(
            [i.type for i in events[-1].response.output], ["function_call"]
        )

    async def test_tool_result_continuation_keeps_phase_unset(self):
        first = await stream(
            [
                DeltaMessage(content="I will run it."),
                DeltaMessage(tool_calls=[tool()]),
            ]
        )
        following = await stream([DeltaMessage(content="It succeeded.")])
        self.assertEqual(messages(first)[0].phase, None)
        self.assertEqual(messages(following)[0].phase, None)

    async def test_length_abort_and_unknown_are_never_final(self):
        for reason in ("length", "abort", None):
            with self.subTest(reason=reason):
                events = await stream([DeltaMessage(content="Partial")], reason)
                self.assertNotEqual(messages(events)[0].phase, "final_answer")

    async def test_error_does_not_complete_a_final_message(self):
        # Stop at the repaired generator boundary. The outer error envelope
        # has a separate existing SDK-union validation issue, outside this fix.
        server = serving()
        request = ResponsesRequest(model="phase-fixture", input="fixture")
        context = SimpleContext()
        seen = []

        async def outputs():
            for text, finish in [("Partial", None), ("", "error")]:
                context.last_output = RequestOutput(
                    request_id=request.request_id,
                    prompt="fixture",
                    prompt_token_ids=[0],
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=text,
                            token_ids=[1],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason=finish,
                        )
                    ],
                    finished=finish is not None,
                )
                yield context

        with self.assertRaises(GenerationError):
            async for event in server._process_simple_streaming_events(
                request,
                SamplingParams(max_tokens=32),
                outputs(),
                context,
                "phase-fixture",
                None,
                RequestResponseMetadata(request_id=request.request_id),
                1,
                lambda event: event,
            ):
                seen.append(event)
        self.assertFalse(any(m.phase == "final_answer" for m in messages(seen)))

    async def test_cancel_does_not_flush_a_final_message(self):
        server = serving()
        request = ResponsesRequest(model="phase-fixture", input="fixture", store=False)
        context = SimpleContext()

        async def cancelled():
            raise asyncio.CancelledError
            yield context

        with self.assertRaises(asyncio.CancelledError):
            async for _ in server._process_simple_streaming_events(
                request,
                SamplingParams(max_tokens=32),
                cancelled(),
                context,
                "phase-fixture",
                None,
                RequestResponseMetadata(request_id=request.request_id),
                1,
                lambda event: event,
            ):
                pass

    def test_nonstream_keeps_phase_unset(self):
        server = serving()
        request = ResponsesRequest(model="phase-fixture", input="fixture")
        for reason, with_tools, expected in [
            ("stop", True, "commentary"),
            ("stop", False, "final_answer"),
            ("length", False, None),
            ("abort", False, None),
            (None, False, None),
        ]:
            with self.subTest(reason=reason, tools=with_tools):
                deltas = [DeltaMessage(content="visible")]
                if with_tools:
                    deltas.append(DeltaMessage(tool_calls=[tool()]))
                output = CompletionOutput(
                    index=0,
                    text="fixture",
                    token_ids=[1],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason=reason,
                )
                items = server._make_response_output_items(
                    request, output, None, parser=ParsedFixture(deltas)
                )
                message = next(i for i in items if i.type == "message")
                self.assertEqual(message.phase, None)

    def test_parsable_context_rounds_and_fallback(self):
        for parser in (
            ParsedFixture(
                [
                    DeltaMessage(content="Checking."),
                    DeltaMessage(tool_calls=[tool()]),
                ]
            ),
            None,
        ):
            context = ParsableContext(
                response_messages=[],
                tokenizer=None,
                parser_cls=None,
                request=ResponsesRequest(model="phase-fixture", input="fixture"),
                available_tools=None,
                chat_template=None,
                chat_template_content_format="auto",
                response_parser=parser,
                enable_auto_tools=True,
            )
            context.append_output(
                RequestOutput(
                    request_id="fixture",
                    prompt="fixture",
                    prompt_token_ids=[0],
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="Checking.",
                            token_ids=[1],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason="stop",
                        )
                    ],
                    finished=True,
                )
            )
            message = next(i for i in context.response_messages if i.type == "message")
            self.assertEqual(message.phase, None)

    def test_nonstream_without_parser_keeps_phase_unset(self):
        for reason, expected in [("stop", "final_answer"), ("length", None)]:
            output = CompletionOutput(
                index=0,
                text="answer",
                token_ids=[1],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason=reason,
            )
            items = serving()._make_response_output_items(
                ResponsesRequest(model="phase-fixture", input="fixture"),
                output,
                None,
            )
            self.assertEqual(items[0].phase, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)


@pytest.mark.parametrize(
    "profile",
    [
        "mimo",
        pytest.param(
            "glm",
        ),
    ],
)
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_real_selected_parser_responses_lifecycle(profile, finish_reason):
    import json

    from tests.parser.engine.replay_harness import load_local_model_parser
    from vllm.entrypoints.openai.responses.utils import construct_input_messages

    tokenizer, parser_cls = load_local_model_parser(profile)
    request = ResponsesRequest(
        model="test",
        input="hi",
        store=False,
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
    value = "\n  quoted <think>value</think>\n"
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

    async def run(split):
        server = serving()
        server.tool_server = None
        context = SimpleContext(response_parser=parser_cls(tokenizer, request.tools))

        async def outputs():
            previous = ""
            for start, end in [(0, split), (split, len(ids)), (len(ids), len(ids))]:
                decoded = tokenizer.decode(ids[:end]).rstrip("\ufffd")
                finish = finish_reason if start == end == len(ids) else None
                context.append_output(
                    RequestOutput(
                        request_id=request.request_id,
                        prompt="hi",
                        prompt_token_ids=tokenizer.encode(
                            "<think>", add_special_tokens=False
                        ),
                        prompt_logprobs=None,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text=decoded[len(previous) :],
                                token_ids=ids[start:end],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason=finish,
                            )
                        ],
                        finished=finish is not None,
                    )
                )
                previous = decoded
                yield context

        return [
            event
            async for event in server.responses_stream_generator(
                request,
                SamplingParams(max_tokens=128),
                outputs(),
                context,
                "test",
                tokenizer,
                RequestResponseMetadata(request_id=request.request_id),
                created_time=1,
            )
        ]

    for split in range(len(ids) + 1):
        events = asyncio.run(run(split))
        added = [e.item for e in events if e.type == "response.output_item.added"]
        done = [e.item for e in events if e.type == "response.output_item.done"]
        terminal = events[-1].response
        assert [item.id for item in added] == [item.id for item in done]
        assert len({item.id for item in done}) == len(done)
        assert [item.model_dump() for item in terminal.output] == [
            item.model_dump() for item in done
        ]
        assert [e.sequence_number for e in events] == list(range(len(events)))
        assert terminal.status == (
            "completed" if finish_reason == "stop" else "incomplete"
        )
        assert (
            "".join(e.delta for e in events if e.type == "response.output_text.delta")
            == "visible"
        )
        assert (
            "".join(
                e.delta for e in events if e.type == "response.reasoning_text.delta"
            )
            == "firstsecond"
        )
        calls = [item for item in terminal.output if item.type == "function_call"]
        assert len(calls) == 2 and calls[0].call_id != calls[1].call_id
        for call in calls:
            opened = next(item for item in added if item.id == call.id)
            assert opened.call_id == call.call_id
            args = "".join(
                e.delta
                for e in events
                if e.type == "response.function_call_arguments.delta"
                and e.item_id == call.id
            )
            assert args == call.arguments
            assert json.loads(args) == {"text": value}
        messages = construct_input_messages(
            request_input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "ok",
                }
                for call in calls
            ],
            prev_response_output=terminal.output,
        )
        assistant = next(message for message in messages if "tool_calls" in message)
        assert [call["id"] for call in assistant["tool_calls"]] == [
            call.call_id for call in calls
        ]
        assert [
            message["tool_call_id"] for message in messages if message["role"] == "tool"
        ] == [call.call_id for call in calls]


@pytest.mark.parametrize("profile", ["glm", "mimo"])
@pytest.mark.parametrize("include_reasoning", [True, False])
def test_real_parsable_stream_keeps_call_ids_and_resets_each_tool_round(
    profile, include_reasoning
):
    """Actual parser -> serving events -> builtin results -> fresh next-turn parser."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tests.parser.engine.replay_harness import load_local_model_parser

    tokenizer, parser_cls = load_local_model_parser(profile)
    request = ResponsesRequest(
        model="test",
        input="hi",
        stream=True,
        store=False,
        include_reasoning=include_reasoning,
        tools=[
            {
                "type": "function",
                "name": "code_interpreter",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                },
            }
        ],
    )

    def factory():
        return parser_cls(tokenizer, request.tools)

    ctx = ParsableContext(
        response_messages=[],
        tokenizer=tokenizer,
        parser_cls=parser_cls,
        response_parser=factory(),
        parser_factory=factory,
        request=request,
        available_tools=["python"],
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
    )
    session = SimpleNamespace(
        call_tool=AsyncMock(
            return_value=SimpleNamespace(content=[SimpleNamespace(text="ok")])
        )
    )
    ctx._tool_sessions = {"python": session}
    if profile == "glm":
        call = (
            "<tool_call>code_interpreter<arg_key>code</arg_key>"
            "<arg_value>pass</arg_value></tool_call>"
        )
        prompt = "<|assistant|><think>"
    else:
        call = (
            "<tool_call><function=code_interpreter><parameter=code>"
            "pass</parameter></function></tool_call>"
        )
        prompt = "<|im_start|>assistant\n<think>"
    turns = ["first</think>working" + call + call, "second</think>answer"]
    observed_ids: list[str] = []

    async def outputs():
        for turn, text in enumerate(turns):
            ids = tokenizer.encode(text, add_special_tokens=False)
            previous = ""
            for index in range(len(ids)):
                decoded = tokenizer.decode(ids[: index + 1]).rstrip("\ufffd")
                final = index == len(ids) - 1
                ctx.append_output(
                    RequestOutput(
                        request_id=f"fixture_{turn}",
                        prompt=prompt,
                        prompt_token_ids=tokenizer.encode(
                            prompt, add_special_tokens=False
                        ),
                        prompt_logprobs=None,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text=decoded[len(previous) :],
                                token_ids=[ids[index]],
                                cumulative_logprob=None,
                                logprobs=None,
                                finish_reason="stop" if final else None,
                            )
                        ],
                        finished=final,
                    )
                )
                previous = decoded
                yield ctx
            if turn == 0:
                calls = [m for m in ctx.response_messages if m.type == "function_call"]
                assert len(calls) == 2
                observed_ids.extend(call.call_id for call in calls)
                result = await ctx.call_tool()
                assert [item.call_id for item in result] == observed_ids
                ctx.append_tool_output(result)
                yield ctx
                assert not ctx.need_builtin_tool_call()
                ctx.start_new_turn()

    async def run():
        return [
            event
            async for event in serving()._process_simple_streaming_events(
                request,
                SamplingParams(max_tokens=128),
                outputs(),
                ctx,
                "test",
                tokenizer,
                RequestResponseMetadata(request_id=request.request_id),
                1,
                lambda event: event,
            )
        ]

    events = asyncio.run(run())
    done = [e.item for e in events if e.type == "response.output_item.done"]
    completed = [m for m in done if m.type == "mcp_call"]
    assert [m.id for m in completed] == [f"mcp_{call_id}" for call_id in observed_ids]
    assert [m.output for m in completed] == ["ok", "ok"]
    added = [
        e.item
        for e in events
        if e.type == "response.output_item.added" and e.item.type == "mcp_call"
    ]
    assert [m.id for m in added] == [m.id for m in completed]
    assert len(set(observed_ids)) == 2
    visible = [m for m in done if m.type == "message"]
    assert [m.content[0].text for m in visible] == ["working", "answer"]
    assert [m.phase for m in visible] == [None, None]
    assert session.call_tool.await_count == 2
    assert ctx.num_reasoning_tokens == 2
    assert [
        m.call_id for m in ctx.response_messages if m.type == "function_call"
    ] == observed_ids
    assert [
        m.model_dump() for m in ctx.make_response_output_items() if m.type == "mcp_call"
    ] == [m.model_dump() for m in completed]

    assert len([m for m in ctx.response_messages if m.type == "reasoning"]) == 2
    assert len([m for m in done if m.type == "reasoning"]) == (
        2 if include_reasoning else 0
    )
    assert request.include_reasoning is include_reasoning
    assert [
        e.output_index for e in events if e.type == "response.output_item.added"
    ] == list(range(len(done)))


@pytest.mark.parametrize("with_tool", [False, True])
def test_parsable_stored_history_renders_generated_turns_once(with_tool):
    from types import SimpleNamespace

    from openai.types.responses.response_function_tool_call_output_item import (
        ResponseFunctionToolCallOutputItem,
    )

    from vllm.entrypoints.openai.responses.utils import construct_input_messages

    original = [{"role": "user", "content": "original prompt"}]
    request = ResponsesRequest(model="test", input="original prompt", store=True)
    parser = ParsedFixture(
        [
            DeltaMessage(
                reasoning="private",
                content="Working.",
                tool_calls=[tool(name="python")] if with_tool else [],
            ),
        ]
    )
    ctx = ParsableContext(
        response_messages=original,
        tokenizer=None,
        parser_cls=None,
        response_parser=parser,
        request=request,
        available_tools=[],
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
    )

    def append():
        ctx.append_output(
            RequestOutput(
                request_id=request.request_id,
                prompt="prompt",
                prompt_token_ids=[0],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="fixture",
                        token_ids=[1],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="stop",
                    )
                ],
                finished=True,
            )
        )

    server = serving()
    server.response_store = {}
    server.msg_store = {request.request_id: original}
    server.tool_server = None
    append()
    if with_tool:
        ctx.append_tool_output(
            [
                ResponseFunctionToolCallOutputItem(
                    id="fco_test",
                    call_id="call_fixture_0",
                    type="function_call_output",
                    output="tool result",
                    status="completed",
                )
            ]
        )
        ctx.response_parser = ParsedFixture([DeltaMessage(content="Done.")])
        append()

    async def render(req, **kwargs):
        return construct_input_messages(
            request_input=req.input,
            prev_msg=kwargs["previous_messages"],
            prev_response_output=kwargs["previous_response_outputs"],
        )

    async def empty():
        if False:
            yield ctx

    async def run():
        server.response_store_lock = asyncio.Lock()
        server.online_renderer = SimpleNamespace(render_responses=render)
        response = await server.responses_full_generator(
            request,
            SamplingParams(max_tokens=32),
            empty(),
            ctx,
            "test",
            None,
            RequestResponseMetadata(request_id=request.request_id),
        )
        followup = ResponsesRequest(
            model="test",
            input="next prompt",
            previous_response_id=response.id,
        )
        first = await server._render_resolved_response_inputs(followup, response)
        second = await server._render_resolved_response_inputs(followup, response)
        return first, second

    first, second = asyncio.run(run())
    assert first == second
    assert original == [{"role": "user", "content": "original prompt"}]
    assert [m["content"] for m in first if m["role"] == "assistant"] == (
        ["Working.", "Done."] if with_tool else ["Working."]
    )
    assert all("reasoning" not in m for m in first)
    if with_tool:
        assert [m["content"] for m in first if m["role"] == "tool"] == ["tool result"]
        calls = [c for m in first for c in m.get("tool_calls", [])]
        assert [c["id"] for c in calls] == ["call_fixture_0"]


def test_parsable_hidden_reasoning_remains_available_for_builtin_continuation():
    from vllm.entrypoints.openai.responses.utils import (
        construct_chat_messages_with_tool_call,
    )

    request = ResponsesRequest(
        model="test",
        input="prompt",
        include_reasoning=False,
        store=False,
    )
    ctx = ParsableContext(
        response_messages=[],
        tokenizer=None,
        parser_cls=None,
        response_parser=ParsedFixture(
            [
                DeltaMessage(
                    reasoning="internal plan",
                    content="Working.",
                    tool_calls=[tool(name="python")],
                ),
            ]
        ),
        request=request,
        available_tools=[],
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
    )
    ctx.append_output(
        RequestOutput(
            request_id=request.request_id,
            prompt="prompt",
            prompt_token_ids=[0],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="fixture",
                    token_ids=[1],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
        )
    )
    chat = construct_chat_messages_with_tool_call(ctx.response_messages)
    assert chat[0]["reasoning"] == "internal plan"
    assert all(item.type != "reasoning" for item in ctx.make_response_output_items())
    assert request.include_reasoning is False


@pytest.mark.parametrize("finish_reason", ["error", "abort"])
def test_failed_generation_does_not_dispatch_pending_builtin(finish_reason):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    request = ResponsesRequest(model="test", input="prompt", store=False)
    ctx = ParsableContext(
        response_messages=[],
        tokenizer=None,
        parser_cls=None,
        response_parser=ParsedFixture([DeltaMessage(tool_calls=[tool(name="python")])]),
        request=request,
        available_tools=["python"],
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
    )
    ctx.call_tool = AsyncMock(
        side_effect=AssertionError("dispatched failed generation")
    )

    async def generate(*args, **kwargs):
        yield RequestOutput(
            request_id=request.request_id,
            prompt="prompt",
            prompt_token_ids=[0],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="fixture",
                    token_ids=[1],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            finished=True,
        )

    server = serving()
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.engine_client = SimpleNamespace(generate=generate)
    server._log_inputs = lambda *args, **kwargs: None

    async def run():
        return [
            item
            async for item in server._generate_with_builtin_tools(
                request.request_id,
                {"prompt_token_ids": [0]},
                SamplingParams(max_tokens=32),
                ctx,
            )
        ]

    if finish_reason == "error":
        with pytest.raises(GenerationError):
            asyncio.run(run())
    else:
        assert len(asyncio.run(run())) == 1
    ctx.call_tool.assert_not_called()


@pytest.mark.parametrize("finish_reason", ["error", "abort"])
def test_parsable_final_status_preserves_failure(finish_reason):
    request = ResponsesRequest(model="test", input="prompt", store=False)
    ctx = ParsableContext(
        response_messages=[],
        tokenizer=None,
        parser_cls=None,
        response_parser=ParsedFixture([DeltaMessage(content="partial")]),
        request=request,
        available_tools=[],
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
    )
    ctx.append_output(
        RequestOutput(
            request_id=request.request_id,
            prompt="prompt",
            prompt_token_ids=[0],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="fixture",
                    token_ids=[1],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            finished=True,
        )
    )

    async def empty():
        if False:
            yield ctx

    async def run():
        return await serving().responses_full_generator(
            request,
            SamplingParams(max_tokens=32),
            empty(),
            ctx,
            "test",
            None,
            RequestResponseMetadata(request_id=request.request_id),
        )

    if finish_reason == "error":
        with pytest.raises(GenerationError):
            asyncio.run(run())
    else:
        assert asyncio.run(run()).status == "cancelled"
