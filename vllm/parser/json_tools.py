# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSON-envelope helpers shared by model-local serving adapters."""

from __future__ import annotations

import json
from dataclasses import replace

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)


def _json_tool_config(config: ParserEngineConfig) -> ParserEngineConfig:
    transitions = {
        key: transition
        for key, transition in config.transitions.items()
        if key[0] in (ParserState.REASONING, ParserState.CONTENT)
        and transition.next_state in (ParserState.REASONING, ParserState.CONTENT)
    }
    for terminal in ("JSON_OBJECT_START", "JSON_ARRAY_START"):
        transitions[ParserState.CONTENT, terminal] = Transition(
            ParserState.TOOL_BETWEEN, (EventType.TEXT_CHUNK,)
        )
    # Once a required/named JSON body starts, all its bytes are arguments,
    # including quoted spellings (or token IDs) of reasoning delimiters.
    return replace(
        config,
        terminals={
            **{
                key: value
                for key, value in config.terminals.items()
                if key in ("THINK_START", "THINK_END")
            },
            "JSON_OBJECT_START": "{",
            "JSON_ARRAY_START": "[",
        },
        token_id_terminals={
            key: value
            for key, value in config.token_id_terminals.items()
            if key in ("THINK_START", "THINK_END")
        },
        transitions=transitions,
        content_events={
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.TOOL_BETWEEN: EventType.TEXT_CHUNK,
        },
        arg_converter=None,
        arg_value_extender=None,
    )


class _RequiredJSONStream:
    """Scan required-tool envelopes once, retaining only un-emitted bytes.

    Parameters may precede the name. Only that case buffers argument bytes
    across chunks; completed calls and already emitted prefixes are released.
    JSON argument strings are opaque, including reasoning/tool delimiters.
    """

    def __init__(self) -> None:
        self.state = "array"
        self.index = -1
        self.name: str | None = None
        self.key = ""
        self.arguments_started = False
        self.emitted = False
        self.pending: list[str] = []
        self.capture: list[str] = []
        self.stack: list[str] = []
        self.quoted = False
        self.escaped = False
        self.scalar = False
        self.reading_key = False

    def _begin_value(self, char: str, *, key: bool = False) -> None:
        self.reading_key = key
        self.capture = []
        self.stack = [char] if char in "{[" else []
        self.quoted = char == '"'
        self.escaped = False
        self.scalar = char not in '{["'
        self.state = "value"
        if not key and self.key == "parameters":
            self.arguments_started = True
        self._append(char)

    def _append(self, char: str) -> None:
        if self.reading_key or self.key == "name":
            self.capture.append(char)
        elif self.key == "parameters":
            self.pending.append(char)

    def _finish_value(self) -> None:
        if self.reading_key or self.key == "name":
            try:
                value = json.loads("".join(self.capture))
            except (ValueError, TypeError):
                self.state = "invalid"
                return
            if not isinstance(value, str):
                self.state = "invalid"
                return
            if self.reading_key:
                self.key = value
            else:
                self.name = value
        self.capture.clear()
        self.state = "colon" if self.reading_key else "after_value"

    def _flush(self, updates: list[tuple[int, str, str, bool]]) -> None:
        if self.name is None or not self.arguments_started:
            return
        if self.pending or not self.emitted:
            updates.append(
                (self.index, self.name, "".join(self.pending), not self.emitted)
            )
            self.pending.clear()
            self.emitted = True

    def feed(self, text: str) -> list[tuple[int, str, str, bool]]:
        updates: list[tuple[int, str, str, bool]] = []
        for char in text:
            # A primitive ends before its delimiter; process that delimiter
            # again as envelope syntax. No earlier input is rescanned.
            while True:
                if self.state == "value":
                    if self.scalar and char in " \t\r\n,}]":
                        self._finish_value()
                        continue
                    self._append(char)
                    if self.escaped:
                        self.escaped = False
                    elif self.quoted:
                        if char == "\\":
                            self.escaped = True
                        elif char == '"':
                            self.quoted = False
                            if not self.stack:
                                self._finish_value()
                    elif char == '"':
                        self.quoted = True
                    elif char in "{[":
                        self.stack.append(char)
                    elif char in "}]":
                        if not self.stack or self.stack.pop() != (
                            "{" if char == "}" else "["
                        ):
                            self.state = "invalid"
                        elif not self.stack:
                            self._finish_value()
                    break

                if char in " \t\r\n":
                    break
                if self.state == "array" and char == "[":
                    self.state = "envelope"
                elif self.state == "envelope" and char == "{":
                    self.index += 1
                    self.name = None
                    self.key = ""
                    self.arguments_started = self.emitted = False
                    self.pending.clear()
                    self.state = "key"
                elif self.state == "envelope" and char == "]":
                    self.state = "done"
                elif self.state == "key" and char == '"':
                    self._begin_value(char, key=True)
                elif self.state == "colon" and char == ":":
                    self.state = "value_start"
                elif self.state == "value_start":
                    self._begin_value(char)
                elif self.state == "after_value" and char == ",":
                    self.state = "key"
                elif self.state in ("key", "after_value") and char == "}":
                    self._flush(updates)
                    self.state = "between"
                elif self.state == "between" and char == ",":
                    self.state = "envelope"
                elif self.state == "between" and char == "]":
                    self.state = "done"
                else:
                    self.state = "invalid"
                break
        self._flush(updates)
        return updates
