# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native decode-stream setup with explicit prompt/output ownership."""

from collections.abc import Sequence

import tokenizers


def common_prefix_length(left: str, right: str) -> int:
    """Character offset shared by two decoded prefixes, without text cleanup."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


class _PartialPromptDecodeStream:
    def __init__(self, stream, prompt_text: str):
        self._stream = stream
        self._prompt_text: str | None = prompt_text

    def step(self, tokenizer, token_id):
        text = self._stream.step(tokenizer, token_id)
        if text and self._prompt_text is not None:
            # tokenizers 0.23.3 deliberately re-emits the whole prefix when
            # native prefill ends in incomplete UTF-8. Those prompt characters
            # are not generated output. A completed cross-boundary glyph is.
            offset = common_prefix_length(self._prompt_text, text)
            self._prompt_text = None
            return text[offset:]
        return text


def create_decode_stream(
    tokenizer,
    prompt_token_ids: Sequence[int] | None = None,
    *,
    skip_special_tokens: bool = False,
):
    """Keep native prefill/decoding; account for ByteLevel partial prompts.

    A ByteLevel stream can end in incomplete UTF-8 only if its last non-skipped
    token also does. Probe that token first, so ordinary complete chat prompts
    do not incur another full-prompt decode. No token is removed from prefill.
    """
    ids = list(prompt_token_ids) if prompt_token_ids is not None else None
    # Resolve through the module, as upstream does, to preserve backend hooks.
    stream = tokenizers.decoders.DecodeStream(
        ids=ids, skip_special_tokens=skip_special_tokens
    )
    if ids and isinstance(tokenizer.decoder, tokenizers.decoders.ByteLevel):
        for token_id in reversed(ids):
            tail = tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens)
            if tail:
                if tail.endswith("\ufffd"):
                    prompt_text = tokenizer.decode(
                        ids, skip_special_tokens=skip_special_tokens
                    )
                    if prompt_text.endswith("\ufffd"):
                        return _PartialPromptDecodeStream(stream, prompt_text)
                break
    return stream
