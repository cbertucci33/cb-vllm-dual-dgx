# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from xgrammar.structural_tag import TriggeredTagsFormat

from vllm.parser.engine.registered_adapters import MiMoParserToolAdapter


class MiMoToolParser(MiMoParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "mimo"

    def get_structural_tag(self, request, *, reasoning=False):
        tag = super().get_structural_tag(request, reasoning=reasoning)
        if (
            tag is not None
            and request.parallel_tool_calls is False
            and isinstance(tag.format, TriggeredTagsFormat)
        ):
            tag.format.stop_after_first = True
        return tag
