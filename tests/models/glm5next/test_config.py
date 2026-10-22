# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import Glm5NextTextConfig

from vllm.models.glm5next.common.model import _validate_supported_config


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "index_dsa_use_layernorm",
            False,
            "requires index_dsa_use_layernorm=True",
        ),
        ("index_kpool_compress", False, "requires index_kpool_compress=True"),
        (
            "index_kpool_always_select_tail",
            False,
            "requires index_kpool_always_select_tail=True",
        ),
        ("hres_vwnstyle", False, "requires hres_vwnstyle=True"),
        ("mhc_no_norm_weight", True, "requires mhc_no_norm_weight=False"),
    ],
)
def test_unsupported_glm_config_fails_closed(field, value, message):
    config = Glm5NextTextConfig(
        index_topk=2048,
        mhc=True,
        **{field: value},
    )

    with pytest.raises(NotImplementedError, match=message):
        _validate_supported_config(config)


def test_accepted_glm_config_capabilities_pass():
    config = Glm5NextTextConfig(
        index_topk=2048,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        mhc=True,
    )

    _validate_supported_config(config)
