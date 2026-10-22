# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE grouped topk kernel.

Run `pytest tests/kernels/moe/test_grouped_topk.py`.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk,
    GroupedTopKRouter,
    fused_grouped_topk,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _run_single_group_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_grouped_topk(
        hidden_states=torch.empty(
            (logits.shape[0], 0), dtype=logits.dtype, device=logits.device
        ),
        gating_output=logits,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=bias,
        num_expert_group=1,
        topk_group=1,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
    )


def _single_group_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "sigmoid":
        # Independent logistic oracle, rounded only at the FP32 score boundary.
        scores = logits.double().sigmoid().float()
    else:
        scores = torch.softmax(logits, dim=-1).float()
    indices = torch.argsort(
        scores + bias.float(), dim=-1, descending=True, stable=True
    )[:, :topk]
    values = scores.gather(1, indices)
    if renormalize:
        total = values.double().sum(dim=-1, keepdim=True)
        if scoring_func == "softmax":
            total += 1e-20
        else:
            total = torch.where(total == 0, torch.ones_like(total), total)
        values = (values.double() / total).float()
    values *= routed_scaling_factor
    return values, indices.to(torch.int32)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("skip_padding", [False, True])
def test_kimi_grouped_topk_masks_padding_when_enabled(skip_padding: bool):
    hidden_states = torch.randn((2, 16), device="cuda")
    gating_output = torch.randn((2, 896), device="cuda")
    is_padding = torch.tensor([False, True], device="cuda")
    router = GroupedTopKRouter(
        top_k=16,
        global_num_experts=896,
        num_expert_group=1,
        topk_group=1,
        skip_padding=skip_padding,
    )

    with set_forward_context(None, VllmConfig(), is_padding=is_padding):
        topk_weights, topk_ids = router._compute_routing(
            hidden_states, gating_output, torch.int32
        )

    assert torch.all(topk_ids[0] >= 0)
    if skip_padding:
        assert torch.all(topk_ids[1] == -1)
        assert torch.all(topk_weights[1] == 0)
    else:
        assert torch.all(topk_ids[1] >= 0)
        assert torch.all(topk_weights[1] > 0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("n_token", [1, 33, 64])
@pytest.mark.parametrize("n_hidden", [1024, 2048])
@pytest.mark.parametrize(
    "n_expert,topk,num_expert_group,topk_group",
    [
        (16, 2, 8, 2),
        (128, 2, 8, 2),
        (256, 8, 8, 4),
        (384, 8, 1, 1),
        (512, 22, 1, 1),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float32])
def test_grouped_topk(
    monkeypatch: pytest.MonkeyPatch,
    n_token: int,
    n_hidden: int,
    n_expert: int,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype,
):
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all", "+grouped_topk"])
    )
    get_cached_compilation_config.cache_clear()

    set_random_seed(0)
    hidden_states = torch.randn((n_token, n_hidden), dtype=input_dtype, device="cuda")
    gating_output = torch.randn((n_token, n_expert), dtype=input_dtype, device="cuda")
    e_score_correction_bias = torch.randn((n_expert,), dtype=bias_dtype, device="cuda")

    with set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        m.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
        m.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        grouped_topk = GroupedTopk(
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )
        assert grouped_topk._forward_method.__name__ == "forward_cuda"
        baseline_topk_weights, baseline_topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            e_score_correction_bias=e_score_correction_bias,
        )

        test_topk_weights, test_topk_ids = fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        torch.testing.assert_close(
            baseline_topk_weights, test_topk_weights, atol=2e-2, rtol=0
        )
        torch.testing.assert_close(baseline_topk_ids, test_topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
def test_grouped_topk_single_group_large_batch():
    set_random_seed(0)
    logits = torch.randn((1536, 896), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((896,), dtype=torch.float32, device="cuda")

    expected_values, expected_ids = _single_group_reference(
        logits, bias, 16, scoring_func="sigmoid", renormalize=True
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits, bias, 16, scoring_func="sigmoid", renormalize=True
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,topk,input_dtype,bias_dtype",
    [
        (512, 9, torch.bfloat16, torch.float32),
        (512, 16, torch.float16, torch.float16),
        (513, 9, torch.float32, torch.bfloat16),
        (513, 16, torch.bfloat16, torch.float32),
        (895, 9, torch.float16, torch.bfloat16),
        (896, 16, torch.float32, torch.float16),
        (897, 9, torch.bfloat16, torch.bfloat16),
        (897, 16, torch.float16, torch.float32),
        (1024, 9, torch.float32, torch.bfloat16),
        (1024, 16, torch.bfloat16, torch.float16),
    ],
)
@pytest.mark.parametrize(
    "scoring_func,renormalize,routed_scaling_factor",
    [
        ("sigmoid", True, 1.0),
        ("sigmoid", False, 2.5),
        ("softmax", True, 2.5),
        ("softmax", False, 1.0),
    ],
)
def test_grouped_topk_single_group_tiers(
    num_experts: int,
    topk: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
):
    set_random_seed(7)
    logits = torch.randn((17, num_experts), dtype=input_dtype, device="cuda")
    bias = torch.randn((num_experts,), dtype=bias_dtype, device="cuda")

    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,topk,scoring_func",
    [
        (128, 8, "sigmoid"),
        (129, 8, "sigmoid"),
        (257, 8, "sigmoid"),
        (385, 8, "sigmoid"),
        (512, 9, "sigmoid"),
        (513, 9, "sigmoid"),
        (769, 9, "sigmoid"),
        (897, 16, "sigmoid"),
        (1024, 16, "sigmoid"),
        (128, 4, "softmax"),
        (128, 5, "softmax"),
        (129, 8, "softmax"),
        (161, 8, "softmax"),
        (256, 9, "softmax"),
        (257, 8, "softmax"),
        (512, 9, "softmax"),
        (512, 17, "softmax"),
        (512, 23, "softmax"),
        (513, 8, "softmax"),
        (577, 9, "softmax"),
        (769, 9, "softmax"),
        (897, 9, "softmax"),
        (1024, 16, "softmax"),
    ],
)
def test_grouped_topk_single_group_capacity_tiers(
    num_experts: int,
    topk: int,
    scoring_func: str,
):
    set_random_seed(11)
    logits = torch.randn((3, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [512, 896, 1024])
def test_grouped_topk_single_group_stable_ties(num_experts: int):
    logits = torch.zeros((1, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")

    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
    expected_values = torch.full((1, 16), 2.5 / 16, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [512, 896, 1024])
@pytest.mark.parametrize("num_finite", [0, 15])
@pytest.mark.parametrize("renormalize", [False, True])
def test_grouped_topk_single_group_nonfinite_scores(
    num_experts: int, num_finite: int, renormalize: bool
):
    logits = torch.full(
        (1, num_experts), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    if num_finite:
        logits[0, :num_finite] = torch.arange(
            num_finite, dtype=torch.bfloat16, device="cuda"
        )
    logits[0, num_finite] = torch.inf
    logits[0, num_finite + 1] = -torch.inf
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")

    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=2.5,
    )

    if num_finite == 0:
        expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
        if renormalize:
            expected_values = torch.full(
                (1, 16), 1 / 16, dtype=torch.float32, device="cuda"
            )
        else:
            expected_values = torch.zeros((1, 16), dtype=torch.float32, device="cuda")
    else:
        expected_ids = torch.cat(
            (
                torch.arange(num_finite - 1, -1, -1, dtype=torch.int32, device="cuda"),
                torch.tensor([num_finite], dtype=torch.int32, device="cuda"),
            )
        )[None]
        finite_values = logits[0, :num_finite].float().sigmoid().flip(0)
        if renormalize:
            finite_values /= finite_values.sum()
        finite_values *= 2.5
        expected_values = torch.cat(
            (finite_values, torch.zeros(1, dtype=torch.float32, device="cuda"))
        )[None]

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


# Observed top-eight correction biases, not a full checkpoint tensor. Other
# experts use a lower synthetic bias. This preserves the real selected IDs and
# rounded-addition witness without requiring private model artifacts in CI.
_MODEL_ROUTER_CASES = [
    pytest.param(
        256,
        1.0,
        [91, 137, 252, 221, 156, 62, 204, 19],
        [
            2.183300733566284,
            2.1645851135253906,
            2.115870714187622,
            2.0943751335144043,
            2.0889148712158203,
            2.0862226486206055,
            2.080620765686035,
            2.0785624980926514,
        ],
        0.0,
        id="mimo-256-top8",
    ),
    pytest.param(
        288,
        2.5,
        [47, 153, 140, 19, 142, 197, 134, 205],
        [
            7.786213397979736,
            7.785406589508057,
            7.78521203994751,
            7.7845025062561035,
            7.783807277679443,
            7.783804416656494,
            7.783605575561523,
            7.783412456512451,
        ],
        7.416070938110352,
        id="glm-288-top8-scale2.5",
    ),
]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize(
    "num_experts,scale,selected_ids,selected_biases,other_bias", _MODEL_ROUTER_CASES
)
@pytest.mark.parametrize("num_rows", [1, 8, 1024, 1025, 16384])
def test_model_grouped_topk_preserves_scores_before_bias(
    num_experts, scale, selected_ids, selected_biases, other_bias, num_rows
):
    """G03/F01: actual top8 block/warp tiers retain scores erased by bias addition.

    Equal -17 logits keep this independent of G04: even the old tanh score is
    positive here, and retaining it yields exactly the same normalized weights.
    16384 is the inspected GLM batched-token/prefill capacity, not a context cap.
    """
    logits = torch.full(
        (num_rows, num_experts), -17.0, dtype=torch.float32, device="cuda"
    )
    bias = torch.full((num_experts,), other_bias, dtype=torch.float32, device="cuda")
    bias[selected_ids] = torch.tensor(
        selected_biases, dtype=torch.float32, device="cuda"
    )
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        8,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=scale,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        8,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=scale,
    )
    torch.testing.assert_close(
        expected_ids[0], torch.tensor(selected_ids, dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=2e-6, atol=0)
    torch.testing.assert_close(
        actual_values.sum(-1), torch.full((num_rows,), scale, device="cuda")
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("num_experts,scale", [(256, 1.0), (288, 2.5)])
@pytest.mark.parametrize("num_rows", [1, 1024, 1025])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("tail_logit", [-20.0, -80.0])
def test_model_grouped_topk_sigmoid_negative_tail(
    num_experts, scale, num_rows, input_dtype, renormalize, tail_logit
):
    """G04/F02: zero bias isolates tanh loss and epsilon-dominated normalization."""
    logits = torch.full(
        (num_rows, num_experts), tail_logit - 20, dtype=input_dtype, device="cuda"
    )
    logits[:, :8] = tail_logit - torch.arange(8, device="cuda") * 0.25
    bias = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        8,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=scale,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        8,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=scale,
    )
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=2e-6, atol=0)
    if renormalize:
        torch.testing.assert_close(
            actual_values.sum(-1), torch.full((num_rows,), scale, device="cuda")
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("num_experts", [128, 256, 288, 512, 768])
@pytest.mark.parametrize("renormalize", [False, True])
def test_grouped_topk_softmax_preserves_precomputed_scores(num_experts, renormalize):
    """The modified warp also consumes original precomputed softmax scores."""
    logits = torch.zeros((1025, num_experts), dtype=torch.float32, device="cuda")
    logits[:, :8] = -17.0
    bias = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
    bias[:8] = 2.0
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        8,
        scoring_func="softmax",
        renormalize=renormalize,
        routed_scaling_factor=2.5,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        8,
        scoring_func="softmax",
        renormalize=renormalize,
        routed_scaling_factor=2.5,
    )
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=2e-6, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize(
    "num_experts,num_groups,topk_groups",
    [
        (256, 8, 4),  # small-expert grouped kernel
        (512, 8, 4),  # generic grouped kernel
        (1152, 1, 1),  # single-group fallback beyond the specialized tiers
    ],
)
@pytest.mark.parametrize("tail_logit", [-20.0, -80.0])
def test_grouped_topk_other_sigmoid_consumers_negative_tail(
    num_experts, num_groups, topk_groups, tail_logit
):
    """All generic consumers of the shared helper retain finite negative tails."""
    logits = torch.full(
        (1, num_experts), tail_logit, dtype=torch.float32, device="cuda"
    )
    bias = torch.arange(num_experts, device="cuda", dtype=torch.float32) / num_experts
    scores = logits.double().sigmoid().float()
    selection = scores + bias
    group_scores = selection.view(1, num_groups, -1).topk(2, dim=-1).values.sum(-1)
    groups = group_scores.argsort(descending=True, stable=True)[:, :topk_groups]
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
    selection = selection.masked_fill(
        ~mask.repeat_interleave(num_experts // num_groups, dim=1), -torch.inf
    )
    expected_ids = selection.argsort(descending=True, stable=True)[:, :8]
    expected_values = scores.gather(1, expected_ids)
    expected_values = 2.5 * expected_values / expected_values.sum(-1, keepdim=True)
    actual_values, actual_ids = fused_grouped_topk(
        hidden_states=logits[:, :0],
        gating_output=logits,
        topk=8,
        renormalize=True,
        e_score_correction_bias=bias,
        num_expert_group=num_groups,
        topk_group=topk_groups,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
    )
    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=2e-6, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("num_experts,scale", [(256, 1.0), (288, 2.5)])
@pytest.mark.parametrize("num_rows", [1, 1025])
@pytest.mark.parametrize(
    "case", ["ties", "invalid_logits", "invalid_bias", "all_invalid", "zero_scores"]
)
def test_model_grouped_topk_preserves_selection_edge_contracts(
    num_experts, scale, num_rows, case
):
    """Both model tiers keep stable ties and the existing finite-input masks."""
    logits = torch.zeros((num_rows, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
    first_valid = 0
    if case == "invalid_logits":
        logits[:, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf], device="cuda")
        first_valid = 3
    elif case == "invalid_bias":
        bias[:3] = torch.tensor([torch.nan, torch.inf, -torch.inf], device="cuda")
        first_valid = 3
    elif case == "all_invalid":
        logits.fill_(torch.nan)
    elif case == "zero_scores":
        logits.fill_(-1000.0)
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        8,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=scale,
    )
    expected_ids = torch.arange(
        first_valid, first_valid + 8, dtype=torch.int32, device="cuda"
    )[None].expand(num_rows, -1)
    # Preserve the existing all-invalid fallback, including its unscaled value.
    expected_weight = (1.0 if case == "all_invalid" else scale) / 8
    if case == "zero_scores":
        expected_weight = 0.0
    expected_values = torch.full((num_rows, 8), expected_weight, device="cuda")
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=2e-6, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize(
    "num_experts,scale,selected_ids,selected_biases,other_bias", _MODEL_ROUTER_CASES
)
def test_model_grouped_router_factory_reaches_original_score_contract(
    monkeypatch, num_experts, scale, selected_ids, selected_biases, other_bias
):
    """Exercise actual factory, router, wrapper and native op, not a substitute topk."""
    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "1")
    logits = torch.full((1025, num_experts), -20.0, dtype=torch.float32, device="cuda")
    bias = torch.full((num_experts,), other_bias, dtype=torch.float32, device="cuda")
    bias[selected_ids] = torch.tensor(
        selected_biases, dtype=torch.float32, device="cuda"
    )
    router = create_fused_moe_router(
        top_k=8,
        global_num_experts=num_experts,
        renormalize=True,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=scale,
        e_score_correction_bias=bias,
    )
    assert isinstance(router, GroupedTopKRouter)
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        8,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=scale,
    )
    actual_values, actual_ids = router.select_experts(
        hidden_states=logits[:, :0],
        router_logits=logits,
        topk_indices_dtype=torch.int64,
    )
    assert actual_ids.dtype == torch.int64
    assert actual_values.dtype == torch.float32
    torch.testing.assert_close(actual_ids, expected_ids.long(), rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=2e-6, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("num_experts", [513, 768])
@pytest.mark.parametrize("num_rows", [1024, 1025])
@pytest.mark.parametrize(
    "logit,selection_bias",
    [(-17.0, 2.0), (-20.0, 0.0), (-80.0, 0.0)],
    ids=["bias-cancellation", "sigmoid-tail", "tiny-normalizer"],
)
def test_grouped_topk_sigmoid_lane_owned_block_and_warp(
    num_experts, num_rows, logit, selection_bias
):
    """Tier768/top16 uses hierarchical block then lane-owned warp, not tuned896."""
    logits = torch.full(
        (num_rows, num_experts), logit, dtype=torch.float32, device="cuda"
    )
    bias = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
    bias[:16] = selection_bias
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=2e-6, atol=0)
