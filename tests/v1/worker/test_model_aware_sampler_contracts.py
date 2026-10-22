# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for production sampling state and warmup orchestration.

Extract exact production definitions to avoid CUDA-only module imports. Kernel
arithmetic and graph execution require separate GPU qualification.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = (
    Path(os.environ["SAMPLER_SOURCE_ROOT"])
    if "SAMPLER_SOURCE_ROOT" in os.environ
    else Path(__file__).resolve().parents[3]
)


def definition(path, name, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    ns = {"np": np, "torch": torch, **(namespace or {})}
    exec("from __future__ import annotations", ns)
    exec(
        compile(
            ast.fix_missing_locations(module),
            path,
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        ns,
    )
    return ns[name]


STATES = "vllm/v1/worker/gpu/sample/states.py"


def states(k, p, logical=8):
    cls = definition(STATES, "SamplingStates")
    state = object.__new__(cls)
    state.vocab_size = logical
    state.top_k = SimpleNamespace(np=np.array(k), gpu=torch.tensor(k))
    state.top_p = SimpleNamespace(np=np.array(p), gpu=torch.tensor(p))
    return state


@pytest.mark.parametrize(
    "k,p,width,expected",
    [
        ([8], [0.9], 16, True),  # None tensor despite logical < physical width
        ([8], [1.0], 16, False),
        ([2], [0.9], 16, False),
        ([2, 8], [0.9, 0.9], 16, False),  # supplied k=8 is active at width 16
        ([2, 8], [0.9, 0.9], 8, True),
        ([2, 8], [0.9, 1.0], 8, False),
        ([8, 8], [0.9, 1.0], 16, True),
        ([8, 8], [1.0, 1.0], 8, False),
    ],
)
def test_hint_matches_actual_tensor_representation(k, p, width, expected):
    state = states(k, p)
    idx = torch.arange(len(k))
    actual_k, actual_p, hint = state.get_top_k_top_p(idx, idx.numpy(), width)
    oracle = actual_p is not None and (
        actual_k is None or bool(torch.any((actual_k >= width) & (actual_p < 1)))
    )
    assert hint == expected == oracle


def test_padded_vocab_negative_control():
    state = states([8], [0.9])
    idx = torch.tensor([0])
    k, p, hint = state.get_top_k_top_p(idx, idx.numpy(), 16)
    old_hint = bool(np.any((state.top_k.np >= 16) & (state.top_p.np < 1.0)))
    assert k is None and p is not None and hint and not old_hint


@pytest.mark.parametrize("disabled", [0, -1, 9, 8])
def test_disabled_top_k_normalizes_to_logical_vocab(disabled):
    cls = definition(
        STATES,
        "SamplingStates",
        {
            "NO_LOGPROBS": -1,
            "_NP_INT64_MIN": np.iinfo(np.int64).min,
            "_NP_INT64_MAX": np.iinfo(np.int64).max,
        },
    )
    state = object.__new__(cls)
    state.vocab_size = 8
    for name in ["temperature", "top_k", "top_p", "min_p", "seeds"]:
        setattr(state, name, SimpleNamespace(np=np.zeros(1)))
    state.seeds_set = np.zeros(1, dtype=bool)
    state.num_logprobs = np.zeros(1)
    state.add_request(
        0,
        SimpleNamespace(
            temperature=1.0,
            top_k=disabled,
            top_p=0.9,
            min_p=0.0,
            seed=-7,
            logprobs=None,
        ),
    )
    assert state.top_k.np[0] == 8
    assert state.seeds.np[0] == -7


@pytest.mark.parametrize(
    "has_sampler,adaptive,fail_at",
    [
        (False, False, None),
        (True, False, None),
        (True, True, None),
        (True, True, 0),
        (True, True, 1),
        (True, False, 1),
    ],
)
def test_warmup_order_and_exception_restoration(has_sampler, adaptive, fail_at):
    manager = object()
    rs = SimpleNamespace(enable_adaptive_verification=adaptive) if has_sampler else None
    runner = SimpleNamespace(adaptive_verification=manager, rejection_sampler=rs)
    calls = []

    def inner(m, execute, sample, greedy=False):
        assert m.adaptive_verification is None
        if rs:
            assert not rs.enable_adaptive_verification
        calls.append(greedy)
        if len(calls) - 1 == fail_at:
            raise RuntimeError("warmup fixture failure")

    fn = definition(
        "vllm/v1/worker/gpu/warmup.py", "warmup_kernels", {"_warmup_kernels": inner}
    )
    if fail_at is None:
        fn(runner, None, None)
    else:
        with pytest.raises(RuntimeError, match="fixture failure"):
            fn(runner, None, None)
    assert runner.adaptive_verification is manager
    assert rs is None or rs.enable_adaptive_verification == adaptive
    assert calls == ([False] if not has_sampler or fail_at == 0 else [False, True])


@pytest.mark.parametrize(
    "batch,width,sm", [(1, 128, 20), (1, 262144, 20), (64, 262144, 20), (128, 4096, 20)]
)
def test_split_count_power_of_two_within_workspace(batch, width, sm):
    ns = {"_SPLIT_MAX_SPLITS": 32, "_SPLIT_MIN_SLICE": 7168}
    fn = definition("vllm/v1/sample/ops/topk_topp_triton.py", "_topp_split_count", ns)
    actual = fn(batch, width, sm)
    assert 1 <= actual <= 32 and actual & (actual - 1) == 0
    if actual < 32:
        assert actual * 2 * batch > sm and actual * 2 * 7168 > width


@pytest.mark.parametrize(
    "enabled,temperatures,expected",
    [
        ([False, False], [1.0, 1.0], [2, 3]),
        ([True, False], [1.0, 1.0], [7, 3]),
        ([True, True], [0.0, 1.0], [2, 7]),
        ([True, True], [1.0, 1.0], [7, 7]),
    ],
)
def test_draft_watermark_forwards_fallback_mask_and_updates_context(
    enabled, temperatures, expected
):
    cls = definition(
        "vllm/v1/watermarking/spec_decode.py",
        "DraftWatermarker",
        {"RandomSampler": SimpleNamespace},
    )
    obj = object.__new__(cls)
    obj.enabled = torch.tensor(enabled)
    obj.contexts = torch.tensor([[1, 2], [3, 4]])
    obj.deduplicate_contexts = "none"
    logits = torch.ones(2, 8)
    cache = torch.empty(2, 1, 8)
    steps = torch.zeros(2, dtype=torch.int64)
    temperature = torch.tensor(temperatures)
    positions = torch.tensor([3, 4])
    seeds = torch.tensor([7, 8])

    def sample(processed, contexts, *, random_sampler, skip_mask):
        assert torch.equal(skip_mask, ~(obj.enabled & (temperature != 0)))
        assert random_sampler.logits_cache is cache
        assert random_sampler.logits_cache_source is logits
        assert random_sampler.seeds is seeds
        assert random_sampler.positions is positions
        assert torch.equal(random_sampler.logits_cache_col, steps)
        assert contexts.data_ptr() == obj.contexts.data_ptr()
        torch.testing.assert_close(processed, logits)
        return SimpleNamespace(
            token_ids=torch.where(skip_mask, torch.tensor([2, 3]), 7)
        )

    obj.watermarker = SimpleNamespace(sample=sample)
    result = obj.sample(
        logits=logits,
        idx_mapping=torch.tensor([0, 1]),
        temperature=temperature,
        seed=seeds,
        pos=positions,
        apply_temperature=True,
        is_drafting=True,
        logits_cache=cache,
        logits_cache_col=steps,
    )
    assert torch.equal(result, torch.tensor(expected))
    assert torch.equal(obj.contexts[:, -1], result)


def test_base_and_watermark_super_forward_hint():
    recorded = []

    def filtering(logits, k, p, has_topp_only_rows=True):
        recorded.append(has_topp_only_rows)
        return logits

    def sample(*args, **kwargs):
        return torch.tensor([2])

    base = definition(
        "vllm/v1/worker/gpu/sample/sampler.py",
        "Sampler",
        {"apply_top_k_top_p": filtering, "gumbel_sample": sample},
    )
    cls = definition(
        "vllm/v1/watermarking/gpu_sampler.py", "GPUWatermarkSampler", {"Sampler": base}
    )
    obj = object.__new__(cls)
    obj.watermarking = SimpleNamespace(np=np.array([False]))
    obj.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array([1.0]), gpu=torch.tensor([1.0])),
        seeds=SimpleNamespace(gpu=torch.tensor([7])),
    )
    obj.use_fp64_gumbel = False
    result = obj._sample_random(
        torch.zeros(1, 8),
        torch.tensor([0]),
        np.array([0]),
        torch.tensor([1]),
        None,
        torch.tensor([0.9]),
        True,
        False,
    )
    assert torch.equal(result[0], torch.tensor([2]))
    assert recorded == [True]


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("seed", [-7, 0, 2**62])
def test_rejection_only_mixes_block_verification_seed(block, seed):
    tree = ast.parse(
        (ROOT / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py").read_text()
    )
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "step_seed" for t in n.targets)
    )
    expression = compile(
        ast.Expression(assignment.value), "production_step_seed", "eval"
    )
    values = [
        eval(
            expression,
            {
                "seed": seed,
                "i": i,
                "USE_BLOCK_VERIFICATION": block,
                "spec_step_seed": lambda s, step: s ^ (step << 32),
            },
        )
        for i in range(3)
    ]
    assert values[0] == seed
    assert len(set(values)) == (3 if block else 1)


def test_grammar_mask_does_not_disable_top_p_only_hint():
    recorded = []

    def filter_rows(logits, k, p, has_topp_only_rows):
        recorded.append((logits, k, p, has_topp_only_rows))
        return logits

    cls = definition(STATES, "SamplingStates", {"apply_top_k_top_p": filter_rows})
    obj = object.__new__(cls)
    obj.vocab_size = 8
    obj.top_k = SimpleNamespace(np=np.array([8]), gpu=torch.tensor([8]))
    obj.top_p = SimpleNamespace(np=np.array([0.9]), gpu=torch.tensor([0.9]))
    logits = torch.full((1, 16), float("-inf"))
    logits[0, :2] = torch.tensor([0.0, 1.0])
    result = obj.apply_top_k_top_p(logits, torch.tensor([0]), np.array([0]))
    assert result is logits and recorded[0][0] is logits
    assert recorded[0][1] is None and recorded[0][3] is True


@pytest.mark.parametrize(
    "batch,width,sm", [(1, 128, 20), (1, 262144, 20), (64, 262144, 20)]
)
def test_three_split_dispatchers_share_shape_and_workspace_bound(batch, width, sm):
    path = "vllm/v1/sample/ops/topk_topp_triton.py"
    count = definition(
        path, "_topp_split_count", {"_SPLIT_MAX_SPLITS": 32, "_SPLIT_MIN_SLICE": 7168}
    )
    ns = {"_topp_split_count": count, "_SPLIT_FANOUT": 8, "_SPLIT_ROUNDS": 5}
    logits = SimpleNamespace(shape=(batch, width))
    functions = [
        definition(path, name, ns)
        for name in ["_topp_split_stats", "_topp_split_step", "_topp_split_mask"]
    ]
    results = [
        functions[0](logits, None, None, None, sm),
        functions[1](logits, None, None, None, None, 0, sm),
        functions[2](logits, None, None, None, None, float("-inf"), sm),
    ]
    expected = count(batch, width, sm)
    assert all(
        grid == (batch * expected,)
        and kwargs["S"] == expected
        and kwargs["VOCAB_SIZE"] == width
        for grid, kwargs in results
    )
    assert expected <= 32
