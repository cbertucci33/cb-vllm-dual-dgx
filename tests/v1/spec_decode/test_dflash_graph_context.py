# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for context staging; these do not execute CUDA graphs."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator


def make_spec(hidden, aux, graph_rows=4):
    spec = object.__new__(DFlashSpeculator)
    spec.max_num_tokens = 12
    spec.num_speculative_steps = 1
    spec.hidden_states = torch.full((12, hidden), -9.0)
    spec._aux_staging = torch.zeros(graph_rows, hidden * aux) if aux else None
    spec._num_rejected = None
    spec.context_positions = torch.arange(12)
    spec._context_slot_mappings = torch.arange(24).view(2, 12).clone()
    spec._layer_group_idx = [1, 0, 1]
    spec.target_input_buffers = NS(query_start_loc=torch.tensor([0, 2, 4]))
    calls: list[tuple[object, ...]] = []

    def combine(x):
        calls.append(("combine", len(x)))
        return x.view(len(x), aux, hidden).sum(1)

    def store(h, pos, slots):
        calls.append(("kv", h.clone(), pos.clone(), [s.clone() for s in slots]))

    spec.model = NS(
        combine_hidden_states=combine, precompute_and_store_context_kv=store
    )
    return spec, calls


@pytest.mark.parametrize("hidden,aux", [(3, 2), (5, 3), (7, 0)])
@pytest.mark.parametrize("tokens", [2, 4, 7])
def test_full_staging_fixed_addresses_padding_and_overflow(hidden, aux, tokens):
    spec, calls = make_spec(hidden, aux)
    last = torch.arange(tokens * hidden).view(tokens, hidden).float()
    features = [last + i for i in range(aux)] or None
    expected = sum(features) if features else last
    batch = NS(num_reqs=1, num_tokens=tokens)
    staging_ptr = spec._aux_staging.data_ptr() if aux else None
    hidden_ptr = spec.hidden_states.data_ptr()
    spec._prepare_graph_replay(
        batch, 2, last, features, torch.zeros(1, dtype=torch.int32), False
    )
    if tokens > 4:
        assert calls[-1][0] == "kv"
        torch.testing.assert_close(calls[-1][1], expected[4:])
    else:
        assert not any(c[0] == "kv" for c in calls)
        assert (spec._context_slot_mappings[:, tokens:4] == -1).all()
    spec._prepare_graph_forward(2, None)
    torch.testing.assert_close(spec.hidden_states[:tokens], expected)
    assert spec.hidden_states.data_ptr() == hidden_ptr
    if aux:
        assert spec._aux_staging.data_ptr() == staging_ptr
    # Every layer keeps its own group slot mapping and positions.
    kv = calls[-1]
    for layer, group in enumerate([1, 0, 1]):
        torch.testing.assert_close(kv[3][layer], spec._context_slot_mappings[group, :4])
    # The next replay reads new values at the same staging address.
    features = [x + 10 for x in features] if features else None
    spec._prepare_graph_replay(
        batch, 2, last + 10, features, torch.zeros(1, dtype=torch.int32), False
    )
    spec._prepare_graph_forward(2, None)
    expected = sum(features) if features else last + 10
    torch.testing.assert_close(spec.hidden_states[:tokens], expected)


@pytest.mark.parametrize("aux", [0, 2])
def test_eager_profile_stores_all_features_without_graph_staging(aux):
    spec, calls = make_spec(3, aux)
    last = torch.arange(21).view(7, 3).float()
    features = [last + i for i in range(aux)] or None
    spec._store_hidden_states(7, last, features)
    torch.testing.assert_close(
        spec.hidden_states[:7], sum(features) if features else last
    )
    assert all(c[0] != "kv" for c in calls)


def test_dummy_replay_masks_all_context_writes():
    spec, calls = make_spec(3, 2)
    last = torch.ones(7, 3)
    spec._prepare_graph_replay(
        NS(num_reqs=1, num_tokens=7),
        2,
        last,
        [last, last],
        torch.zeros(1, dtype=torch.int32),
        True,
    )
    assert (spec._context_slot_mappings[:, :4] == -1).all()
    assert not any(c[0] == "kv" for c in calls)


def test_context_combine_anchor_kv_order_and_rejection_staging():
    spec, calls = make_spec(3, 2)
    spec._num_rejected = torch.zeros(2, dtype=torch.int32)
    last = torch.arange(21).view(7, 3).float()
    spec.prepare_context_anchor = lambda n, loc, rejected: calls.append(
        ("anchor", n, loc, rejected.clone())
    )
    spec._prepare_graph_replay(
        NS(num_reqs=1, num_tokens=7), 2, last, [last, last], torch.tensor([1]), False
    )
    calls.clear()
    spec._prepare_graph_forward(2, None)
    assert [c[0] for c in calls] == ["combine", "anchor", "kv"]
    assert calls[1][1] == 2
    assert calls[1][2] is spec.target_input_buffers.query_start_loc
    assert calls[1][3].tolist() == [1, 0]


def test_lilicorr_fixed_inputs_anchor_ignores_padded_rejections():
    spec = NS(
        hidden_states=torch.arange(24).view(8, 3).float(),
        model=NS(model=NS(hidden_norm=lambda x: 2 * x)),
        anchor_hidden=torch.full((2, 3), 99.0),
        anchor_valid=torch.ones(2, dtype=torch.bool),
    )
    loc = torch.tensor([0, 2, 2])
    rejected = torch.tensor([1, 5])
    LiLiCorrSpeculator.prepare_context_anchor(spec, 2, loc, rejected)
    torch.testing.assert_close(spec.anchor_hidden[0], 2 * spec.hidden_states[0])
    assert spec.anchor_valid.tolist() == [True, False]
    assert not spec.anchor_hidden[1].any()
    loc.copy_(torch.tensor([0, 7, 7]))
    rejected.zero_()
    LiLiCorrSpeculator.prepare_context_anchor(spec, 2, loc, rejected)
    torch.testing.assert_close(spec.anchor_hidden[0], 2 * spec.hidden_states[6])
    assert not spec.anchor_hidden[1].any()
