# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # No speculative config, decode alongside prefill
    "regular_decode_with_prefill": GDNBuildTestCase(
        seq_lens=[40, 100],
        query_lens=[1, 50],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=1,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    mamba_cache_mode: str = "none",
    num_prefill_checkpoint_blocks: int = 0,
    prefix_match_unit: int | None = None,
    device: torch.device = DEVICE,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    vllm_config.cache_config.prefix_match_unit = prefix_match_unit
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        prefill_checkpoint_alignment=(
            16 if num_prefill_checkpoint_blocks > 0 else None
        ),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=device,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    block_table: torch.Tensor | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    if block_table is not None:
        common = common.replace(block_table_tensor=block_table)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


def _cpu_aligned_state_indices(
    block_table: torch.Tensor, seq_lens: torch.Tensor, spec: MambaSpec
) -> torch.Tensor:
    """Reference the CUDA align gather without launching Triton on CPU tensors."""
    num_slots = 1 + spec.num_speculative_blocks
    starts = ((seq_lens - 1) // spec.block_size).clamp_min(0)
    columns = starts[:, None] + torch.arange(num_slots, dtype=torch.int32)
    return block_table.gather(1, columns.to(torch.int64))


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
@pytest.mark.parametrize("full_cuda_graph", [False, True])
@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_update_block_table_matches_build(
    test_case: GDNBuildTestCase, full_cuda_graph: bool, mamba_cache_mode: str
):
    """Updating another group's metadata must match an independent build."""
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    src, dst, ref = (
        _create_gdn_builder(test_case.num_speculative_tokens, full_cuda_graph)
        for _ in range(3)
    )
    for builder in (src, dst, ref):
        builder.vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    common = create_common_attn_metadata(batch, BLOCK_SIZE, DEVICE)
    if mamba_cache_mode == "align":
        dst.mamba_aligned_state_indices = ref.mamba_aligned_state_indices = (
            _cpu_aligned_state_indices(
                common.block_table_tensor, common.seq_lens, ref.kv_cache_spec
            )
        )
    draft_tokens = test_case.num_decode_draft_tokens
    expected = _build(ref, batch, draft_tokens, common.block_table_tensor)
    source = _build(src, batch, draft_tokens)
    fields = (
        "spec_state_indices_tensor",
        "non_spec_state_indices_tensor",
        "prefill_state_indices",
    )
    source_indices = [getattr(source, field) for field in fields]
    source_indices = [
        tensor if tensor is None else tensor.clone() for tensor in source_indices
    ]

    meta = dst.update_block_table(
        source, common.block_table_tensor, common.slot_mapping
    )

    for field, source_index in zip(fields, source_indices):
        actual = getattr(meta, field)
        torch.testing.assert_close(actual, getattr(expected, field))
        torch.testing.assert_close(getattr(source, field), source_index)
        if full_cuda_graph and meta.num_prefills == 0 and actual is not None:
            assert actual.data_ptr() == getattr(dst, field).data_ptr()

    assert meta.num_accepted_tokens is source.num_accepted_tokens
    assert meta.spec_sequence_masks is source.spec_sequence_masks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_update_block_table_rebuilds_checkpoint_state_indices():
    """GLM checkpoint writes must use the destination group's block table."""
    device = torch.device("cuda")
    builders = [
        _create_gdn_builder(
            mamba_cache_mode="align",
            num_prefill_checkpoint_blocks=1,
            prefix_match_unit=BLOCK_SIZE,
            device=device,
        )
        for _ in range(3)
    ]
    src, dst, ref = builders
    batch = BatchSpec(seq_lens=[50, 32], query_lens=[50, 16])
    source_common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    destination_table = source_common.block_table_tensor.clone()
    destination_table[destination_table >= 0] += 100
    destination_common = source_common.replace(block_table_tensor=destination_table)

    for builder, common in ((dst, destination_common), (ref, destination_common)):
        builder.mamba_aligned_state_indices = mamba_get_block_table_tensor(
            common.block_table_tensor,
            common.seq_lens,
            builder.kv_cache_spec,
            "align",
        )

    source = src.build(0, source_common)
    expected = ref.build(0, destination_common)
    assert source.checkpoint is not None
    source_checkpoint_indices = source.checkpoint.state_indices.clone()

    actual = dst.update_block_table(
        source,
        destination_common.block_table_tensor,
        destination_common.slot_mapping,
    )

    assert actual.checkpoint is not None
    assert expected.checkpoint is not None
    torch.testing.assert_close(
        actual.checkpoint.state_indices, expected.checkpoint.state_indices
    )
    torch.testing.assert_close(
        actual.checkpoint.checkpoint_offsets, expected.checkpoint.checkpoint_offsets
    )
    torch.testing.assert_close(
        source.checkpoint.state_indices, source_checkpoint_indices
    )
    assert actual.checkpoint is not source.checkpoint


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_internal_checkpoint_metadata_targets_last_aligned_boundary():
    device = torch.device("cuda")
    builder = _create_gdn_builder(
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1,
        prefix_match_unit=BLOCK_SIZE,
        device=device,
    )
    batch = BatchSpec(seq_lens=[50, 32], query_lens=[50, 16])
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )

    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert meta.checkpoint is not None
    torch.testing.assert_close(
        meta.checkpoint.state_indices,
        torch.tensor([2, NULL_BLOCK_ID], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        meta.checkpoint.checkpoint_offsets,
        torch.tensor([48, 0], dtype=torch.int32, device=device),
    )


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)
