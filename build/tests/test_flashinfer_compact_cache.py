# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU regression: execute real FlashInfer guards; no CUDA initialization."""

import importlib.util
import unittest
from pathlib import Path

import torch
from flashinfer.mla import _core
from flashinfer.mla import _sparse_mla_sm120 as sparse

spec = importlib.util.spec_from_file_location(
    "runner_verify", Path(__file__).parents[1] / "verify_runtime.py"
)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class CompactCacheContract(unittest.TestCase):
    def test_glm_public_to_inner_consumer(self):
        verify.verify_flashinfer_compact_cache_route(0, 528)

    def test_rope_public_to_inner_consumer(self):
        verify.verify_flashinfer_compact_cache_route(64, 656)

    def test_invalid_layouts_rejected(self):
        for nope, shape, dtype in (
            (True, (2, 64, 527), torch.uint8),
            (False, (2, 64, 528), torch.uint8),
            (False, (2, 64, 657), torch.uint8),
            (True, (2, 2, 64, 528), torch.uint8),
            (True, (2, 528), torch.uint8),
            (True, (2, 64, 528), torch.bfloat16),
        ):
            with (
                self.subTest(nope=nope, shape=shape, dtype=dtype),
                self.assertRaises(ValueError),
            ):
                _core._check_sm120_sparse_v32_kv_cache(
                    torch.empty(shape, dtype=dtype), glm53_nope=nope
                )

    def test_glm_padded_decode_layout(self):
        for width in (528, 656):
            cache = torch.empty((2, 1, 64, width), dtype=torch.uint8)
            self.assertIs(
                _core._check_sm120_sparse_v32_kv_cache(cache, glm53_nope=True), cache
            )
            self.assertEqual(
                sparse._packed_kv_page_block_size(
                    cache, model_type=sparse._MODEL_TYPE_GLM53_NOPE, name="kv_cache"
                ),
                64,
            )

    def test_capability_and_calibration_footprint(self):
        from flashinfer.mla import _sparse_mla_sm120_cpb

        self.assertEqual(
            sparse.supported_sparse_mla_sm120_configs()["glm53_nope"].bytes_per_token,
            528,
        )
        self.assertEqual(_sparse_mla_sm120_cpb._BYTES_PER_TOKEN["glm53_nope"], 528)

    def test_model_disambiguation_and_flat_layout(self):
        self.assertEqual(
            sparse._resolve_model_type(512, "arbitrary_fp32"),
            sparse._MODEL_TYPE_GLM53_NOPE,
        )
        self.assertEqual(
            sparse._bytes_per_token_for_model_type(sparse._MODEL_TYPE_DSV4), 584
        )
        cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
        self.assertEqual(
            sparse._packed_kv_page_block_size(
                cache, model_type=sparse._MODEL_TYPE_GLM53_NOPE, name="kv_cache"
            ),
            64,
        )


if __name__ == "__main__":
    result = unittest.main(exit=False)
    assert not torch.cuda.is_initialized(), "CPU contract unexpectedly initialized CUDA"
    raise SystemExit(not result.result.wasSuccessful())
