# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import site
import sys
import types
import unittest
from unittest.mock import patch

from contract import validate_events, validate_rows


class CaptureContractTests(unittest.TestCase):
    def events(self):
        return [
            dict(
                sample=sample,
                stage=stage,
                layer=layer,
                rank=rank,
                phase=phase,
                rows=1,
                k_absmax=1.0,
                v_absmax=2.0,
                provenance_version=2,
                request_sha256="request",
                capture_id="run",
            )
            for sample, stage in (("train", "calibration"), ("test", "heldout"))
            for layer in range(5)
            for rank in (0, 1)
            for phase in ("context", "query")
        ]

    def samples(self):
        return [
            dict(sample="train", stage="calibration"),
            dict(sample="test", stage="heldout"),
        ]

    def test_site_install_error_terminates_instead_of_being_ignored(self):
        module = types.ModuleType("draft_kv_capture")

        def fail():
            raise RuntimeError("capture dependency unavailable")

        module.install = fail
        with (
            patch.dict(sys.modules, {"draft_kv_capture": module}),
            patch.dict(os.environ, {"MIMO_DFLASH_CALIBRATION_DIR": "test"}),
            self.assertRaises(SystemExit),
        ):
            sys.modules.pop("sitecustomize", None)
            site.execsitecustomize()

    def test_stage_union_cannot_hide_missing_heldout_query_rank(self):
        events = [
            e
            for e in self.events()
            if not (
                e["stage"] == "heldout" and e["rank"] == 1 and e["phase"] == "query"
            )
        ]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_events(events, self.samples())

    def test_complete_stage_rank_phase_coverage(self):
        self.assertEqual(len(validate_events(self.events(), self.samples())), 10)

    def test_capture_hash_must_match_pinned_rendered_request(self):
        samples = self.samples()
        samples[0]["request"] = {"messages": [{"content": "fixture"}]}
        with self.assertRaisesRegex(ValueError, "pinned fixture"):
            validate_events(self.events(), samples)

    def test_missing_rendered_provenance_fails(self):
        with self.assertRaises(ValueError):
            validate_rows([], [2], ["request"], [5])

    def test_negative_padding_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([17], [2], ["request"], [-1])

    def test_aligned_actual_rows(self):
        validate_rows([17, 42], [2, 3], ["request", "request"], [5, 6])


if __name__ == "__main__":
    unittest.main()
