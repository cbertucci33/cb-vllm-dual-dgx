# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

if os.environ.get("MIMO_DFLASH_CALIBRATION_DIR"):
    try:
        from draft_kv_capture import install

        install()
    except BaseException as error:
        # site.py ignores ordinary Exception from sitecustomize.
        raise SystemExit(
            "DFlash capture installation failed: " + repr(error)
        ) from error
