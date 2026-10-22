import os, resource
import torch, pytest
assert not torch.cuda.is_initialized()
rc = pytest.main(["--noconftest", "-q", "-p", "no:cacheprovider", "/source/tests/kernels/moe/test_sparkinfer_route_histogram.py", "/source/tests/v1/spec_decode/test_fp8_draft_head_precision.py"])
print("maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print("cuda_initialized", torch.cuda.is_initialized())
raise SystemExit(rc)
