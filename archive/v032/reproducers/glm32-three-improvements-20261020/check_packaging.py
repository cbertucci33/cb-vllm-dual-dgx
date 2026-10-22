import resource, torch, pytest
rc = pytest.main(["--noconftest", "-q", "-p", "no:cacheprovider", "/source/tests/kernels/moe/test_sparkinfer_route_histogram.py::test_histogram_patch_packaging"])
print("maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print("cuda_initialized", torch.cuda.is_initialized())
raise SystemExit(rc)
