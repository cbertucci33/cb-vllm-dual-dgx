# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).parents[2] / "build" / "verify_runtime.py"
_SPEC = spec_from_file_location("unified_verify_runtime", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
verify_runtime = module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_runtime)
_GLM_TEMPLATE = _MODULE_PATH.parent / "assets" / "glm53" / "chat_template.jinja"


class _FakeDistribution:
    def __init__(
        self,
        name: str,
        version: str,
        *,
        requires: list[str] | None = None,
        path: str | None = None,
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.requires = requires
        self._path = Path(path or f"/{name}-{version}.dist-info")


def test_runtime_assets_accept_exact_glm_template(monkeypatch) -> None:
    monkeypatch.setattr(verify_runtime, "GLM_CHAT_TEMPLATE", _GLM_TEMPLATE)
    verify_runtime.verify_runtime_assets()


def test_runtime_assets_reject_missing_glm_template(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        verify_runtime, "GLM_CHAT_TEMPLATE", tmp_path / "chat_template.jinja"
    )
    with pytest.raises(RuntimeError, match="required GLM chat template is missing"):
        verify_runtime.verify_runtime_assets()


def test_runtime_assets_reject_modified_glm_template(monkeypatch, tmp_path) -> None:
    template = tmp_path / "chat_template.jinja"
    template.write_bytes(_GLM_TEMPLATE.read_bytes() + b"modified")
    monkeypatch.setattr(verify_runtime, "GLM_CHAT_TEMPLATE", template)
    with pytest.raises(RuntimeError, match="GLM chat template hash is"):
        verify_runtime.verify_runtime_assets()


def _set_torch_contract(monkeypatch, distribution: str, cuda: str | None) -> None:
    monkeypatch.setattr(
        verify_runtime.metadata,
        "version",
        lambda name: distribution if name == "torch" else pytest.fail(name),
    )
    monkeypatch.setattr(
        verify_runtime.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(version=SimpleNamespace(cuda=cuda))
            if name == "torch"
            else pytest.fail(name)
        ),
    )


@pytest.mark.parametrize("distribution", ["2.13.0+cu130", "2.13+cu130"])
def test_torch_contract_accepts_matching_cuda_local_version(
    monkeypatch, distribution: str
) -> None:
    _set_torch_contract(monkeypatch, distribution, "13.0")
    verify_runtime.require_torch_distribution("2.13.0", "13.0.3")


@pytest.mark.parametrize(
    ("distribution", "cuda", "error"),
    [
        ("2.13.1+cu130", "13.0", "torch version is 2.13.1+cu130"),
        ("2.13.0+cu130", "13.1", "torch CUDA version is 13.1"),
        ("2.13.0+cu130", None, "does not report a CUDA build version"),
    ],
)
def test_torch_contract_rejects_wrong_runtime(
    monkeypatch, distribution: str, cuda: str | None, error: str
) -> None:
    _set_torch_contract(monkeypatch, distribution, cuda)
    with pytest.raises(RuntimeError) as exc_info:
        verify_runtime.require_torch_distribution("2.13.0", "13.0.3")
    assert error in str(exc_info.value)


def test_flashinfer_cache_accepts_import_created_directories_and_log(tmp_path) -> None:
    cache_root = tmp_path / "flashinfer"
    workspace = cache_root / "revision" / ".cache" / "flashinfer" / "0.7.0"
    workspace.mkdir(parents=True)
    (workspace / "flashinfer_jit.log").write_text("import initialized\n")
    verify_runtime.verify_flashinfer_cache_root(cache_root)


@pytest.mark.parametrize("payload", ["cached_ops/topk.so", "generated/kernel.ptx"])
def test_flashinfer_cache_rejects_compiled_payload(tmp_path, payload: str) -> None:
    cache_root = tmp_path / "flashinfer"
    artifact = cache_root / payload
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="stale payloads"):
        verify_runtime.verify_flashinfer_cache_root(cache_root)


def test_flashinfer_cache_rejects_symlink(tmp_path) -> None:
    cache_root = tmp_path / "flashinfer"
    cache_root.mkdir()
    target = tmp_path / "target"
    target.write_text("stale")
    (cache_root / "flashinfer_jit.log").symlink_to(target)
    with pytest.raises(RuntimeError, match="stale payloads"):
        verify_runtime.verify_flashinfer_cache_root(cache_root)


def test_distribution_identity_rejects_same_root_duplicate(monkeypatch) -> None:
    distributions = [
        _FakeDistribution("ninja", "1.12", path="/site/ninja-1.12.dist-info"),
        _FakeDistribution("Ninja", "1.13", path="/site/ninja-1.13.dist-info"),
    ]
    monkeypatch.setattr(verify_runtime.site, "getsitepackages", lambda: ["/site"])
    monkeypatch.setattr(
        verify_runtime.metadata,
        "distributions",
        lambda path=None: distributions,
    )

    with pytest.raises(RuntimeError, match="duplicate distribution metadata"):
        verify_runtime.verify_distribution_identity()


def test_distribution_identity_rejects_cross_root_duplicate(monkeypatch) -> None:
    distributions = {
        "/site-a": [_FakeDistribution("ninja", "1.12", path="/site-a/ninja.dist-info")],
        "/site-b": [_FakeDistribution("Ninja", "1.13", path="/site-b/ninja.dist-info")],
    }
    monkeypatch.setattr(
        verify_runtime.site, "getsitepackages", lambda: list(distributions)
    )
    monkeypatch.setattr(
        verify_runtime.metadata,
        "distributions",
        lambda path=None: distributions[path[0]],
    )

    with pytest.raises(RuntimeError, match="duplicate distribution metadata"):
        verify_runtime.verify_distribution_identity()


def test_distribution_identity_allows_unmanaged_base_duplicates(monkeypatch) -> None:
    distributions = [
        _FakeDistribution(
            "cryptography", "50.0.2", path="/site/cryptography-50.0.2.dist-info"
        ),
        _FakeDistribution(
            "cryptography", "41.0.7", path="/site/cryptography-41.0.7.egg-info"
        ),
    ]
    monkeypatch.setattr(verify_runtime.site, "getsitepackages", lambda: ["/site"])
    monkeypatch.setattr(
        verify_runtime.metadata, "distributions", lambda path=None: distributions
    )

    verify_runtime.verify_distribution_identity()


def test_dependency_closure_requires_reconciled_nccl_metadata(
    monkeypatch,
) -> None:
    torch_dist = _FakeDistribution(
        "torch",
        "2.13.0",
        requires=['nvidia-nccl-cu13==2.29.7; platform_system == "Linux"'],
    )
    monkeypatch.setattr(verify_runtime.metadata, "distributions", lambda: [torch_dist])
    monkeypatch.setattr(
        verify_runtime.metadata,
        "version",
        lambda name: "2.30.7" if name == "nvidia-nccl-cu13" else pytest.fail(name),
    )

    with pytest.raises(RuntimeError, match="requires nvidia-nccl-cu13==2.29.7"):
        verify_runtime.verify_dependency_closure()
    torch_dist.requires = ['nvidia-nccl-cu13==2.30.7; platform_system == "Linux"']
    verify_runtime.verify_dependency_closure()


def test_dependency_closure_requires_matching_source_flashinfer(monkeypatch) -> None:
    vllm_dist = _FakeDistribution(
        "vllm",
        "0.32.1",
        requires=["flashinfer-python==0.7.0.post1"],
    )
    monkeypatch.setattr(verify_runtime.metadata, "distributions", lambda: [vllm_dist])
    monkeypatch.setattr(
        verify_runtime.metadata,
        "version",
        lambda name: "0.7.0" if name == "flashinfer-python" else pytest.fail(name),
    )

    with pytest.raises(RuntimeError, match="requires flashinfer-python==0.7.0.post1"):
        verify_runtime.verify_dependency_closure()
    vllm_dist.requires = ["flashinfer-python==0.7.0"]
    verify_runtime.verify_dependency_closure()


def test_dependency_closure_rejects_arbitrary_active_mismatch(monkeypatch) -> None:
    owner = _FakeDistribution("owner", "1", requires=["dependency==1"])
    monkeypatch.setattr(verify_runtime.metadata, "distributions", lambda: [owner])
    monkeypatch.setattr(verify_runtime.metadata, "version", lambda name: "2")

    with pytest.raises(RuntimeError, match="requires dependency==1"):
        verify_runtime.verify_dependency_closure()


def _sparse_mla_wrapper_contract() -> tuple[str, list[str]]:
    source = (_MODULE_PATH.parents[1] / "vllm/utils/flashinfer.py").read_text()
    exports = next(
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    )
    return source, exports


def test_sparse_mla_verifier_accepts_actual_scoped_public_route() -> None:
    """The shipped autotuning wrapper must not fail the old direct-alias check."""
    verify_runtime.verify_flashinfer_sparse_mla_wrapper(*_sparse_mla_wrapper_contract())


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ('"flashinfer.decode",', '"flashinfer.mla",'),
        ('"trtllm_batch_decode_with_kv_cache_mla",', '"wrong_decode",'),
        ("fallback_fn=_missing_sparse_mla,", "fallback_fn=_missing,"),
        (
            (
                "flashinfer_trtllm_batch_decode_with_kv_cache_mla = "
                "_autotune_sparse_mla_call("
            ),
            "flashinfer_trtllm_batch_decode_with_kv_cache_mla = (",
        ),
    ],
)
def test_sparse_mla_verifier_rejects_broken_route(original, replacement) -> None:
    source, exports = _sparse_mla_wrapper_contract()
    assert original in source
    with pytest.raises(RuntimeError, match="must retain scoped autotuning"):
        verify_runtime.verify_flashinfer_sparse_mla_wrapper(
            source.replace(original, replacement, 1), exports
        )


def test_sparse_mla_verifier_requires_public_export() -> None:
    source, exports = _sparse_mla_wrapper_contract()
    exports.remove("flashinfer_trtllm_batch_decode_with_kv_cache_mla")
    with pytest.raises(RuntimeError, match="must retain scoped autotuning"):
        verify_runtime.verify_flashinfer_sparse_mla_wrapper(source, exports)


@pytest.mark.parametrize("repair", ["cleanup", "startup-safety"])
def test_instanttensor_patch_payload_is_allowlisted_in_runner_context(repair) -> None:
    root = _MODULE_PATH.parents[1]
    dockerfile = (root / "build" / "Dockerfile.runner").read_text()
    assert "COPY build/patches/ /opt/vllm-runner/patches/" in dockerfile
    version_command = next(
        line
        for line in dockerfile.splitlines()
        if "build_instanttensor.py --version " in line
    )
    version = version_command.partition("--version ")[2].split()[0]
    builder = (root / "build" / "build_instanttensor.py").read_text()
    assert f'f"instanttensor-{{args.version}}-{repair}.patch"' in builder
    payload = f"build/patches/instanttensor-{version}-{repair}.patch"
    assert (root / payload).is_file()
    rules = [
        line.strip()
        for line in (root / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    # Both parent directories must be reopened before the literal payload
    # exception: the runner copies this allowlisted directory, not host files.
    build_open = rules.index("!/build/")
    patches_open = rules.index("!/build/patches/")
    patches_excluded = rules.index("/build/patches/*")
    assert build_open < patches_open < patches_excluded
    assert f"!/{payload}" in rules, f"Docker context excludes required {payload}"
    payload_open = rules.index(f"!/{payload}")
    assert patches_excluded < payload_open
    assert f"/{payload}" not in rules[payload_open + 1 :]
    assert not any("instanttensor-0.2.1-" in rule for rule in rules)
