"""Offline head comparison. CPU checks now; --run ONLY in an approved GPU window."""
import argparse
import ast
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import torch

ROOT = Path("/opt/project/worktrees/glm32-host-clamp-20261021")
HERE = Path(__file__).parent

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def source_edges():
    path = ROOT / "vllm/model_executor/models/qwen3_dflash2.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_score_edges")
    env = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), env)
    return env["_score_edges"]

def marginals(edges, temperature):
    # Exact distribution of the sequential categorical walk, not a sampled path.
    conditional = (edges.float() / temperature[:, None, None, None]).softmax(-1)
    state = conditional[:, 0, 0]
    result = [state]
    for step in range(1, edges.shape[1]):
        state = torch.einsum("bp,bpc->bc", state, conditional[:, step])
        result.append(state)
    return torch.stack(result, dim=1), conditional

def distance(ids_a, p_a, ids_b, p_b):
    match = ids_a[..., :, None] == ids_b[..., None, :]
    aligned = (match * p_b[..., None, :]).sum(-1)
    missing = (p_b * ~match.any(-2)).sum(-1)
    return ((p_a - aligned).abs().sum(-1) + missing) * 0.5

def compare(ids, probabilities, ref_ids, ref_probabilities):
    matches = ids[..., :, None] == ref_ids[..., None, :]
    tv = distance(ids, probabilities, ref_ids, ref_probabilities)
    return {
        "top16_overlap_fraction": matches.any(-1).float().mean().item(),
        "reference_top1_present_fraction": (ids == ref_ids[..., :1]).any(-1).float().mean().item(),
        "proposal_marginal_tv_mean": tv.mean().item(),
        "proposal_marginal_tv_max": tv.max().item(),
    }

def self_test():
    torch.manual_seed(12)
    edges_fn = source_edges()
    cases = 0
    for batch in (1, 2):
        for steps in (1, 3, 7):
            k, vocab, rank = 3, 32, 8
            ids = torch.stack([torch.randperm(vocab)[:k] for _ in range(batch * steps)]).view(batch, steps, k)
            edges = edges_fn(torch.randn(vocab, rank), torch.randn(vocab, rank), ids,
                             torch.randn(batch, steps, k), torch.randn(batch, steps, rank),
                             torch.arange(batch), k)
            for t in (0.5, 1.0, 2.0):
                p, c = marginals(edges, torch.full((batch,), t))
                assert torch.allclose(p.sum(-1), torch.ones(batch, steps), atol=1e-6)
                assert torch.equal(distance(ids, p, ids, p), torch.zeros(batch, steps))
                assert torch.allclose(distance(ids, p, ids.flip(-1), p.flip(-1)), torch.zeros(batch, steps))
                assert torch.allclose(distance(ids, p, ids + vocab, p), torch.ones(batch, steps))
                exact = torch.zeros_like(p)
                for path in itertools.product(range(k), repeat=steps):
                    mass = torch.ones(batch)
                    previous = 0
                    for step, index in enumerate(path):
                        mass = mass * c[:, step, previous, index]
                        previous = index
                    for step, index in enumerate(path):
                        exact[:, step, index] += mass
                assert torch.allclose(p, exact, atol=2e-5)
                cases += 1
    assert not torch.cuda.is_initialized()
    print(json.dumps({"cases": cases, "cuda_initialized": False, "torch": torch.__version__,
                      "scope": "actual score_edges plus reporting math; synthetic CPU inputs only"}))

def run(args):
    from safetensors import safe_open
    from flashinfer import top_k
    old = module(HERE / "old_fp8_draft_head.py", "old_head")
    current = module(ROOT / "vllm/model_executor/layers/fp8_draft_head.py", "current_head")
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    required = {"hidden", "selector_hidden", "anchor_ids", "temperature", "shards", "provenance"}
    assert required <= capture.keys(), required - capture.keys()
    assert capture["provenance"]["finite_repaired_runner"] is True
    hidden, projected = capture["hidden"], capture["selector_hidden"]
    assert hidden.ndim == 3 and hidden.shape[1:] == (7, 4096)
    assert projected.shape == (*hidden.shape[:2], 256)
    assert hidden.dtype == projected.dtype == torch.bfloat16
    assert torch.isfinite(hidden).all() and torch.isfinite(projected).all()
    assert (capture["temperature"] > 0).all()
    shards = capture["shards"]
    assert len(shards) == 2
    assert shards[0]["start"] == 0 and sum(s["real_rows"] for s in shards) == 154880
    assert shards[1]["start"] == shards[0]["real_rows"]
    device = torch.device("cuda")
    x = hidden.flatten(0, 1).to(device)
    table_path = "/opt/project/models/glm53-dflash2-mxfp8/model.safetensors"
    weight_path = "/opt/project/models/glm53-exl3-tp2-rank-sliced/model-00083.safetensors"
    k = 16
    results = {}
    for mode in ("bf16_reference", "old_fp8_bf16", "current_fp8_fp32"):
        local_values, local_ids, timings = [], [], []
        for shard in shards:
            start, count, padded = shard["start"], shard["real_rows"], shard["padded_rows"]
            assert padded >= count
            with safe_open(weight_path, framework="pt", device="cpu") as f:
                w = f.get_slice("lm_head.weight")[start:start + count].to(device)
            w = torch.nn.functional.pad(w, (0, 0, 0, padded - count))
            implementation = old if mode == "old_fp8_bf16" else current
            head = None if mode == "bf16_reference" else implementation.quantize_draft_head(w)
            def operation():
                logits = torch.nn.functional.linear(x, w) if head is None else implementation.fp8_draft_head_logits(x, head)
                if count < padded:
                    logits[:, count:] = -float("inf")
                return top_k(logits, k, sorted=True, deterministic=True)
            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeats):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                values, ids = operation()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_values, graph_ids = operation()
            graph_samples = []
            for _ in range(args.repeats):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                graph_samples.append(begin.elapsed_time(end))
            assert torch.equal(values, graph_values) and torch.equal(ids, graph_ids), "eager/graph candidate mismatch"
            timings.append({"shard": shard, "eager_median_ms": statistics.median(samples),
                            "eager_samples_ms": samples, "graph_median_ms": statistics.median(graph_samples),
                            "graph_samples_ms": graph_samples})
            del graph, graph_values, graph_ids
            local_values.append(values)
            local_ids.append(ids.to(torch.int64) + start)
            del head, w
        # Rank-major top-k. Offline serial shard replay does not measure NCCL.
        values, selected = top_k(torch.cat(local_values, -1).contiguous(), k, sorted=True, deterministic=True)
        ids = torch.cat(local_ids, -1).gather(-1, selected)
        values = values.float()
        softcap = capture["provenance"]["soft_cap"]
        if softcap is not None:
            values = torch.tanh(values / softcap) * softcap
        values *= capture["provenance"]["output_multiplier"]
        ids = ids.view(*hidden.shape[:2], k)
        values = values.view_as(ids)
        with safe_open(table_path, framework="pt", device="cpu") as f:
            pre = f.get_tensor("candidate_selector.predecessor_codebook").to(device)
            suc = f.get_tensor("candidate_selector.successor_codebook").to(device)
        edges = source_edges()(pre, suc, ids, values, projected.to(device), capture["anchor_ids"].to(device), k)
        probs, conditional = marginals(edges, capture["temperature"].to(device))
        results[mode] = {"ids": ids.cpu(), "unary": values.cpu(), "probabilities": probs.cpu(),
                         "conditional": conditional.cpu(), "timing": timings}
        del pre, suc, edges, probs, conditional
    ref = results["bf16_reference"]
    report = {mode: {**compare(r["ids"], r["probabilities"], ref["ids"], ref["probabilities"]),
                     "local_head_plus_topk_timing": r["timing"]} for mode, r in results.items()}
    Path(args.output).write_text(json.dumps(report, indent=2))
    torch.save(results, str(args.output) + ".pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--capture")
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.capture, args.output, args.warmup, args.repeats)):
            parser.error("--run requires capture, output, warmup and repeats from the approved run plan")
        with torch.inference_mode():
            run(args)

