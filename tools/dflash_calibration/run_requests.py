# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--base", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--runtime-contract", type=Path, required=True)
parser.add_argument("--capture-id", required=True)
parser.add_argument("--rank0-host", required=True)
parser.add_argument("--rank0-python", required=True)
parser.add_argument("--rank0-container", required=True)
parser.add_argument("--rank1-container", required=True)
parser.add_argument("--request-timeout-seconds", type=float, required=True)
args = parser.parse_args()
if args.request_timeout_seconds <= 0:
    raise ValueError("An explicit positive request timeout is required")
ROOT = args.root
BASE = args.base
MODEL = args.model
RUNTIME_HASH = hashlib.sha256(args.runtime_contract.read_bytes()).hexdigest()


def remote(code):
    return subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            args.rank0_host,
            shlex.quote(args.rank0_python) + " -c " + shlex.quote(code),
        ],
        text=True,
    )


def require_ready(capture, rank, remote_rank=False):
    code = (
        "from pathlib import Path\nimport json\n"
        f"p=Path({str(capture)!r})\n"
        "d=json.loads((p/'capture-ready.json').read_text())\n"
        "if d['boot_id'] != Path('/proc/sys/kernel/random/boot_id')"
        ".read_text().strip(): raise ValueError('stale boot')\n"
        "if (p/'active.json').exists() or list(p.glob('inflight-*.json')) "
        "or list(p.glob('producer-failed-*.json')): "
        "raise ValueError('unfinished producer lease')\n"
        "print(json.dumps(d))"
    )
    if remote_rank:
        receipt = json.loads(remote(code))
    else:
        receipt = json.loads(
            subprocess.check_output([sys.executable, "-c", code], text=True)
        )
    if (
        receipt["rank"] != rank
        or receipt["source"] != args.source
        or receipt["capture_id"] != args.capture_id
        or receipt["runtime_contract_sha256"] != RUNTIME_HASH
    ):
        raise ValueError("Missing/stale instrumentation readiness")
    # Receipt PIDs are container-namespace PIDs, not host PIDs.
    live_code = (
        "import os\nfrom pathlib import Path\n"
        f"os.kill({receipt['pid']!r},0)\n"
        "if Path('/proc/sys/kernel/random/boot_id').read_text().strip() "
        f"!= {receipt['boot_id']!r}: raise ValueError('stale boot')"
    )
    command = [
        "docker",
        "exec",
        args.rank0_container if remote_rank else args.rank1_container,
        receipt["executable"],
        "-c",
        live_code,
    ]
    if remote_rank:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.rank0_host, shlex.join(command)],
            check=True,
        )
    else:
        subprocess.run(command, check=True)
    if (
        receipt["hook_sha256"]
        != hashlib.sha256(
            Path(__file__).with_name("draft_kv_capture.py").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Installed hook differs from the maintained source")
    return receipt


def require_producer_complete(name):
    def check(capture):
        code = (
            "from pathlib import Path\nimport json\n"
            f"p=Path({str(capture)!r})\n"
            "if list(p.glob('inflight-*.json')) "
            "or list(p.glob('producer-failed-*.json')): "
            "raise ValueError('unfinished or failed producer')\n"
            "rows=[json.loads(x) for x in "
            "(p/'producer-completions.jsonl').read_text().splitlines()]\n"
            f"if not any(x['sample']=={name!r} "
            f"and x['capture_id']=={args.capture_id!r} "
            "and x['status']=='producer_completed' for x in rows): "
            "raise ValueError('missing producer completion')"
        )
        return code

    subprocess.run([sys.executable, "-c", check(ROOT / "capture/node-2")], check=True)
    remote(check(ROOT / "capture/node-1"))


require_ready(ROOT / "capture/node-2", 1)
require_ready(ROOT / "capture/node-1", 0, remote_rank=True)


def post(path, body, name):
    payload = json.dumps(body, separators=(",", ":")).encode()
    (ROOT / (name + "-request.json")).write_bytes(payload)
    req = urllib.request.Request(
        BASE + path, data=payload, headers={"Content-Type": "application/json"}
    )
    start = time.time()
    try:
        with urllib.request.urlopen(
            req, timeout=args.request_timeout_seconds
        ) as response:
            data = response.read()
    except urllib.error.HTTPError as error:
        (ROOT / (name + "-error.json")).write_bytes(error.read())
        raise
    (ROOT / (name + "-response.json")).write_bytes(data)
    result = json.loads(data)
    print(
        name,
        "completed",
        round(time.time() - start, 2),
        "s",
        result.get("usage"),
        flush=True,
    )
    return result


def marker(data):
    path = ROOT / "capture/node-2/active.json"
    if path.exists():
        raise ValueError("An unfinished capture lease already exists")
    path.write_text(json.dumps(data))
    code = (
        "from pathlib import Path; p=Path("
        + repr(str(ROOT / "capture/node-1/active.json"))
        + ");\nif p.exists(): raise ValueError('unfinished lease')\np.write_text("
        + repr(json.dumps(data))
        + ")"
    )
    remote(code)


def deactivate(name):
    require_producer_complete(name)
    path = ROOT / "capture/node-2/active.json"
    path.rename(path.with_name("finished-" + name + ".json"))
    code = (
        "from pathlib import Path; p=Path("
        + repr(str(ROOT / "capture/node-1/active.json"))
        + "); p.rename(p.with_name("
        + repr("finished-" + name + ".json")
        + "))"
    )
    remote(code)


canary = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "What is 17 plus 25? Give the numeric answer."}
    ],
    "max_tokens": 4096,
    "temperature": 1.0,
    "top_p": 0.95,
    "seed": 0,
    "chat_template_kwargs": {"enable_thinking": True},
}
first = post("/v1/chat/completions", canary, "fresh-functional-canary")
message = first["choices"][0]["message"]
text = message.get("content") or ""
if "42" not in text or first["choices"][0]["finish_reason"] == "length":
    raise ValueError("Functional canary failed")
print("Fresh functional canary passed", flush=True)
results = []
for row in json.loads((ROOT / "requests-text-only.json").read_text()):
    request = row["request"]
    name = row["sample"]
    body_hash = hashlib.sha256(
        json.dumps(request, separators=(",", ":")).encode()
    ).hexdigest()
    control = {
        "stage": row["stage"],
        "sample": name,
        "request_sha256": body_hash,
        "capture_id": args.capture_id,
        "runtime_contract_sha256": RUNTIME_HASH,
    }
    marker(control)
    try:
        result = post("/v1/chat/completions", request, name)
        choice = result["choices"][0]
        msg = choice["message"]
        if result["model"] != MODEL or result.get("error"):
            raise ValueError("Wrong model or failed request")
        if not any(
            msg.get(key)
            for key in ("content", "reasoning", "reasoning_content", "tool_calls")
        ):
            raise ValueError("Empty calibration response")
        if "tools" in request and not any(
            t.get("function", {}).get("name") == "lookup_weather"
            for t in msg.get("tool_calls", [])
        ):
            raise ValueError("Expected tool was not produced")
        results.append(
            {
                "sample": name,
                "stage": row["stage"],
                "usage": result.get("usage"),
                "finish_reason": choice["finish_reason"],
                "request_sha256": body_hash,
            }
        )
        (ROOT / "fresh-request-results.json").write_text(json.dumps(results, indent=2))
    except BaseException:
        # Keep both leases active until server termination is proven.
        # A client timeout does not prove producer completion.
        raise
    else:
        deactivate(name)
print("Fresh calibration/held-out requests complete", len(results), flush=True)
