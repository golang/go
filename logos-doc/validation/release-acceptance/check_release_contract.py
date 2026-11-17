#!/usr/bin/env python3
"""Check the static release-workflow contract; this never runs Go or a job."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import cast

import yaml


def mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a mapping with string keys")
    return cast(dict[str, object], value)


def strings(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{context} must be a list of strings")
    return cast(list[str], value)


def text(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def runs(job: dict[str, object]) -> str:
    steps = job.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")
    return "\n".join(
        text(step["run"], "run")
        for item in steps
        if "run" in (step := mapping(item, "step"))
    )


def function(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{\n(.*?)^\}}$", source, re.M | re.S)
    if match is None:
        return ""
    return "\n".join(line for line in match[1].splitlines() if not line.lstrip().startswith("#"))


def check(root: Path) -> dict[str, object]:
    workflow_path = root / ".github/workflows/release-acceptance.yml"
    script_path = root / ".github/scripts/release-acceptance.sh"
    request_path = root / "logos-doc/validation/release-acceptance-request.json"
    failures: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    require(workflow_path.is_file(), "dedicated release workflow is absent")
    require(script_path.is_file(), "release executor is absent")
    require(request_path.is_file(), "explicit source-tree request is absent")
    if failures:
        failures.extend([
            "six explicit no-short go/logos O1/O2/O3 std/cmd cells are absent",
            "complete unfiltered parity release/stamp job is absent",
            "two independent reproducibility hosts and digest comparison are absent",
            "mandatory real gopls/QEMU/Wasm execution gate is absent",
        ])
        return {"passed": False, "failures": failures}

    document = mapping(yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader), "workflow")
    jobs = mapping(document.get("jobs"), "jobs")
    source = script_path.read_text()
    optimization = mapping(jobs.get("optimization"), "optimization")
    strategy = mapping(optimization.get("strategy"), "strategy")
    matrix = mapping(strategy.get("matrix"), "matrix")
    require(set(strings(matrix.get("driver"), "drivers")) == {"go", "logos"}, "matrix must include both drivers")
    require(set(strings(matrix.get("level"), "levels")) == {"1", "2", "3"}, "matrix must include O1, O2 and O3")
    require(strategy.get("fail-fast") == "false", "one failed matrix cell must not cancel its siblings")
    require("release-acceptance.sh matrix" in runs(optimization), "matrix must invoke its actual executor")
    body = function(source, "matrix_gate")
    require("go list std cmd" in body, "matrix must derive the complete std/cmd package inventory")
    require('"-O=$level"' in body and '"$driver" test' in body, "matrix must pass its selected driver and optimization level")
    require("-count=1" in body and "-short" not in body, "matrix must be uncached and non-short")
    require('"${packages[@]}"' in body and "-parallel=1" in body and '"$skip" os' in body, "matrix must run both bulk packages and the separate os gate")
    require("logos_redteam_80.txt" in body and "logos_redteam_80" in body, "matrix must use and execute the frozen 31-name skip contract")
    require("31" in body and "-skip" in body, "matrix must verify exactly the pinned explicit exceptions")

    parity = mapping(jobs.get("parity"), "parity")
    require("release-acceptance.sh parity" in runs(parity), "parity job must invoke its actual executor")
    body = function(source, "parity_gate")
    require("cmd/internal/goparity" in body and "-goparity.release" in body and "-goparity.stamp=" in body, "parity must request the complete release suite and a real stamp")
    require("-count=1" in body and not any(flag in body for flag in ("-short", "-run", "-skip", "upstream=off")), "stamped parity must be uncached, unfiltered and oracle-enabled")
    require(body.count("clean_tip") >= 2 and "cmp" in body and "goparity.stamp" in body, "parity must bind its stamp to the unchanged clean tip")
    require("ledger_no_planned" in body, "parity must reject unpromoted planned rows before expensive work")
    require("goparity.release" not in function(source, "ledger_no_planned"), "ledger preflight must not substitute a filtered release invocation")

    host_a = mapping(jobs.get("repro-a"), "repro-a")
    host_b = mapping(jobs.get("repro-b"), "repro-b")
    require(host_a.get("runs-on") != host_b.get("runs-on"), "reproducibility must use two explicitly independent hosted environments")
    require(all("release-acceptance.sh repro" in runs(job) for job in (host_a, host_b)), "both hosts must execute the same reproducibility recipe")
    comparison = mapping(jobs.get("repro-compare"), "repro-compare")
    require(set(strings(comparison.get("needs"), "repro comparison needs")) == {"repro-a", "repro-b"}, "comparison must wait for both successful hosts")
    require("release-acceptance.sh compare" in runs(comparison), "comparison must consume the two actual job outputs")
    body = function(source, "repro_gate")
    require("logos_repro" in body and "logos_eject_example" in body, "repro gate must retain the unchanged reproducibility and example contracts")
    require("order_service.lgo" in body and "logos_eject_example.txt" in body, "the actual named M22 example must use its committed source/module")
    require("linux/amd64 linux/arm64 js/wasm" in body and "for copy in a b" in body, "the actual example needs two builds for all three targets")
    require("mktemp -d" in body and "GOCACHE=$cache" in body and "-trimpath -buildvcs=false" in body, "each actual-example build needs a new empty cache and stable build flags")
    body = function(source, "compare_gate")
    require("host_fingerprint" in body and "!= " in body, "comparison must reject reuse of the same host")
    require(".digests" in body and ".src_tree" in body and ".head" in body and ".recipe" in body, "comparison must match both source identity and every digest/recipe")

    runtime = mapping(jobs.get("runtimes"), "runtimes")
    require("release-acceptance.sh runtimes" in runs(runtime), "real-runtime job must invoke its actual executor")
    body = function(source, "install_runtimes")
    require("golang.org/x/tools/gopls@v0.24.0" in body and "h1:nPO/LffYa4sP62Ueuema5PPnZPGaZjGl+vZGvbTek4g=" in body, "gopls must retain its exact version and authenticated module sum")
    require("stock-cache" in body and "version -m" in body and "GOROOT=$stock" in body, "gopls must use stock Go, a separate cache and verified build information")
    require("ca6ede739327a20ae5a3498230c8a3a9a072c586e131bc6bcc1201eb1725e336" in body and "sha256sum --check" in body, "QEMU must be verified before execution")
    body = function(source, "runtime_gate")
    require("LOGOS_REQUIRE_WASMFX=1" in body and "node --version" in body, "Wasm execution needs explicit pinned Node and wasmfx prerequisites")
    require("logos_lsp" in body and "logos_sum_crossarch" in body and "logos_go_parity_wasm" in body and "logos_wasm_" in body, "runtime selection must include LSP, arm64 sum, and all Wasm scripts")
    require("-race" in body and "-v" in body and "-short" not in body, "runtime gate must execute uncached verbose cmd/go race scripts without short mode")
    require("11" in body and "lsp-markers" in body and "verify_script" in body, "runtime result must prove all eleven frozen LSP messages and no script skip")
    require("exec qemu-aarch64" in body and "cmp" in body and "native.txt" in body, "runtime result must prove the actual arm64 run and comparison")
    require("experimental-wasm-wasmfx" in body, "runtime result must prove the stack-switching execution command")
    body = function(source, "verify_script")
    require("PASS" in body and "SKIP" in body and "return 1" in body, "guarded script PASS/skip checks must fail closed")

    closure = mapping(jobs.get("closure"), "closure")
    require("release-acceptance.sh closure" in runs(closure), "full release closure must be reported independently")
    require("216|217|218|226" in function(source, "closure_gate"), "closure must retain exactly the four historic blocked exceptions")
    for name in ("optimization", "parity", "repro-a", "repro-b", "runtimes"):
        require("closure" not in str(mapping(jobs[name], name).get("needs", "")), f"{name} must remain independent of blocked binding closure")
    require(mapping(document.get("concurrency"), "concurrency").get("cancel-in-progress") == "false", "later branch updates must not erase an in-progress release proof")
    require(not re.search(r"^\s*uses:", workflow_path.read_text(), re.M), "repository action policy permits shell-only jobs")
    require("src_tree" in function(source, "identity") and "status --porcelain" in function(source, "clean_tip"), "identity must verify the requested source tree and clean checkout")
    require("base64" in function(source, "retain") and "sha256sum" in function(source, "retain"), "recoverable evidence needs a hashed raw-log bundle")
    require("MemAvailable" in function(source, "preflight") and "df -B1" in function(source, "preflight"), "heavy jobs need measured resource preflight")
    return {
        "passed": not failures,
        "failures": failures,
        "input_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (workflow_path, script_path, request_path)
        },
        "boundary": "Static coverage contract only. No hosted job or runtime was executed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.root)
    except (OSError, ValueError, yaml.YAMLError) as error:
        result = {"passed": False, "failures": [str(error)]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
