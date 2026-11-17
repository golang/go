from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *arguments: str) -> str:
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], env=environment, text=True,
        timeout=30,
    ).strip()


def main() -> None:
    task = Path("/workspace/scratch/9949907ca71e")
    fork = task / "logos-json-integration"
    candidate = task / "logos-acceptance-fixes"
    record = task / "language-evidence/bounded-acceptance"
    old = task / "hosted-evidence/language/const-map-field-oracle"
    expected = {
        "bin/go": "987052b593c41cf60390eec80d98b01b89f9536d6d87b4e73672ab6e7a4eb2ef",
        "bin/logos": "987052b593c41cf60390eec80d98b01b89f9536d6d87b4e73672ab6e7a4eb2ef",
        "pkg/tool/linux_amd64/compile": "61749f8f2b2242d26e613993270f654cf47c750739a22b223f94eedd5370bba9",
    }
    assert git(fork, "rev-parse", "HEAD") == "45c29ff215477f7e403188d7452fc79aef05d791"
    assert git(fork, "status", "--porcelain") == ""
    actual = {name: digest(fork / name) for name in expected}
    assert actual == expected
    assert (candidate / "bin").is_symlink()
    assert (candidate / "bin").resolve() == fork / "bin"
    positive = candidate / "test/logos/run/const_func_map_field_store.lgo"
    output = candidate / "test/logos/run/const_func_map_field_store.out"
    assert digest(positive) == "1ea5cfa8168c9f1736aaeb77b7440126b496bfaaea7216e378989c52096ea2f5"
    assert digest(output) == "7de1555df0c2700329e815b93b32c571c3ea54dc967b89e81ab73b9972b72d1d"
    preparation = json.loads((old / "preparation.json").read_text())
    for item in preparation["ctfeFrozenChecks"]:
        assert digest(candidate / item["path"]) == item["sha256"]
    source = Path(tempfile.mkdtemp(prefix="logos-language-9949907ca71e-", dir="/home/agent"))
    module = source / "m"
    module.mkdir()
    (source / "tmp").mkdir()
    shutil.copyfile(positive, module / "main.lgo")
    shutil.copyfile(output, source / "expected.txt")
    (module / "go.mod").write_text("module example.com/constmapfield\n\ngo 1.27\n")
    (module / "logos.toml").write_text('[build]\narena = false\ntarget_cpu = "baseline"\n')
    objects = task / "language-acceptance-objects"
    objects.mkdir(exist_ok=True)
    common = {
        "GOMAXPROCS": "1", "GOGC": "20", "GOMEMLIMIT": "512MiB",
        "GOFLAGS": "-p=1", "GOTOOLCHAIN": "local", "GOENV": "off",
        "GOWORK": "off", "GOPROXY": "off", "GOSUMDB": "off",
        "GOEXPERIMENT": "", "CGO_ENABLED": "1", "CC": "/usr/bin/cc",
        "GOCACHE": str(task / "postbuild-cache"),
        "GOPATH": str(task / "acceptance-gopath"),
        "GOTMPDIR": str(objects), "TMPDIR": str(objects),
        "UPROOT1272": str(task / "toolchains/go1.27.2"),
        "UPROOT1273": str(task / "toolchains/go1.27.3"),
        "UPROOT1274": str(task / "toolchains/go1.27.4-ci"),
    }
    for name, root in (("frozen", fork), ("candidate", candidate)):
        values = dict(common, GOROOT=str(root), PATH=f'{root / "bin"}:{os.environ["PATH"]}')
        (record / f"{name}.env.json").write_text(json.dumps(values, indent=2) + "\n")
        values.update(TMPDIR=str(source / "tmp"), GOTMPDIR=str(source / "tmp"))
        (record / f"{name}-home.env.json").write_text(json.dumps(values, indent=2) + "\n")
    for version in ("1.27.2", "1.27.3", "1.27.4"):
        suffix = "-ci" if version == "1.27.4" else ""
        root = task / f"toolchains/go{version}{suffix}"
        values = dict(common, GOROOT=str(root), PATH=f'{root / "bin"}:{os.environ["PATH"]}')
        values["GOCACHE"] = str(task / "language-upstream-acceptance-cache")
        (record / f"upstream-{version}.env.json").write_text(json.dumps(values, indent=2) + "\n")
    state = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "module": str(module), "home_temp": str(source / "tmp"),
        "real_home_directory": source.resolve() == source,
        "fork_source": git(fork, "rev-parse", "HEAD", "HEAD^{tree}", "HEAD:src"),
        "fork_clean": git(fork, "status", "--porcelain") == "",
        "candidate_status": git(candidate, "status", "--porcelain"),
        "candidate_diff_sha256": hashlib.sha256(git(candidate, "diff", "--binary").encode()).hexdigest(),
        "installed_hashes": actual, "positive_sha256": digest(positive),
        "expected_output_sha256": digest(output),
        "ctfe_frozen_checks": preparation["ctfeFrozenChecks"],
        "candidate_bin_target": str((candidate / "bin").resolve()),
        "scope": "frozen45 installed compiler; candidate test inputs are component validation",
    }
    (record / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
