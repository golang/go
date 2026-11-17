# pyright: strict
"""Exercise only the shell environment boundary, never a Go executable."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

WRAPPER = Path(__file__).resolve().parents[3] / ".github/scripts/logos-test-env.sh"


def invoke(env: dict[str, str], *arguments: str) -> dict[str, object]:
    script = "import json, os, sys; print(json.dumps({'env': dict(os.environ), 'args': sys.argv[1:]}))"
    result = subprocess.run(
        (str(WRAPPER), sys.executable, "-c", script, *arguments),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""
    value: object = json.loads(result.stdout)
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def environment(result: dict[str, object]) -> dict[str, str]:
    value = result["env"]
    assert isinstance(value, dict)
    fields = cast(dict[object, object], value)
    assert all(
        isinstance(key, str) and isinstance(item, str) for key, item in fields.items()
    )
    return cast(dict[str, str], fields)


@pytest.mark.parametrize(
    "experiment", (None, "", "none,simd", "logosarena,nogreenteagc")
)
def test_experiment_value_and_presence_are_preserved(experiment: str | None) -> None:
    env = {"PATH": "/usr/bin:/bin"}
    if experiment is not None:
        env["GOEXPERIMENT"] = experiment
    forwarded = environment(invoke(env))
    assert ("GOEXPERIMENT" in forwarded) == (experiment is not None)
    assert forwarded.get("GOEXPERIMENT") == experiment


@pytest.mark.parametrize("value", (None, "", "/usr/bin:/bin"))
def test_path_value_and_exported_presence_are_preserved(value: str | None) -> None:
    env: dict[str, str] = {}
    if value is not None:
        env["PATH"] = value
    forwarded = environment(invoke(env))
    assert ("PATH" in forwarded) == (value is not None)
    assert forwarded.get("PATH") == value


@pytest.mark.parametrize(
    "key",
    (
        "HOME",
        "USER",
        "TMPDIR",
        "GO_BUILDER_NAME",
        "UPROOT1272",
        "UPROOT1273",
        "UPROOT1274",
    ),
)
@pytest.mark.parametrize("value", (None, "", "/existing path/with spaces"))
def test_host_and_oracle_values_are_not_synthesized(
    key: str, value: str | None
) -> None:
    env = {"PATH": "/usr/bin:/bin"}
    if value is not None:
        env[key] = value
    forwarded = environment(invoke(env))
    assert (key in forwarded) == (value is not None)
    assert forwarded.get(key) == value


@pytest.mark.parametrize(
    "key",
    (
        "GOFIPS140",
        "GOPROXY",
        "GOSUMDB",
        "GOPRIVATE",
        "GONOPROXY",
        "GONOSUMDB",
        "GOINSECURE",
        "GOVCS",
        "GOAUTH",
    ),
)
@pytest.mark.parametrize("value", (None, "", "configured-value with spaces"))
def test_go_selection_and_module_policy_are_preserved(
    key: str, value: str | None
) -> None:
    env = {"PATH": "/usr/bin:/bin"}
    if value is not None:
        env[key] = value
    forwarded = environment(invoke(env))
    assert (key in forwarded) == (value is not None)
    assert forwarded.get(key) == value


def test_build_cache_resource_and_tool_paths_are_preserved(tmp_path: Path) -> None:
    env = {
        "HOME": str(tmp_path),
        "USER": "runner",
        "TMPDIR": str(tmp_path / "temp"),
        "PATH": f"{tmp_path}/c23-cc:{tmp_path}/node:/usr/bin:/bin",
        "GOFLAGS": "-p=1",
        "GOROOT": str(tmp_path / "goroot"),
        "GOENV": "off",
        "GOTOOLCHAIN": "local",
        "GOCACHE": str(tmp_path / "cache"),
        "GOMODCACHE": str(tmp_path / "modules"),
        "GOMAXPROCS": "2",
        "GOMEMLIMIT": "2GiB",
        "GOGC": "50",
        "GO_TEST_TIMEOUT_SCALE": "4",
        "GO_TEST_SHARDS": "10",
        "GO_TEST_SHORT": "1",
        "GO_BUILDER_NAME": "actual-builder",
        "CC": "clang-18",
        "CXX": "clang++-18",
        "CGO_ENABLED": "1",
        "CGO_CFLAGS": "-I/a path",
        "LD_LIBRARY_PATH": str(tmp_path / "lib"),
        "UPROOT1272": "/oracle/1.27.2",
        "UPROOT1273": "/oracle/1.27.3",
        "UPROOT1274": "/oracle/1.27.4",
    }
    forwarded = environment(invoke(env))
    assert all(forwarded.get(key) == value for key, value in env.items())


def test_actions_and_transport_settings_are_not_forwarded() -> None:
    blocked = (
        "GITHUB_ACTIONS",
        "GITHUB_WORKSPACE",
        "GITHUB_TOKEN",
        "ACTIONS_RUNTIME_TOKEN",
        "AWS_SESSION_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "SSH_AUTH_SOCK",
        "NODE_OPTIONS",
    )
    env = {
        "PATH": "/usr/bin:/bin",
        **{key: f"discard-{index}" for index, key in enumerate(blocked)},
    }
    forwarded = environment(invoke(env))
    assert all(key not in forwarded for key in blocked)


def test_argument_and_environment_text_are_never_evaluated() -> None:
    arguments = (
        "space separated",
        "$(printf command-substitution)",
        "; exit 91",
        "line\nbreak",
    )
    value = "none,simd; $(printf environment-substitution)"
    result = invoke({"PATH": "/usr/bin:/bin", "GOEXPERIMENT": value}, *arguments)
    assert result["args"] == list(arguments)
    forwarded = environment(result)
    assert forwarded["GOEXPERIMENT"] == value


def test_no_command_is_an_error() -> None:
    result = subprocess.run(
        (str(WRAPPER),), env={"PATH": "/usr/bin:/bin"}, check=False, capture_output=True
    )
    assert result.returncode == 2
    assert result.stdout == b""
    assert b"usage:" in result.stderr
