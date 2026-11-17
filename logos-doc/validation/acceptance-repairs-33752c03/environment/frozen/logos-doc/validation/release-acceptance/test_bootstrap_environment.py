#!/usr/bin/env python3
"""Execute the release bootstrap boundary with build/download commands stubbed."""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Mapping


SHELL = r"""set -euo pipefail
repo=$1
job_root=$2
RUNNER_TEMP=$3
evidence="$job_root/evidence"

snapshot() {
  local phase=$1 name
  for name in GOMAXPROCS GOGC GOMEMLIMIT GOFLAGS TMPDIR GOTMPDIR GOCACHE CC CXX PATH CGO_ENABLED; do
    if [[ -v $name ]]; then
      printf 'state\t%s\t%s\t%s\n' "$phase" "$name" "${!name}"
    fi
  done
}
preflight() { printf 'guard\tpreflight\n'; }
clean_tip() { printf 'guard\tclean_tip\n'; }
run() {
  printf 'run\t%s\n' "$1"
  if [[ $1 == make ]]; then
    snapshot make
  fi
}
go() {
  if [[ $* != 'env GOAMD64 GOARM64 GOEXPERIMENT CGO_ENABLED' ]]; then
    printf 'unexpected direct Go invocation: %s\n' "$*" >&2
    return 97
  fi
  printf 'v1\nv8.0\n\n1\n'
}
source "$4"
bootstrap
snapshot ready
printf 'complete\n'
"""


class BootstrapEnvironmentTests(unittest.TestCase):
    root: ClassVar[Path]
    temporary: ClassVar[tempfile.TemporaryDirectory[str]]
    sandbox: ClassVar[Path]
    compiler_bin: ClassVar[Path]
    states: ClassVar[Mapping[str, Mapping[str, str]]]
    stages: ClassVar[tuple[str, ...]]

    @classmethod
    def setUpClass(cls) -> None:
        source = (cls.root / ".github/scripts/release-acceptance.sh").read_text()
        definitions = re.findall(r"^bootstrap\(\) \{\n.*?^\}$", source, re.M | re.S)
        if len(definitions) != 1:
            raise ValueError("release executor must define exactly one bootstrap function")

        cls.temporary = tempfile.TemporaryDirectory(prefix="logos-bootstrap-contract-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.sandbox = Path(cls.temporary.name)
        repo = cls.sandbox / "repo"
        runner = cls.sandbox / "runner"
        job_root = runner / "job"
        cls.compiler_bin = cls.sandbox / "commands"
        for directory in (repo / "src", repo / "bin", repo / "pkg/tool/linux_amd64",
                          job_root / "evidence", cls.compiler_bin):
            directory.mkdir(parents=True)

        # The restricted PATH contains only filesystem utilities and fake compilers.
        # run() never executes a downloader, Go command, make, or C compilation.
        for name in ("mkdir", "ls", "ln", "sha256sum"):
            command = shutil.which(name)
            if command is None:
                raise ValueError(f"test prerequisite is absent: {name}")
            (cls.compiler_bin / name).symlink_to(command)
        for name, version in (("gcc-14", "14.2.0"), ("g++-14", "14.2.0"),
                              ("g++", "13.1.0")):
            compiler = cls.compiler_bin / name
            compiler.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version}'\n")
            compiler.chmod(0o755)
        for name in ("bin/go", "bin/logos", "pkg/tool/linux_amd64/compile",
                     "pkg/tool/linux_amd64/link", "pkg/tool/linux_amd64/asm",
                     "pkg/tool/linux_amd64/cgo"):
            (repo / name).write_text("inert tool identity fixture\n")

        bootstrap = cls.sandbox / "bootstrap.sh"
        bootstrap.write_text(f"{definitions[0]}\n")
        process = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", SHELL, "bootstrap-test",
             str(repo), str(job_root), str(runner), str(bootstrap)],
            cwd=repo,
            env={"PATH": str(cls.compiler_bin), "LC_ALL": "C"},
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        states: dict[str, dict[str, str]] = {"make": {}, "ready": {}}
        stages: list[str] = []
        for line in process.stdout.splitlines():
            fields = line.split("\t", 3)
            if fields[0] == "state" and len(fields) == 4:
                states[fields[1]][fields[2]] = fields[3]
            elif fields[0] == "run" and len(fields) == 2:
                stages.append(fields[1])
        if not process.stdout.endswith("complete\n"):
            raise ValueError("bootstrap did not reach its acceptance boundary")
        cls.states = MappingProxyType({
            phase: MappingProxyType(values) for phase, values in states.items()
        })
        cls.stages = tuple(stages)

    def test_make_is_bounded_and_acceptance_uses_a_new_cache(self) -> None:
        build = self.states["make"]
        for name, expected in (("GOMAXPROCS", "2"), ("GOGC", "50"),
                               ("GOMEMLIMIT", "2GiB"), ("GOFLAGS", "-p=2")):
            with self.subTest(name=name):
                self.assertEqual(build[name], expected)
        self.assertEqual(self.stages.count("make"), 1)
        self.assertLess(self.stages.index("make"), self.stages.index("installed-go"))
        self.assertIn("c23-compile", self.stages)
        self.assertIn("c23-run", self.stages)
        self.assertNotEqual(build["GOCACHE"], self.states["ready"]["GOCACHE"])
        self.assertEqual(list(Path(self.states["ready"]["GOCACHE"]).iterdir()), [])

    def test_acceptance_does_not_inherit_build_limits(self) -> None:
        for name in ("GOMAXPROCS", "GOGC", "GOMEMLIMIT", "GOFLAGS"):
            with self.subTest(name=name):
                self.assertNotIn(
                    name, self.states["ready"],
                    f"make-only {name} reaches the acceptance environment",
                )

    def test_c_and_cxx_compilers_have_matching_lto_versions(self) -> None:
        ready = self.states["ready"]
        versions: dict[str, str] = {}
        for name, fallback in (("CC", "cc"), ("CXX", "g++")):
            command = shlex.split(ready.get(name, fallback))
            executable = shutil.which(command[0], path=ready["PATH"])
            self.assertIsNotNone(executable, f"effective {name} is absent")
            assert executable is not None
            self.assertTrue(Path(executable).resolve().is_relative_to(self.compiler_bin))
            process = subprocess.run(
                [executable, *command[1:], "-dumpfullversion", "-dumpversion"],
                check=True, capture_output=True, text=True, timeout=5,
                env={"PATH": str(self.compiler_bin), "LC_ALL": "C"},
            )
            versions[name] = process.stdout.strip()
        self.assertEqual(versions["CC"], versions["CXX"],
                         f"C and C++ LTO versions differ: {versions}")

    def test_go_test_temp_files_are_inside_the_local_temp_boundary(self) -> None:
        ready = self.states["ready"]
        temporary_root = Path(ready["TMPDIR"]).resolve()
        with tempfile.TemporaryDirectory(dir=ready["GOTMPDIR"]) as directory:
            attack = Path(directory) / "linker-input.o"
            attack.write_bytes(b"local linker security fixture")
            self.assertTrue(
                attack.resolve().is_relative_to(temporary_root),
                f"GOTMPDIR fixture {attack} lies outside local TMPDIR {temporary_root}",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    BootstrapEnvironmentTests.root = args.root.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BootstrapEnvironmentTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
