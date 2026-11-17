from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import subprocess


@dataclass(frozen=True)
class Result:
    command: tuple[str, ...]
    status: int
    stdout: str
    stderr: str


def run(script: str, cwd: Path) -> Result:
    command = ("bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", script)
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return Result(command, completed.returncode, completed.stdout, completed.stderr)


def main() -> None:
    root = Path("/workspace/scratch/9949907ca71e/logos-json-integration")
    evidence = Path(__file__).parent
    old = run('go() { return 37; }; export GOROOT_BOOTSTRAP="$(go env GOROOT)"; printf "BUILD_REACHED\\n"', root)
    fixed = run('go() { return 37; }; GOROOT_BOOTSTRAP=$(go env GOROOT); export GOROOT_BOOTSTRAP; printf "BUILD_REACHED\\n"', root)
    assert old.status == 0 and old.stdout == "BUILD_REACHED\n", old
    assert fixed.status == 37 and fixed.stdout == "", fixed
    patterns = (
        "cmd/go/scriptconds_test.go cmd/go/scriptconds_simd_logos_test.go "
        "cmd/compile/internal/ssa/zcse.go cmd/compile/internal/ssa/branchelim.go "
        "cmd/compile/internal/ssa/*logos*.go"
    )
    scalar = run(f'files="{patterns}"; printf "%s\\n" $files', root / "src")
    array = run(f'files=({patterns}); printf "%s\\n" "${{files[@]}}"', root / "src")
    assert scalar.status == array.status == 0
    assert scalar.stdout == array.stdout
    paths = tuple(array.stdout.splitlines())
    assert len(paths) == 17
    assert all((root / "src" / path).is_file() for path in paths)
    result = {
        "scope": "Fresh isolated Bash behavior probes; no compiler build or repository mutation",
        "bootstrap_lookup_failure": {"old": asdict(old), "fixed": asdict(fixed)},
        "simd_expansion": {"old": asdict(scalar), "fixed": asdict(array), "identical": True, "file_count": len(paths)},
    }
    (evidence / "fresh-shell-probes.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: masked lookup exits 0; corrected lookup exits 37 before build; SIMD expansion retains all 17 files")


if __name__ == "__main__":
    main()
