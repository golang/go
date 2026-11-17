from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


def main(root: Path) -> int:
    failures: list[str] = []
    run_blocks: int = 0
    workflows: list[Path] = sorted(
        [*root.glob(".github/workflows/*.yml"), *root.glob(".github/workflows/*.yaml")]
    )
    paths: list[Path] = [root / ".github/actionlint.yaml", *workflows]
    for path in paths:
        document: object = yaml.safe_load(path.read_text())
        assert isinstance(document, dict), f"expected YAML mapping: {path}"
        if path not in workflows:
            continue
        jobs: object = document.get("jobs")
        assert isinstance(jobs, dict), f"expected jobs mapping: {path}"
        for job_name, job in jobs.items():
            assert isinstance(job, dict), f"expected job mapping: {job_name}"
            steps: object = job.get("steps", [])
            assert isinstance(steps, list), f"expected steps list: {job_name}"
            for index, step in enumerate(steps):
                assert isinstance(step, dict), f"expected step mapping: {index}"
                if "run" not in step:
                    continue
                script: object = step["run"]
                assert isinstance(script, str), f"expected run string: {path}"
                run_blocks += 1
                result = subprocess.run(
                    ["/usr/bin/bash", "-n"], input=script, text=True,
                    capture_output=True, check=False,
                )
                if result.returncode:
                    failures.append(f"{path.relative_to(root)}:{job_name}:{index}: {result.stderr}")
    scripts: tuple[str, str] = (
        ".github/scripts/install-ci-tool.sh",
        ".github/scripts/install-upstream-oracles.sh",
    )
    for script_path in scripts:
        result = subprocess.run(
            ["/usr/bin/bash", "-n", str(root / script_path)],
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            failures.append(f"{script_path}: {result.stderr}")
    if (len(paths), run_blocks, len(scripts)) != (8, 67, 2):
        failures.append("input coverage differs from the frozen 8/67/2 inventory")
    print(json.dumps({
        "yaml_documents": len(paths), "run_blocks": run_blocks,
        "standalone_scripts": len(scripts), "failures": failures,
    }, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
