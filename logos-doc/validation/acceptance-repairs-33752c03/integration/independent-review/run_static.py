from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/workspace/scratch/9949907ca71e')
REPO = ROOT / 'logos-acceptance-fixes'
AUDIT = ROOT / 'recovery-evidence/final-metadata-audit-33752c03'
TOOLS = ROOT / 'language-ci-tools/bin'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    name = sys.argv[1]
    if name == 'contract':
        inputs = [REPO / 'logos-doc/validation/release-acceptance/check_release_contract.py',
                  REPO / '.github/workflows/release-acceptance.yml',
                  REPO / '.github/scripts/release-acceptance.sh',
                  REPO / 'logos-doc/validation/release-acceptance-request.json']
        command = [sys.executable, str(inputs[0]), str(REPO)]
    elif name == 'actionlint':
        inputs = sorted((REPO / '.github/workflows').glob('*.yml'))
        command = [str(TOOLS / 'actionlint'), '-oneline', '-shellcheck', str(TOOLS / 'shellcheck'), *map(str, inputs)]
    elif name == 'shellcheck':
        inputs = sorted((REPO / '.github/scripts').glob('*.sh'))
        command = [str(TOOLS / 'shellcheck'), '--shell=bash', '-x', *map(str, inputs)]
    else:
        raise ValueError(f'unsupported static check: {name}')
    before = {str(path.relative_to(REPO)): sha(path) for path in inputs}
    output = AUDIT / name
    output.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    environment = os.environ.copy()
    environment['PATH'] = str(TOOLS) + os.pathsep + environment['PATH']
    completed = subprocess.run(command, cwd=REPO, env=environment, capture_output=True)
    (output / 'stdout.log').write_bytes(completed.stdout)
    (output / 'stderr.log').write_bytes(completed.stderr)
    after = {str(path.relative_to(REPO)): sha(path) for path in inputs}
    assert before == after
    result = {
        'command': command,
        'cwd': str(REPO),
        'started_utc': started,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': round(time.monotonic() - clock, 3),
        'exit_code': completed.returncode,
        'input_sha256': before,
        'executable_sha256': sha(Path(command[0])),
        'stdout_sha256': sha(output / 'stdout.log'),
        'stderr_sha256': sha(output / 'stderr.log'),
        'input_hashes_unchanged': True,
        'scope': 'Static validation only; no Go compilation, Go test or hosted execution.',
    }
    if name == 'actionlint':
        result['shellcheck_executable_sha256'] = sha(TOOLS / 'shellcheck')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'check': name, 'exit_code': completed.returncode,
                      'input_count': len(inputs), 'result': str(output / 'result.json')}, indent=2))
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
