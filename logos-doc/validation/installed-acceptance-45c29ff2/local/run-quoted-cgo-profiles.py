from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path('/workspace/scratch/9949907ca71e')
REPO = ROOT / 'logos-json-integration'
OUT = ROOT / 'recovery-evidence/installed-acceptance-recovered-45c29ff2'


def resources() -> dict[str, int]:
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemAvailable', 'Shmem')}
    disk = os.statvfs(ROOT)
    home = os.statvfs('/home/agent')
    return {**memory, 'workspace_available_bytes': disk.f_bavail * disk.f_frsize, 'home_available_bytes': home.f_bavail * home.f_frsize}


def digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')


def run_phase(label: str, command: list[str], environment: dict[str, str], selected: dict[str, str], directory: Path) -> int:
    started = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    before = resources()
    assert before['MemAvailable'] >= 5 * 1024**3 // 2, before
    write_json(directory / f'{label}-inputs.json', {'started_utc': started, 'command': command, 'cwd': str(REPO / 'src'), 'environment': selected, 'absent_environment': ['GO_GCFLAGS', 'BOOT_GO_GCFLAGS'], 'resources_before': before})
    log_path = directory / f'{label}.log'
    offset = 0
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=REPO / 'src', env=environment, stdout=log, stderr=subprocess.STDOUT)
        while True:
            status = process.poll()
            with log_path.open('rb') as source:
                source.seek(offset)
                chunk = source.read()
                offset += len(chunk)
            if chunk:
                print(chunk.decode(errors='replace'), end='', flush=True)
            if status is not None:
                break
            print(json.dumps({'profile': directory.name, 'phase': label, 'elapsed_seconds': round(time.monotonic() - began, 3)}), flush=True)
            time.sleep(20)
    result = {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'duration_seconds': round(time.monotonic() - began, 3), 'exit_code': status, 'raw_log_sha256': digest(log_path), 'resources_after': resources()}
    write_json(directory / f'{label}-result.json', result)
    print(json.dumps({'profile': directory.name, 'phase': label, 'result': result}), flush=True)
    return status if status is not None and status >= 0 else 1


def main() -> int:
    spec = importlib.util.spec_from_file_location('acceptance_runner', ROOT / 'recovery-evidence/run-installed-acceptance.py')
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    validator = json.loads((OUT / 'default-validation/result.json').read_text())
    assert validator['acceptance_passed'] and validator['script_executed'] and validator['fixtures_unchanged']
    assert len(validator['unconditional_script_builds']) == 11
    build = json.loads((ROOT / 'recovery-evidence/fresh-make-45c29ff2/result.json').read_text())
    fixture = REPO / 'src/cmd/compile/internal/logos/json_cgo_quoted_test.go'
    fixture_sha = digest(fixture)
    for profile in ('jsonv2', 'nojsonv2'):
        source = runner.source_identity()
        assert digest(fixture) == fixture_sha
        for name, sha in build['installed_binary_sha256'].items():
            assert digest(REPO / name) == sha, name
        directory = OUT / f'quoted-cgo-{profile}'
        directory.mkdir()
        environment, selected = runner.make_environment()
        settings = {'GOEXPERIMENT': profile, 'CGO_ENABLED': '1'}
        environment.update(settings)
        selected.update(settings)
        assert 'GO_GCFLAGS' not in environment and 'BOOT_GO_GCFLAGS' not in environment
        binary = directory / 'logos-race.test'
        compile_command = [str(REPO / 'bin/go'), 'test', '-p=1', '-race', '-c', '-o', str(binary), 'cmd/compile/internal/logos']
        if run_phase('compile', compile_command, environment, selected, directory) != 0:
            return 1
        binary_sha = digest(binary)
        source_temp = Path(tempfile.mkdtemp(prefix=f'logos-quoted-{profile}-9949907ca71e-', dir='/home/agent'))
        assert source_temp.resolve(strict=True) == source_temp and not source_temp.is_symlink()
        execution = {**environment, 'TMPDIR': str(source_temp), 'GOTMPDIR': str(source_temp)}
        execution_selected = {**selected, 'TMPDIR': str(source_temp), 'GOTMPDIR': str(source_temp)}
        command = [str(binary), '-test.v', '-test.count=1', '-test.parallel=1', '-test.timeout=20m', '-test.run=^TestEjectQuotedCgoTypes$']
        status = run_phase('execute', command, execution, execution_selected, directory)
        text = (directory / 'execute.log').read_text(errors='replace')
        started = re.search(r'^=== RUN\s+TestEjectQuotedCgoTypes$', text, re.MULTILINE) is not None
        passed = re.search(r'^--- PASS: TestEjectQuotedCgoTypes \(', text, re.MULTILINE) is not None
        skips = re.findall(r'^\s*--- SKIP: (.+)$', text, re.MULTILINE)
        unchanged = digest(fixture) == fixture_sha and digest(binary) == binary_sha
        success = status == 0 and started and passed and not skips and unchanged
        result = {'profile': profile, 'source': source, 'source_after': runner.source_identity(), 'fixture_sha256': fixture_sha, 'fixture_and_binary_unchanged': unchanged, 'binary': str(binary), 'binary_sha256': binary_sha, 'source_temp': str(source_temp), 'source_temp_is_real_directory': True, 'test_started': started, 'test_passed': passed, 'skipped_tests': skips, 'acceptance_passed': success, 'compilation_used_disk_temp': True, 'execution_used_home_temp': True, 'cache': str(ROOT / 'postbuild-cache')}
        write_json(directory / 'result.json', result)
        print(json.dumps(result, indent=2), flush=True)
        if not success:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
