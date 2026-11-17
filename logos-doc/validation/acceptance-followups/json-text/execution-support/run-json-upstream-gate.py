from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


ROOT = Path('/workspace/scratch/9949907ca71e')
REPO = ROOT / 'logos-json-integration'
EVIDENCE = ROOT / 'recovery-evidence/upstream-e5b95a2e-comparison'
HEAD = '45c29ff215477f7e403188d7452fc79aef05d791'
TREE = '32b96f3a7c289c987bc021fe5a5adbf35cfe226b'
SRC = '099367e1031585fa553ef7987b199288df36256a'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path: Path, value: object) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def source_identity() -> dict[str, str]:
    result = {
        name: subprocess.check_output(['git', '--no-optional-locks', 'rev-parse', query], cwd=REPO, text=True).strip()
        for name, query in {'head': 'HEAD', 'tree': 'HEAD^{tree}', 'src': 'HEAD:src'}.items()
    }
    result['status'] = subprocess.check_output(['git', '--no-optional-locks', 'status', '--porcelain=v1'], cwd=REPO, text=True)
    assert result == {'head': HEAD, 'tree': TREE, 'src': SRC, 'status': ''}, result
    return result


def resources() -> dict[str, int]:
    memory = {
        line.split(':')[0]: int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.split(':')[0] in ('MemAvailable', 'Shmem')
    }
    disk = os.statvfs(ROOT)
    return {**memory, 'workspace_available_bytes': disk.f_bavail * disk.f_frsize}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--profile', choices=('jsonv2', 'nojsonv2'), required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--run', default='.')
    parser.add_argument('--vet', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.label), args.label
    command = [str(REPO / 'bin/go')]
    if args.vet:
        command += ['vet', '-p=1', '-overlay=' + str(args.overlay), 'logos/rt/json']
    else:
        command += ['test', '-p=1', '-count=1', '-parallel=1', '-race', '-timeout=20m', '-v', '-overlay=' + str(args.overlay), 'logos/rt/json', '-run', args.run]
    if not args.execute:
        print(json.dumps({'command': command, 'profile': args.profile, 'cwd': str(REPO / 'src'), 'executed': False}, indent=2))
        return 0

    source = source_identity()
    out = EVIDENCE / 'gates' / args.label
    out.mkdir(parents=True, exist_ok=False)
    overlay = json.loads(args.overlay.read_text())
    mapped = {path: {'replacement': replacement, 'sha256': digest(Path(replacement))} for path, replacement in overlay['Replace'].items()}
    assert all(Path(path).is_relative_to(REPO / 'src/logos/rt/json') for path in mapped)
    build = json.loads((ROOT / 'recovery-evidence/fresh-make-45c29ff2/result.json').read_text())
    assert build['make_exit_code'] == 0
    for path, expected in build['installed_binary_sha256'].items():
        assert digest(REPO / path) == expected, path
    temp = ROOT / 'json-regression-tmp'
    temp.mkdir(exist_ok=True)
    source_temp = Path(tempfile.mkdtemp(prefix='logos-json-regression-9949907ca71e-', dir='/home/agent'))
    assert source_temp.resolve(strict=True) == source_temp and not source_temp.is_symlink()
    markers = [str(parent / marker) for parent in (source_temp, *source_temp.parents) for marker in ('.git', '.hg', '.svn') if (parent / marker).exists()]
    assert not markers, markers
    selected = {
        'GOROOT': str(REPO), 'GOMAXPROCS': '1', 'GOFLAGS': '-p=1',
        'GOGC': '20', 'GOMEMLIMIT': '512MiB', 'GOTOOLCHAIN': 'local',
        'GOENV': 'off', 'GOWORK': 'off', 'GOPROXY': 'off',
        'GOEXPERIMENT': args.profile, 'GOCACHE': str(ROOT / 'postbuild-cache'),
        'GOPATH': str(ROOT / 'acceptance-gopath'), 'CC': '/usr/bin/cc',
        'GOTMPDIR': str(temp), 'TMPDIR': str(source_temp), 'PWD': str(REPO / 'src'),
        'PATH': str(REPO / 'bin') + os.pathsep + os.environ['PATH'],
    }
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(('GO', 'BOOT_GO_', 'CGO_', 'LOGOS_'))
    }
    environment.update(selected)
    assert all(key not in environment for key in ('GO_GCFLAGS', 'BOOT_GO_GCFLAGS', 'CGO_ENABLED'))
    before = resources()
    started = datetime.now(timezone.utc).isoformat()
    record(out / 'inputs.json', {
        'source': source, 'command': command, 'cwd': str(REPO / 'src'),
        'environment': selected, 'overlay': str(args.overlay),
        'overlay_sha256': digest(args.overlay), 'mapped_file_identities': mapped,
        'wrapper_sha256': digest(Path(__file__)),
        'installed_binary_sha256': build['installed_binary_sha256'],
        'source_temp_is_real': True, 'source_temp_ancestor_vcs_markers': markers,
        'started_utc': started, 'resources_before': before,
    })
    if before['MemAvailable'] < 5 * 1024**3 // 2 or before['workspace_available_bytes'] < 2 * 1024**3:
        record(out / 'result.json', {'started': False, 'reason': 'resource reserve not met'})
        return 2
    began = time.monotonic()
    raw = out / 'raw.log'
    with raw.open('xb') as log:
        process = subprocess.Popen(command, cwd=REPO / 'src', env=environment, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                status = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                print(json.dumps({'label': args.label, 'elapsed_seconds': round(time.monotonic() - began, 3), 'resources': resources()}), flush=True)
    text = raw.read_text(errors='replace')
    unchanged = all(digest(Path(item['replacement'])) == item['sha256'] for item in mapped.values())
    result = {
        'started': True, 'started_utc': started,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': round(time.monotonic() - began, 3),
        'exit_code': status, 'raw_log_sha256': digest(raw),
        'source_after': source_identity(), 'mapped_files_unchanged': unchanged,
        'passed_tests': re.findall(r'^\s*--- PASS: (\S+) \(', text, re.MULTILINE),
        'failed_tests': re.findall(r'^\s*--- FAIL: (\S+) \(', text, re.MULTILINE),
        'skipped_tests': re.findall(r'^\s*--- SKIP: (.+)$', text, re.MULTILINE),
        'resources_after': resources(),
    }
    record(out / 'result.json', result)
    print(text, end='', flush=True)
    print(json.dumps({'label': args.label, 'result': result}, indent=2), flush=True)
    assert unchanged
    return status if status >= 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
