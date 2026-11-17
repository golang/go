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
REPO = ROOT / 'logos-json-integration'
EVIDENCE = ROOT / 'recovery-evidence'
RUN = EVIDENCE / 'fresh-make-45c29ff2'
BOOTSTRAP = ROOT / 'toolchains/go1.27.4-ci'
NODE = ROOT / 'toolchains/node-v26.13.1-ci'
EXPECTED_HEAD = '45c29ff215477f7e403188d7452fc79aef05d791'
EXPECTED_TREE = '32b96f3a7c289c987bc021fe5a5adbf35cfe226b'
EXPECTED_SRC = '099367e1031585fa553ef7987b199288df36256a'
MIN_MEMORY = 5 * 1024**3 // 2
MIN_DISK = 4 * 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def source_identity() -> dict[str, str]:
    return {
        'head': git('rev-parse', 'HEAD'),
        'tree': git('rev-parse', 'HEAD^{tree}'),
        'src_tree': git('rev-parse', 'HEAD:src'),
        'status_porcelain': git('status', '--porcelain=v1'),
    }


def resources() -> dict[str, int]:
    memory = {
        line.split(':')[0]: int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.split(':')[0] in ('MemTotal', 'MemAvailable', 'Shmem', 'SwapTotal')
    }
    disk = os.statvfs(ROOT)
    return {**memory, 'workspace_available_bytes': disk.f_bavail * disk.f_frsize}


def build_environment() -> tuple[dict[str, str], dict[str, str]]:
    inherited = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(('GO', 'BOOT_GO_', 'CGO_', 'LOGOS_'))
    }
    selected = {
        'GOROOT_BOOTSTRAP': str(BOOTSTRAP),
        'GOMAXPROCS': '1',
        'GOGC': '20',
        'GOMEMLIMIT': '512MiB',
        'GOFLAGS': '-p=1',
        'BOOT_GO_GCFLAGS': '-c=1',
        'GO_GCFLAGS': '-c=1',
        'GOEXPERIMENT': '',
        'GOENV': 'off',
        'GOWORK': 'off',
        'GOTOOLCHAIN': 'local',
        'GOPROXY': 'off',
        'GOCACHE': str(ROOT / 'build-cache'),
        'GOTMPDIR': str(ROOT / 'build-tmp'),
        'TMPDIR': str(ROOT / 'build-tmp'),
        'GOPATH': str(ROOT / 'build-gopath'),
        'CC': '/usr/bin/cc',
        'PATH': ':'.join((str(REPO / 'bin'), str(NODE / 'bin'), str(BOOTSTRAP / 'bin'), os.environ['PATH'])),
    }
    return {**inherited, **selected}, selected


def preflight() -> dict[str, object]:
    source = source_identity()
    assert source == {'head': EXPECTED_HEAD, 'tree': EXPECTED_TREE, 'src_tree': EXPECTED_SRC, 'status_porcelain': ''}, source
    identities = EVIDENCE / 'restored-toolchains.json'
    assert digest(identities) == '5f8090dc3d69a17f1e31cc7663294866d9f40d97c9c4a0d2268fd58e8e167031'
    assert digest(BOOTSTRAP / 'bin/go') == 'de0ae3e082ce7599eae4fb69fa2ef1e43b8f31ffd3fce62cdf81930aa9005050'
    assert digest(BOOTSTRAP / 'pkg/tool/linux_amd64/compile') == '40307e5f5c65a4b4220bbf35dbe014348780589ea2964003f0bf318fa14640e1'
    for name in ('build-cache', 'build-tmp', 'postbuild-cache'):
        directory = ROOT / name
        assert directory.is_dir() and not directory.is_symlink(), directory
        assert not any(directory.iterdir()), f'Expected empty cache/temp directory: {directory}'
    assert not RUN.exists(), 'The single build has already been launched'
    environment, selected = build_environment()
    bootstrap_version = subprocess.check_output([str(BOOTSTRAP / 'bin/go'), 'version'], env=environment, text=True).strip()
    assert bootstrap_version == 'go version go1.27.4 linux/amd64', bootstrap_version
    source_files = (
        'VERSION', 'src/make.bash', 'src/cmd/dist/main.go',
        'src/cmd/dist/util.go', 'src/cmd/dist/build.go',
        'src/cmd/dist/buildtool.go', 'src/cmd/go/internal/cfg/cfg.go',
        'src/cmd/go/internal/work/gc.go', 'src/cmd/compile/internal/base/flag.go',
        'src/cmd/compile/default.pgo',
    )
    current = resources()
    return {
        'recorded_utc': utc_now(),
        'source': source,
        'source_file_sha256': {name: digest(REPO / name) for name in source_files},
        'bootstrap_version': bootstrap_version,
        'restored_toolchain_manifest_sha256': digest(identities),
        'wrapper_sha256': digest(Path(__file__)),
        'command': ['./make.bash'],
        'working_directory': str(REPO / 'src'),
        'selected_environment': selected,
        'resources': current,
        'minimum_memory_available_bytes': MIN_MEMORY,
        'minimum_workspace_available_bytes': MIN_DISK,
        'eligible': current['MemAvailable'] >= MIN_MEMORY and current['workspace_available_bytes'] >= MIN_DISK,
        'concurrency_contract': 'GOMAXPROCS=1 bounds both cmd/go package concurrency and the dist direct scheduler to one. make and dist clear GOFLAGS; -p=1 is recorded but is not the dist cap. Initial upstream compiler defaults also follow GOMAXPROCS=1.',
        'compiler_flags_contract': 'BOOT_GO_GCFLAGS supplies -c=1 to dist direct bootstrap compilation. GO_GCFLAGS supplies -c=1 after go_bootstrap. The initial upstream toolchain build does not consume BOOT_GO_GCFLAGS.',
        'pgo_contract': 'Normal dist behavior: toolchain1 omits its profile, toolchain2 uses -pgo=off, final toolchain3 uses the checked-in default.pgo with ordinary optimization.',
        'cache_contract': 'All build temp/cache files are task-owned workspace paths. Dist keeps normal private-cache cleanup. postbuild-cache is empty and reserved for acceptance tests.',
        'monitor_contract': 'Memory/disk samples are observations. No build is killed solely because a soft resource threshold is crossed.',
    }


def run_make(preflight_record: dict[str, object]) -> int:
    assert preflight_record['eligible'], 'Resource reserve is not met'
    immediate = resources()
    if immediate['MemAvailable'] < MIN_MEMORY or immediate['workspace_available_bytes'] < MIN_DISK:
        write_json(EVIDENCE / f'make-launch-deferred-{time.time_ns()}.json', {'recorded_utc': utc_now(), 'resources': immediate})
        print(json.dumps({'launch': 'deferred', 'resources': immediate}), flush=True)
        return 75
    assert source_identity() == preflight_record['source'], 'Frozen source changed before launch'
    RUN.mkdir()
    (ROOT / 'build-gopath').mkdir(exist_ok=True)
    write_json(RUN / 'inputs.json', {**preflight_record, 'resources_immediately_before_launch': immediate})
    environment, _ = build_environment()
    start_utc = utc_now()
    start = time.monotonic()
    offset = 0
    log_path = RUN / 'make.log'
    with log_path.open('xb') as raw_log, (RUN / 'resources.jsonl').open('x') as samples:
        process = subprocess.Popen(['./make.bash'], cwd=REPO / 'src', env=environment, stdout=raw_log, stderr=subprocess.STDOUT)
        write_json(RUN / 'started.json', {'started_utc': start_utc, 'pid': process.pid, 'source_head': EXPECTED_HEAD})
        while True:
            status = process.poll()
            sample = {'recorded_utc': utc_now(), 'elapsed_seconds': round(time.monotonic() - start, 3), **resources()}
            samples.write(json.dumps(sample) + '\n')
            samples.flush()
            with log_path.open('rb') as reader:
                reader.seek(offset)
                chunk = reader.read()
                offset += len(chunk)
            if chunk:
                print(chunk.decode(errors='replace'), end='', flush=True)
            print(json.dumps({'monitor': sample}), flush=True)
            if status is not None:
                break
            time.sleep(30)
    end_utc = utc_now()
    output: dict[str, object] = {
        'started_utc': start_utc,
        'completed_utc': end_utc,
        'duration_seconds': round(time.monotonic() - start, 3),
        'make_exit_code': status,
        'source_before': preflight_record['source'],
        'source_after': source_identity(),
        'raw_log': str(log_path),
        'raw_log_sha256': digest(log_path),
        'postbuild_cache_still_empty': not any((ROOT / 'postbuild-cache').iterdir()),
    }
    binaries = ('bin/go', 'bin/logos', 'pkg/tool/linux_amd64/compile', 'pkg/tool/linux_amd64/link', 'pkg/tool/linux_amd64/asm', 'pkg/tool/linux_amd64/cgo')
    output['installed_binary_sha256'] = {name: digest(REPO / name) for name in binaries if (REPO / name).is_file()}
    if status == 0:
        commands = {
            'go_version': [str(REPO / 'bin/go'), 'version'],
            'logos_version': [str(REPO / 'bin/logos'), 'version'],
            'compiler_version': [str(REPO / 'pkg/tool/linux_amd64/compile'), '-V=full'],
            'inferred_goroot': [str(REPO / 'bin/go'), 'env', 'GOROOT'],
        }
        output['installed_versions'] = {name: subprocess.check_output(command, env=environment, text=True).strip() for name, command in commands.items()}
    write_json(RUN / 'result.json', output)
    print(json.dumps(output, indent=2), flush=True)
    if output['source_after'] != output['source_before'] or not output['postbuild_cache_still_empty']:
        return 2
    return status if status is not None and status >= 0 else 1


def main() -> int:
    if sys.argv[1:] not in (['preflight'], ['run']):
        raise SystemExit('Use preflight or run')
    record = preflight()
    write_json(EVIDENCE / f'make-preflight-{time.time_ns()}.json', record)
    print(json.dumps(record, indent=2), flush=True)
    if not record['eligible']:
        return 75
    return run_make(record) if sys.argv[1:] == ['run'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
