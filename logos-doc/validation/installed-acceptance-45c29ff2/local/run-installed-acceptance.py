from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/workspace/scratch/9949907ca71e')
REPO = ROOT / 'logos-json-integration'
EVIDENCE = ROOT / 'recovery-evidence'
BUILD = EVIDENCE / 'fresh-make-45c29ff2'
OUT = EVIDENCE / 'installed-acceptance-45c29ff2'
HEAD = '45c29ff215477f7e403188d7452fc79aef05d791'
TREE = '32b96f3a7c289c987bc021fe5a5adbf35cfe226b'
SRC = '099367e1031585fa553ef7987b199288df36256a'


def digest(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')
        output.flush()


def source_identity() -> dict[str, str]:
    commands = {'head': ['rev-parse', 'HEAD'], 'tree': ['rev-parse', 'HEAD^{tree}'], 'src': ['rev-parse', 'HEAD:src'], 'status': ['status', '--porcelain=v1']}
    result = {name: subprocess.check_output(['git', *args], cwd=REPO, text=True).strip() for name, args in commands.items()}
    assert result == {'head': HEAD, 'tree': TREE, 'src': SRC, 'status': ''}, result
    return result


def make_environment() -> tuple[dict[str, str], dict[str, str]]:
    inherited = {key: value for key, value in os.environ.items() if not key.startswith(('GO', 'BOOT_GO_', 'CGO_', 'LOGOS_'))}
    selected = {
        'GOROOT': str(REPO), 'GOMAXPROCS': '1', 'GOFLAGS': '-p=1',
        'GOGC': '20', 'GOMEMLIMIT': '512MiB', 'GOTOOLCHAIN': 'local',
        'GOENV': 'off', 'GOWORK': 'off', 'GOPROXY': 'off', 'GOEXPERIMENT': '',
        'GOCACHE': str(ROOT / 'postbuild-cache'), 'GOPATH': str(ROOT / 'acceptance-gopath'),
        'GOTMPDIR': str(ROOT / 'acceptance-tmp'), 'TMPDIR': str(ROOT / 'acceptance-tmp'),
        'UPROOT1272': str(ROOT / 'toolchains/go1.27.2'),
        'UPROOT1273': str(ROOT / 'toolchains/go1.27.3'),
        'UPROOT1274': str(ROOT / 'toolchains/go1.27.4-ci'),
        'CC': '/usr/bin/cc',
        'PATH': ':'.join((str(REPO / 'bin'), str(REPO / 'lib/wasm'), str(ROOT / 'toolchains/node-v26.13.1-ci/bin'), str(ROOT / 'toolchains/qemu-10.0.13/usr/bin'), os.environ['PATH'])),
    }
    environment = {**inherited, **selected}
    assert 'GO_GCFLAGS' not in environment and 'BOOT_GO_GCFLAGS' not in environment
    return environment, selected


def expected_import_cells() -> set[str]:
    symbols = ('local·1', 'hidden·23', 'Box.Method', 'Box[go.shape.int]')
    operations = ('functions', 'readonly', 'empty functions', 'empty readonly', 'declaration clone', 'structural clone', 'readonly symbol')
    libraries = ('generic method', 'generic function', 'nested two parameters', 'nongeneric local control', 'package generic control')
    modes = ('go', 'logos go', 'logos interop', 'logos overlay')
    cells = {f'TestFactScopeLeavesCompilerSymbolsLazy/{symbol}/{operation}'.replace(' ', '_') for symbol in symbols for operation in operations}
    cells.update(f'TestCompileImportedLocalType/{library}/{mode}'.replace(' ', '_') for library in libraries for mode in modes)
    assert len(cells) == 48
    return cells


def run_gate(label: str, args: list[str], overrides: dict[str, str], environment: dict[str, str], selected: dict[str, str], fixture_hashes: dict[str, str]) -> bool:
    source = source_identity()
    assert all(digest(REPO / path) == value for path, value in fixture_hashes.items())
    command = [str(REPO / 'bin/go'), 'test', '-p=1', '-count=1', '-parallel=1', *args]
    gate_env = {**environment, **overrides}
    gate = OUT / label
    gate.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    write_json(gate / 'inputs.json', {'started_utc': started, 'source': source, 'command': command, 'cwd': str(REPO / 'src'), 'environment': {**selected, **overrides}, 'absent_environment': ['GO_GCFLAGS', 'BOOT_GO_GCFLAGS'], 'fixture_sha256': fixture_hashes})
    raw = gate / 'raw.log'
    offset = 0
    with raw.open('xb') as log:
        process = subprocess.Popen(command, cwd=REPO / 'src', env=gate_env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            code = process.poll()
            with raw.open('rb') as reader:
                reader.seek(offset)
                chunk = reader.read()
                offset += len(chunk)
            if chunk:
                print(chunk.decode(errors='replace'), end='', flush=True)
            if code is not None:
                break
            print(json.dumps({'gate': label, 'elapsed_seconds': round(time.monotonic() - began, 3)}), flush=True)
            time.sleep(20)
    text = raw.read_text(errors='replace')
    passed = set(re.findall(r'^\s*--- PASS: (\S+) \(', text, re.MULTILINE))
    skipped = re.findall(r'^\s*--- SKIP: (.+)$', text, re.MULTILINE)
    if label == 'imported-local-types':
        required = expected_import_cells() | {'TestFactScopeLeavesCompilerSymbolsLazy', 'TestCompileImportedLocalType'}
    elif label == 'default-validation':
        required = {'TestScript', 'TestScript/logos_struct_default_validation'}
    else:
        required = {'TestEjectQuotedCgoTypes'}
    missing = sorted(required - passed)
    success = code == 0 and not skipped and not missing
    result = {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'duration_seconds': round(time.monotonic() - began, 3), 'exit_code': code, 'executed_pass_names': sorted(passed), 'missing_pass_names': missing, 'skipped_tests': skipped, 'acceptance_passed': success, 'raw_log_sha256': digest(raw), 'source_after': source_identity(), 'fixtures_unchanged': all(digest(REPO / path) == value for path, value in fixture_hashes.items())}
    if label == 'imported-local-types':
        result['frozen_cells_passed'] = len(expected_import_cells() & passed)
    if label == 'default-validation':
        body = (REPO / 'src/cmd/go/testdata/script/logos_struct_default_validation.txt').read_text().split('-- go.mod --', 1)[0]
        builds = [line for line in body.splitlines() if re.match(r'^!?\s*logos build ', line)]
        assert len(builds) == 11
        result['unconditional_script_builds'] = builds
        result['script_executed'] = 'TestScript/logos_struct_default_validation' in passed
    write_json(gate / 'result.json', result)
    print(json.dumps({'gate': label, 'result': result}, indent=2), flush=True)
    return success and bool(result['fixtures_unchanged'])


def main() -> int:
    build = json.loads((BUILD / 'result.json').read_text())
    assert build['make_exit_code'] == 0 and build['postbuild_cache_still_empty'], build
    source_identity()
    for name, value in build['installed_binary_sha256'].items():
        assert digest(REPO / name) == value, name
    assert not any((ROOT / 'postbuild-cache').iterdir()), 'Fork acceptance cache must start empty'
    assert not OUT.exists(), 'This acceptance queue has already been invoked'
    ancestry = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True, capture_output=True)
    assert ancestry.returncode == 128, ancestry.stdout
    for name in ('acceptance-tmp', 'acceptance-gopath'):
        (ROOT / name).mkdir(exist_ok=True)
    OUT.mkdir()
    fixture_paths = ('src/cmd/compile/internal/logos/compile_local_type_import_test.go', 'src/cmd/compile/internal/logos/optfacts_scope_test.go', 'src/cmd/go/testdata/script/logos_struct_default_validation.txt', 'src/cmd/compile/internal/logos/json_cgo_quoted_test.go')
    fixture_hashes = {path: digest(REPO / path) for path in fixture_paths}
    environment, selected = make_environment()
    write_json(OUT / 'queue.json', {'source': source_identity(), 'build_result_sha256': digest(BUILD / 'result.json'), 'wrapper_sha256': digest(Path(__file__)), 'initial_postbuild_cache_empty': True, 'temporary_root_git_check_exit': ancestry.returncode, 'fixture_sha256': fixture_hashes, 'expected_import_cells': sorted(expected_import_cells())})
    gates = (
        ('imported-local-types', ['-short', '-timeout=20m', '-v', 'cmd/compile/internal/logos', '-run', '^(TestFactScopeLeavesCompilerSymbolsLazy|TestCompileImportedLocalType)$'], {}),
        ('default-validation', ['-race', '-timeout=20m', '-v', 'cmd/go', '-run', '^TestScript$/^logos_struct_default_validation$'], {}),
        ('quoted-cgo-jsonv2', ['-race', '-timeout=20m', '-v', 'cmd/compile/internal/logos', '-run', '^TestEjectQuotedCgoTypes$'], {'GOEXPERIMENT': 'jsonv2', 'CGO_ENABLED': '1'}),
        ('quoted-cgo-nojsonv2', ['-race', '-timeout=20m', '-v', 'cmd/compile/internal/logos', '-run', '^TestEjectQuotedCgoTypes$'], {'GOEXPERIMENT': 'nojsonv2', 'CGO_ENABLED': '1'}),
    )
    for label, args, overrides in gates:
        if not run_gate(label, args, overrides, environment, selected, fixture_hashes):
            print(f'Queue stopped after {label}; no repair or retry performed.', flush=True)
            return 1
    print('All four authorized acceptance gates passed.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
