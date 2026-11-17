from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path('/workspace/scratch/9949907ca71e')
REPO = ROOT / 'logos-acceptance-fixes'
AUDIT = ROOT / 'recovery-evidence/final-metadata-audit-33752c03'
SOURCE = '33752c03819ed5afcb37f87633a3cac668e181a8'
BASE = '2535486fe9596a9212d525a5007963ef9e8682f0'


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=REPO, text=True,
                          capture_output=True, check=True).stdout.rstrip('\n')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash_list(path: Path, cwd: Path) -> int:
    count = 0
    for line in path.read_text().splitlines():
        if not line:
            continue
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](.*)', line)
        assert match is not None, (path, line)
        expected, name = match.groups()
        assert sha(cwd / name) == expected, (path, name)
        count += 1
    return count


def check_file_set(directory: Path, manifest: Path) -> int:
    data = json.loads(manifest.read_bytes())
    listed = data['files']
    assert isinstance(listed, dict), manifest
    for name, entry in listed.items():
        path = directory / name
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == entry['sha256'], path
        assert path.stat().st_size == entry['bytes'], path
    return len(listed)


def main() -> None:
    inputs = json.loads((AUDIT / 'inputs.json').read_bytes())
    for name, expected in inputs['files'].items():
        assert sha(REPO / name) == expected['sha256'], name
    assert git('rev-parse', 'HEAD') == inputs['head']
    assert git('rev-parse', 'HEAD:src') == SOURCE
    assert git('diff', '--name-only', 'HEAD', '--', 'src') == ''

    plan = (REPO / 'logos-doc/IMPLEMENTATION.md').read_text()
    ledger_section = plan.split('## Autoplan ledger\n', 1)[1]
    embedded = re.search(r'```jsonl\n(.*?)\n```', ledger_section, re.S)
    assert embedded is not None
    file_text = (REPO / 'tasks/autoplan-ledger.jsonl').read_text()
    assert embedded.group(1) + '\n' == file_text
    tasks = [json.loads(line) for line in file_text.splitlines() if line]
    assert len(tasks) == len({row['id'] for row in tasks}) == 306
    baseline_tasks = [json.loads(line) for line in git('show', BASE + ':tasks/autoplan-ledger.jsonl').splitlines() if line]
    assert tasks[:-4] == baseline_tasks
    assert [row['id'] for row in tasks[-4:]] == ['AUD-210', 'AUD-211', 'AUD-212', 'AUD-213']

    rows: list[list[str]] = []
    for line in (REPO / 'logos-doc/REDTEAM.md').read_text().splitlines():
        if not line.startswith('| #'):
            continue
        cells = line.removeprefix('| ').removesuffix(' |').split(' | ')
        assert len(cells) == 5, line
        rows.append(cells)
    assert len(rows) == len({row[0] for row in rows}) == 350
    statuses = Counter(row[4] for row in rows)
    assert statuses == {'tested': 341, 'satisfied upstream': 2, 'blocked': 7}
    blocked = [row[0] for row in rows if row[4] == 'blocked']
    assert blocked == ['#216', '#217', '#218', '#226', '#403', '#404', '#405']
    path_pattern = re.compile(r'(?:src|test|lib)/[A-Za-z0-9_./+-]*[A-Za-z0-9_]')
    path_count = 0
    for row in rows:
        if row[4] in ('tested', 'satisfied upstream'):
            paths = path_pattern.findall(row[3])
            assert paths, row[0]
            for name in paths:
                assert (REPO / name).exists() or list(REPO.glob(name + '.*')), (row[0], name)
            path_count += len(paths)
        else:
            assert 'Mechanism:' in row[3] and 'Gate:' in row[3], row[0]

    request_path = REPO / 'logos-doc/validation/release-acceptance-request.json'
    request = json.loads(request_path.read_bytes())
    old_request = json.loads(git('show', BASE + ':logos-doc/validation/release-acceptance-request.json'))
    assert request['src_tree'] == SOURCE
    assert request['frozen_inputs'] == old_request['frozen_inputs']
    assert request['scope'] == old_request['scope']
    assert len(request['frozen_inputs']) == 6
    for name, expected in request['frozen_inputs'].items():
        assert sha(REPO / name) == expected, name
    patch = REPO / 'logos-doc/blocked/M6-S3b.patch'
    original_patch = subprocess.run(['git', 'show', BASE + ':logos-doc/blocked/M6-S3b.patch'], cwd=REPO, capture_output=True, check=True).stdout
    assert patch.read_bytes() == original_patch

    integration = json.loads((ROOT / 'recovery-evidence/acceptance-failure-repairs/integration-source-001.json').read_bytes())
    repaired = {name: expected for group in integration['groups'].values() for name, expected in group['files_sha256'].items()}
    assert len(repaired) == 12
    for name, expected in repaired.items():
        assert sha(REPO / name) == expected, name

    installed = REPO / 'logos-doc/validation/installed-acceptance-33752c03'
    portable = ROOT / 'recovery-evidence/installed-acceptance-33752c03-portable'
    installed_files = {str(path.relative_to(installed)) for path in installed.rglob('*') if path.is_file()}
    portable_files = {str(path.relative_to(portable)) for path in portable.rglob('*') if path.is_file()}
    assert installed_files == portable_files and len(installed_files) == 152
    for name in installed_files:
        assert (installed / name).read_bytes() == (portable / name).read_bytes(), name
        assert not (installed / name).is_symlink(), name
    assert sha(installed / 'manifest.json') == '6e4ce2e3a1fe0c13b66fd18b7493989d4c3829d128557b9a24c312dffeeb07aa'
    assert check_file_set(installed, installed / 'manifest.json') == 151
    assert check_hash_list(installed / 'files.sha256', installed) == 150

    components: dict[str, dict[str, int | str]] = {}
    for name in ('alias', 'projection', 'environment', 'parity'):
        directory = REPO / 'logos-doc/validation/acceptance-repairs-33752c03' / name
        listing = directory / 'files.sha256'
        components[name] = {'checksum_count': check_hash_list(listing, directory), 'checksum_list_sha256': sha(listing)}
    active_manifests: dict[str, int] = {}
    for name, cwd in [
        ('logos-doc/validation/compiler-stability/frozen-tests-v2.sha256', REPO),
        ('logos-doc/validation/release-acceptance/frozen-checker.sha256', REPO / 'logos-doc/validation/release-acceptance'),
    ]:
        active_manifests[name] = check_hash_list(REPO / name, cwd)

    links: list[str] = []
    for relative in (
        'logos-doc/validation/stabilization-candidate.md',
        'logos-doc/validation/acceptance-repairs-33752c03/README.md',
        'logos-doc/validation/installed-acceptance-33752c03/README.md',
    ):
        path = REPO / relative
        for target in re.findall(r'\]\(([^\s)]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            local = target.split('#', 1)[0]
            assert (path.parent / local).exists(), (relative, target)
            links.append(relative + ':' + target)

    for name, expected in inputs['files'].items():
        assert sha(REPO / name) == expected['sha256'], name
    result = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'head': git('rev-parse', 'HEAD'),
        'source_tree': SOURCE,
        'task_count': len(tasks),
        'embedded_ledger_byte_equal': True,
        'prior_task_records_unchanged': True,
        'attack_count': len(rows),
        'attack_status_counts': dict(statuses),
        'blocked_rows': blocked,
        'proving_path_occurrences_checked': path_count,
        'request_frozen_inputs_verified': 6,
        'm6_blocked_patch_unchanged': True,
        'repaired_file_hashes_verified': 12,
        'installed_packet_files_byte_equal': 152,
        'component_checksums': components,
        'active_manifests': active_manifests,
        'local_markdown_links_checked': len(links),
        'frozen_audit_inputs_unchanged': True,
        'status': 'PASS',
        'scope': 'File and metadata checks only. No Go test or hosted execution inferred.',
    }
    output = AUDIT / 'metadata-result.json'
    assert not output.exists(), output
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
