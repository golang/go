import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import path from 'node:path';

const repository = process.argv[2] ?? '/workspace/scratch/9949907ca71e/logos-acceptance-fixes';
const base = '45c29ff215477f7e403188d7452fc79aef05d791';
const prefix = 'src/cmd/go/internal/logos/eject/';
const files = ['eject.go', 'eject_test.go', 'variant_features_test.go', 'variant_host_tags_test.go', 'variant_tags_test.go'];
const git = (...args) => execFileSync('git', args, { cwd: repository, encoding: 'utf8' });
const sha256 = content => createHash('sha256').update(content).digest('hex');
const before = file => git('show', base + ':' + prefix + file);
const after = file => readFileSync(path.join(repository, prefix, file), 'utf8');
const originalAdapter = 'func findVariantContextForFile(pkgDir, name, hostGOOS, hostGOARCH string, hostTags []string) (string, string, []string, bool) {\n\tvc, ok := selectVariantContext(pkgDir, name, hostGOOS, hostGOARCH, hostTags)\n\treturn vc.goos, vc.goarch, vc.tags, ok\n}\n\n';
const testAdapter = 'func findVariantContextFromLinuxAMD64(pkgDir, name string, hostTags []string) (string, string, []string, bool) {\n\tvc, ok := selectVariantContext(pkgDir, name, "linux", "amd64", hostTags)\n\treturn vc.goos, vc.goarch, vc.tags, ok\n}\n\n';
const originalCleanup = '\t\t\t\tt.Cleanup(func() { os.RemoveAll(out) })';
const checkedCleanup = '\t\t\t\tt.Cleanup(func() {\n\t\t\t\t\tif err := os.RemoveAll(out); err != nil {\n\t\t\t\t\t\tt.Errorf("remove variant output %s: %v", out, err)\n\t\t\t\t\t}\n\t\t\t\t})';
const replaceOnce = (source, oldText, newText) => {
  assert.equal(source.split(oldText).length, 2, 'replacement must identify exactly one span');
  return source.replace(oldText, newText);
};

let expectedProduction = replaceOnce(before('eject.go'), '// findVariantContextForFile finds', '// selectVariantContext finds');
expectedProduction = replaceOnce(expectedProduction, originalAdapter, '');
assert.equal(after('eject.go'), expectedProduction, 'production changes exceed removing the test-only adapter and moving its comment');

const transformations = [];
let calls = 0;
for (const file of files.slice(1)) {
  const original = before(file);
  const countBefore = (original.match(/findVariantContextForFile\(/g) ?? []).length;
  let count = 0;
  let expected = original.replace(/findVariantContextForFile\(([^\n]*?), "linux", "amd64", /g, (_, args) => {
    calls++;
    count++;
    return 'findVariantContextFromLinuxAMD64(' + args + ', ';
  });
  assert.equal(count, countBefore, 'all adapter calls must preserve the exact Linux/amd64 host');
  if (file === 'eject_test.go') {
    expected = replaceOnce(expected, 'func TestEmitPackagePlatformFiles', testAdapter + 'func TestEmitPackagePlatformFiles');
  }
  if (file === 'variant_features_test.go') {
    expected = replaceOnce(expected, originalCleanup, checkedCleanup);
  }
  assert.equal(after(file), expected, file + ': an input, assertion, or other byte changed beyond the exact allowed transformations');
  transformations.push({ file: prefix + file, equivalentCallRewrites: count, existingInputsAndAssertionsPreserved: true });
}
assert.equal(calls, 11);
assert.deepEqual(git('diff', '--name-only', '--', prefix).trim().split('\n'), files.map(file => prefix + file));

const hashes = files.map(file => ({ file: prefix + file, oldSHA256: sha256(before(file)), newSHA256: sha256(after(file)) }));
const tracked = git('ls-tree', '-r', '--name-only', base).trim().split('\n');
const manifests = [];
for (const file of tracked.filter(file => /(?:\.sha256$|manifest|frozen.*\.(?:txt|json)$)/i.test(file))) {
  const original = git('show', base + ':' + file);
  const entries = original.split('\n').flatMap((line, index) => {
    const affected = hashes.filter(record => line.includes(record.file) || line.includes(record.oldSHA256));
    return affected.length ? [{ line: index + 1, text: line, affected: affected.map(record => record.file) }] : [];
  });
  if (entries.length === 0) continue;
  const current = readFileSync(path.join(repository, file), 'utf8');
  assert.equal(current, original, 'historical manifest was changed: ' + file);
  manifests.push({ file, SHA256: sha256(original), unchanged: true, entries });
}
assert.deepEqual(manifests.map(manifest => manifest.file), [
  'logos-doc/validation/compiler-stability/frozen-tests.sha256',
  'logos-doc/validation/ejection-target-facts/frozen-tests.sha256',
]);

const result = {
  base, repository, candidateHEAD: git('rev-parse', 'HEAD').trim(),
  ownedDiffSHA256: sha256(git('diff', '--binary', '--', prefix)),
  compilerSelectorAndAllOtherProductionBytesUnchanged: true,
  equivalentCallRewrites: calls,
  transformations, hashes, manifests,
  inventoryScope: 'All manifest-named and SHA256 files tracked at the exact base; supplemental searches covered task evidence, publication, recovery, and prerequisite directories without reading caches.',
  supplementalMatches: 'Only hosted diagnostic metadata and the language lane preflight log name changed tests; no additional frozen hash manifest was found.',
  activeGate: '.github/workflows/compiler-stability.yml:80 checks the historical compiler-stability manifest. A versioned successor must be explicit; this proof does not edit it.',
  historicalManifestCommits: [
    { commit: '53f21cd963662c0ba05d651dbefdab473f74893a', manifest: 'logos-doc/validation/compiler-stability/frozen-tests.sha256' },
    { commit: '3aeea9a8ab10e2c2065fac5e02a22c1583ab1ad4', manifest: 'logos-doc/validation/ejection-target-facts/frozen-tests.sha256' },
    { commit: '57bd5dddee23f8d4acc1e9549064bc96aebfde8b', manifest: 'logos-doc/validation/ejection-target-facts/frozen-tests.sha256' },
  ],
  goCompilationTestsAndLintExecuted: false,
};
console.log(JSON.stringify(result, null, 2));
