import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync, spawn } from 'node:child_process';
import { performance } from 'node:perf_hooks';

const base = '/workspace/scratch/9949907ca71e';
const evidence = path.dirname(new URL(import.meta.url).pathname);
const identityPath = `${base}/recovery-evidence/final-acceptance-tools/identity-e96a570d.json`;
const identityHash = 'a52fd73da636ff7b668f8652af4f2e4de672b0e3b79b9b9014d0bccae95ef262';
const receiptPath = `${base}/recovery-evidence/array-index-acceptance/runs/e96-final-001/guarded-enum-race/retained-binary.json`;
const expectedDriverHash = '9645693e7edb1d9b4a1a02c0a74b158283db73158189b62c0c4feb45b3103f0a';
const qemu = `${base}/toolchains/qemu-10.0.13/usr/bin/qemu-aarch64`;
const qemuHash = '73cc2584f119f9e74c85c0a8549dbb6fd104ae3897a06a7500797f9e5b4d0943';
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const digest = file => hash(fs.readFileSync(file));
const readJSON = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const writeJSON = (file, value) => fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
const requireFact = (value, message) => { if (!value) throw new Error(message); };

requireFact(digest(identityPath) === identityHash, 'Installed identity receipt changed');
const identity = readJSON(identityPath);
const plan = readJSON(`${evidence}/plan.json`);
const receipt = readJSON(receiptPath);
const buildReceiptPath = `${path.dirname(receiptPath)}/inputs.json`;
const buildReceipt = readJSON(buildReceiptPath);
requireFact(receipt.source_commit === identity.commit, 'Retained test binary belongs to another source');
requireFact(receipt.sha256 === expectedDriverHash, 'Retained test binary receipt changed');
requireFact(receipt.driver_sha256 === identity.installed_binary_sha256['bin/go'], 'Driver receipt mismatch');
requireFact(receipt.compiler_sha256 === identity.installed_binary_sha256['pkg/tool/linux_amd64/compile'], 'Compiler receipt mismatch');
const fork = identity.worktree;
requireFact(buildReceipt.command[0] === `${fork}/bin/go` && buildReceipt.command.includes('-race')
  && buildReceipt.command.includes(`-o=${receipt.path}`) && buildReceipt.command.at(-1) === 'cmd/go'
  && !buildReceipt.command.some(argument => argument.startsWith('-overlay')), 'Unexpected retained binary build command');
const scriptCwd = `${fork}/src/cmd/go`;
const runDirectory = `${evidence}/runs/installed-001`;
const linkerBinary = `${runDirectory}/linker-capabilities.test`;
const stages = [
  { name: 'overlay-installed', cwd: scriptCwd, seconds: 660,
    argv: [receipt.path, '-test.v', '-test.count=1', '-test.short', '-test.parallel=1', '-test.timeout=10m', '-test.run=^TestScript$/^logos_eject_overlay_sources$'],
    tests: ['TestScript', 'TestScript/logos_eject_overlay_sources'] },
  { name: 'linker-build', cwd: `${fork}/src`, seconds: 600, compile: true,
    argv: [`${fork}/bin/go`, 'test', '-p=1', '-c', '-o', linkerBinary, 'cmd/link/internal/ld'] },
  { name: 'linker-components', cwd: `${fork}/src/cmd/link/internal/ld`, seconds: 240,
    argv: [linkerBinary, '-test.v', '-test.count=1', '-test.short', '-test.parallel=1', '-test.timeout=3m', '-test.run=Test(CapGraph|LogosCap)'] },
  { name: 'linker-vet', cwd: `${fork}/src`, seconds: 600, compile: true,
    argv: [`${fork}/bin/go`, 'vet', '-p=1', 'cmd/link/internal/ld'] },
  { name: 'capability-scripts', cwd: scriptCwd, seconds: 1260,
    argv: [receipt.path, '-test.v', '-test.count=1', '-test.short', '-test.parallel=1', '-test.timeout=20m', '-test.run=^TestScript$/^logos_cap_(chains|exec|policy|shortest_frames|test_scope)$'],
    tests: ['TestScript', ...['chains', 'exec', 'policy', 'shortest_frames', 'test_scope'].map(name => `TestScript/logos_cap_${name}`)] },
  { name: 'sum-crossarch', cwd: scriptCwd, seconds: 660, cross: true,
    argv: [receipt.path, '-test.v', '-test.count=1', '-test.parallel=1', '-test.timeout=10m', '-test.run=^TestScript$/^logos_sum_crossarch$'],
    tests: ['TestScript', 'TestScript/logos_sum_crossarch'] },
];

if (!process.argv.includes('--execute')) {
  console.log(JSON.stringify({ status: 'PREPARED_ONLY', identityPath, identityHash, receiptPath,
    receiptSHA256: digest(receiptPath), stages, crossStageCGOEnabled: 'UNSET', nativeCGOEnabled: '1' }, null, 2));
  process.exit(0);
}

function sourceBinding() {
  const git = args => execFileSync('git', ['--no-optional-locks', ...args], { cwd: fork, encoding: 'utf8' }).trim();
  const commit = git(['rev-parse', 'HEAD']);
  const tree = git(['rev-parse', 'HEAD^{tree}']);
  const srcTree = git(['rev-parse', 'HEAD:src']);
  const status = git(['status', '--porcelain=v1', '--untracked-files=no']);
  requireFact(commit === identity.commit && tree === identity.tree && srcTree === identity.src_tree, 'Final source identity changed');
  requireFact(status === '', 'Final tracked worktree is dirty');
  const sources = plan.sources.map(entry => {
    const actual = digest(`${fork}/${entry.path}`);
    requireFact(actual === entry.sha256, `Frozen source changed: ${entry.path}`);
    return { path: entry.path, sha256: actual };
  });
  const installedTools = Object.fromEntries(Object.entries(identity.installed_binary_sha256).map(([file, expected]) => {
    const actual = digest(`${fork}/${file}`);
    requireFact(actual === expected, `Installed tool changed: ${file}`);
    return [file, actual];
  }));
  requireFact(digest(receipt.path) === expectedDriverHash, 'Retained cmd/go race binary changed');
  requireFact(digest(qemu) === qemuHash, 'Pinned QEMU binary changed');
  const discovery = fs.readFileSync(`${fork}/src/cmd/link/internal/ld/capabilities_discovery_logos_test.go`, 'utf8');
  requireFact(discovery.includes('context.WithTimeout(t.Context(), 5*time.Second)'), 'Literal five-second child budget changed');
  return { commit, tree, src_tree: srcTree, clean: true, sources, installedTools,
    cmdgoRace: { path: receipt.path, sha256: expectedDriverHash }, qemu: { path: qemu, sha256: qemuHash } };
}

function resources() {
  const meminfo = fs.readFileSync('/proc/meminfo', 'utf8');
  const available = Number(/^MemAvailable:\s+(\d+) kB$/m.exec(meminfo)?.[1]) * 1024;
  const disk = fs.statfsSync(base);
  const home = fs.statfsSync('/home/agent');
  return { MemAvailable: available, workspaceAvailable: disk.bavail * disk.bsize,
    homeAvailable: home.bavail * home.bsize };
}

fs.mkdirSync(path.dirname(runDirectory), { recursive: true });
fs.mkdirSync(runDirectory, { recursive: false });
const before = sourceBinding();
writeJSON(`${runDirectory}/binding-before.json`, before);
const compileTemp = `${base}/final-tooling-compile-tmp`;
fs.mkdirSync(compileTemp, { recursive: true });
const sourceTemp = fs.mkdtempSync('/home/agent/logos-final-tooling-9949907ca71e-');
requireFact(fs.realpathSync(sourceTemp) === sourceTemp, 'Source temporary path is not a real home path');
for (let p = sourceTemp; ; p = path.dirname(p)) {
  for (const marker of ['.git', '.hg', '.svn']) requireFact(!fs.existsSync(path.join(p, marker)), `VCS ancestry: ${p}/${marker}`);
  if (p === path.dirname(p)) break;
}
writeJSON(`${runDirectory}/preflight.json`, { recorded_utc: new Date().toISOString(), resources: resources(),
  identity: { path: identityPath, sha256: identityHash }, retainedBinaryReceipt: { path: receiptPath, sha256: digest(receiptPath), receipt },
  retainedBinaryBuild: { path: buildReceiptPath, sha256: digest(buildReceiptPath), command: buildReceipt.command },
  originalPlanSHA256: digest(`${evidence}/plan.json`), runnerSHA256: digest(new URL(import.meta.url)),
  sourceTemp, compileTemp, crossStageCGOEnabled: 'UNSET', nativeCGOEnabled: '1' });

const inherited = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/^(GO|BOOT_GO_|CGO_|LOGOS_|CMDGO_)/.test(key)));
const selected = {
  GOROOT: fork, GOCACHE: identity.fork_cache, GOMAXPROCS: '1', GOMEMLIMIT: '512MiB', GOGC: '20',
  GOFLAGS: '-p=1', GOENV: 'off', GOTOOLCHAIN: 'local', GOWORK: 'off', GOPROXY: 'off', GOSUMDB: 'off',
  GOEXPERIMENT: '', GOOS: 'linux', GOARCH: 'amd64', GOAMD64: 'v1', CC: '/usr/bin/gcc-14',
  GOPATH: `${base}/final-acceptance-gopath`,
  PATH: `${fork}/bin:${path.dirname(qemu)}:/usr/local/bin:/usr/bin:/bin`,
};
const results = [];
let failure = null;
let active = null;
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { if (active) active.kill(signal); });

try {
  for (const stage of stages) {
    const directory = `${runDirectory}/${stage.name}`;
    fs.mkdirSync(directory);
    const temporary = stage.compile ? compileTemp : sourceTemp;
    const phaseEnvironment = { ...selected, TMPDIR: temporary, GOTMPDIR: temporary, PWD: stage.cwd };
    if (!stage.cross) phaseEnvironment.CGO_ENABLED = '1';
    const environment = { ...inherited, ...phaseEnvironment };
    requireFact(!stage.cross || !Object.hasOwn(environment, 'CGO_ENABLED'), 'Cross-target cgo override leaked');
    const command = ['/usr/bin/timeout', '--kill-after=10s', `${stage.seconds}s`, ...stage.argv];
    writeJSON(`${directory}/inputs.json`, { started_utc: new Date().toISOString(), command, cwd: stage.cwd,
      environment_overrides: phaseEnvironment, removed_environment_keys: Object.keys(process.env).filter(key => /^(GO|BOOT_GO_|CGO_|LOGOS_|CMDGO_)/.test(key)),
      CGO_ENABLED_present: Object.hasOwn(environment, 'CGO_ENABLED'), resources: resources() });
    const log = `${directory}/raw.log`;
    const fd = fs.openSync(log, 'wx');
    const started = performance.now();
    console.log(`START ${stage.name} ${new Date().toISOString()}`);
    const termination = await new Promise((resolve, reject) => {
      active = spawn(command[0], command.slice(1), { cwd: stage.cwd, env: environment, stdio: ['ignore', fd, fd] });
      active.once('error', reject);
      active.once('close', (code, signal) => resolve({ code, signal }));
    });
    active = null;
    fs.closeSync(fd);
    const text = fs.readFileSync(log, 'utf8');
    const passed = [...text.matchAll(/^\s*--- PASS: (\S+) \(/gm)].map(match => match[1]);
    const skipped = [...text.matchAll(/^\s*--- SKIP: (.+)$/gm)].map(match => match[1]);
    const failed = [...text.matchAll(/^\s*--- FAIL: (.+)$/gm)].map(match => match[1]);
    const missing = (stage.tests ?? []).filter(test => !passed.includes(test));
    const observations = {};
    if (stage.name === 'linker-components') {
      observations.topLevelTests = passed.filter(name => !name.includes('/')).length;
      observations.reportedCells = passed.length;
    }
    if (stage.cross) {
      observations.qemuExecutionObserved = /^\s*> exec qemu-aarch64 \$WORK\/sum-arm64\$GOEXE$/m.test(text);
      observations.nativeArm64ComparisonObserved = /^\s*> cmp \$WORK\/native\.txt \$WORK\/arm64\.txt$/m.test(text);
    }
    const result = { name: stage.name, finished_utc: new Date().toISOString(), exit_code: termination.code,
      signal: termination.signal, seconds: Math.round((performance.now() - started) * 1000) / 1000000,
      raw_log_sha256: digest(log), passed_tests: passed, skipped_tests: skipped, failed_tests: failed,
      missing_tests: missing, observations };
    writeJSON(`${directory}/result.json`, result);
    results.push(result);
    console.log(`DONE ${stage.name} exit=${termination.code} seconds=${result.seconds}`);
    requireFact(termination.code === 0 && termination.signal === null, `${stage.name} command failed`);
    requireFact(skipped.length === 0 && failed.length === 0 && missing.length === 0, `${stage.name} acceptance coverage failed`);
    requireFact(!/WARNING: DATA RACE|testing: warning: no tests to run/.test(text), `${stage.name} race or no-tests warning`);
    if (stage.name === 'linker-components') requireFact(observations.topLevelTests === 25 && observations.reportedCells === 75, 'Linker test scope changed');
    if (stage.cross) requireFact(observations.qemuExecutionObserved && observations.nativeArm64ComparisonObserved, 'QEMU execution or comparison not observed');
    if (stage.name === 'linker-build') writeJSON(`${directory}/retained-binary.json`, { path: linkerBinary, sha256: digest(linkerBinary), source_commit: identity.commit });
  }
} catch (error) {
  failure = error instanceof Error ? error.message : String(error);
} finally {
  let after = null;
  try { after = sourceBinding(); writeJSON(`${runDirectory}/binding-after.json`, after); }
  catch (error) { failure = `${failure ?? ''}${failure ? '; ' : ''}restoration check: ${error.message}`; }
  writeJSON(`${runDirectory}/summary.json`, { recorded_utc: new Date().toISOString(), commit: identity.commit,
    results, source_tools_unchanged: after !== null, failure, acceptance_passed: failure === null,
    caveat: 'Native capability cgo cells require transcript review; no full regression or release-parity claim.' });
}
if (failure !== null) { console.error(failure); process.exitCode = 1; }
