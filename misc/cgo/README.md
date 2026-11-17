# Ordinary cgo validation

The cgo workflow supplements the existing LoGOs workflow. Its focused job runs
first. Full dist and benchmark jobs start only after focused validation passes.
A test-only commit can therefore record an actual expected failure before a
follow-up implementation commit, without starting performance measurements.

## Requirements and coverage

Before mutation probes, CI requires tracked cgo/runtime sources to pass gofmt
without rewriting them, then runs full go vet checks with the experiment off/on.

CI uses the existing pinned Go 1.27.4 bootstrap, pinned Node 26.13.1, and verified
Go 1.27.2/1.27.3 oracles for dist tests. GCC, G++, gccgo, and Clang must exist;
their versions are saved. Missing gccgo is a failure, even though the upstream
test helper would otherwise skip its generator regression.

Both explicit settings disable fast-cgo:

- off: GOEXPERIMENT=nologoscgoopt,nologosfastcgo
- on: GOEXPERIMENT=logoscgoopt,nologosfastcgo

The focused job runs the TestCgoOpt family, TestPointerChecks, and every declared
runtime family under normal, race, and cgocheck2 builds for both settings.
A passing package with no selected test does not establish coverage. Enabled
tests must execute and pass. Declared off-mode skips require an explicit
allowSkippedOff entry.

Each declared plain-Go fixture runs directly with -asan and -msan under both
settings, and with stock Go 1.27.4 as an output oracle. Instrumenting a parent
test does not prove that its subprocesses are instrumented; these commands
instrument the fixture itself.

The full job runs every dist group with ./run.bash --no-rebuild -k. It explicitly
uses the command's default short mode (GO_TEST_SHORT=1), two Go workers,
-p=2, and GO_TEST_TIMEOUT_SCALE=6. Its compact environment avoids overflowing
the WebAssembly argument/environment buffer seen with the inherited CI
environment. There are no test-name exclusions or ignored failures.
This is the full default-short dist suite, not the separate goparity release
tier or a non-short release run.

## Declared tests, fixtures, and measurements

manifest.json starts with an empty runtimeFamilies list because the return-frame
slice adds no runtime test. A later test-only commit adds its requirements:

    {
      "package": "runtime",
      "prefix": "TestLogosCgoSpan",
      "tests": ["TestLogosCgoSpanObject", "TestLogosCgoSpanGlobals"],
      "allowSkippedOff": false
    }

The prefix selects the family and tests lists the required exact names.
Every selected test must satisfy the result check. A declared family or exact
test that does not execute fails validation.

Add each direct fixture as a fixtures entry with name, repository-relative path,
args, and exact trimmed stdout. Fixtures must be ordinary Go programs under
src/cmd/cgo/internal/testlogos/testdata so the stock oracle can run them.

Every BenchmarkCgoOpt function must have an exact benchmarks entry. role is
target or reference. A reference row is measured and labeled separately; it
does not establish optimization acceptance or a same-contract floor.

## Checkout-only probes

probes.json describes two controls for the return-frame proof: disabling the
new condition and inverting p.noCallbacks. probe.mjs changes only the local
checkout, refreshes installed cmd/cgo, requires TestCgoOptEmission to execute
and fail with the predicted stack-top diagnostic, restores the source and tool,
then requires the unmodified test to pass. A build error, timeout, unrelated
test failure, or surviving mutation is a failed probe.

A checkout-only control is not evidence that CI ran before implementation was
written. Real test-before-implementation evidence must come from the test-only
commit. Later runtime probes use setting=on to enable their runtime condition
and may use rebuild=toolchain to refresh the whole toolchain. The default probe
setting is off; return-frame tests set their own generator experiments.
Source replacement must match exactly once; restoration runs even
when the probe fails.

## Fixed-work benchmark protocol

benchmark.mjs builds baseline, candidate, and integration binaries sequentially
from the same checked-out source. Baseline disables logoscgoopt. Candidate enables
it. Integration independently rebuilds the enabled configuration with -a;
it is a fresh build, not a claim that another merge commit was tested.
All measurements use GOMAXPROCS=1 and one Go build worker.

For each exact benchmark, adaptive two-second pilots select a common iteration
count. Fixed pilots must show at least one second of useful work in every phase
before that count is frozen. Calibration logs are retained. Scored measurements
then run once per phase with hyperfine --warmup 3 --min-runs 10 --export-json.
There is no automatic retry of a noisy or unfavorable scored campaign.

Raw samples determine the median and sample standard deviation. A phase with
standard deviation above 20% of its median is blocked. A timed sample shorter
than one second, or a corroborating Go benchmark row with less than one second
of useful work, also blocks the result. Startup overhead cannot establish the
useful-work requirement.

Each target requires both baseline/candidate and baseline/integration median
ratios to reach 1.05. Every target must pass independently. A successful row does
not accept another row. Each phase also runs the exact Go benchmark with the
same fixed work, -count=10, and -benchmem. Allocation bytes, allocation counts,
binary bytes, binaries, and a separate CPU profile are preserved.

This job checks measurement admission and the latency threshold. It does not
manufacture a floor or grant G6 promotion without the remaining evidence
required by logos-doc/IMPLEMENTATION.md.

## Reproduction

After building the fork, set HOME, PATH, RUNNER_TEMP, and UPROOT1274 to the stock
Go 1.27.4 root. Set HYPERFINE to the verified hyperfine binary for measurements.
Use an evidence directory outside the checkout:

    node --test misc/cgo/*.test.mjs
    node misc/cgo/probe.mjs "$RUNNER_TEMP/cgo-evidence/probes"
    node misc/cgo/verify.mjs "$RUNNER_TEMP/cgo-evidence/focused"
    node misc/cgo/benchmark.mjs "$RUNNER_TEMP/cgo-evidence/benchmark"

Every executed command records its arguments, compact environment, status,
stdout, and stderr. CI uploads evidence even when validation fails. Full dist
records its complete combined output and the source/compiler fingerprint.

The command runner executes a program with an argument array and explicitly
disables shell execution. Configured executable paths are literal paths; spaces,
quotes, substitutions, and other shell metacharacters in paths or arguments do
not become shell code. The boundary test exercises those values and fails if
shell execution is enabled. Hyperfine receives individually quoted benchmark
arguments for its own command parser.
