// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import test from "node:test";
import assert from "node:assert/strict";
import { timing, benchmarkRows, assessRow, goResults, expectedRed, replaceOnce, verdict, manifest, goFailureSummary } from "./check.mjs";

const name = "BenchmarkCgoOptNocallback";
const iterations = 40000000;
const hyperfine = (seconds) => ({ schema_version: 2, primary_metric: "time_wall_clock",
  results: [{ measurements: Array.from({ length: 10 }, () => ({ exit_code: 0,
    time_wall_clock: { value: seconds, unit: "second" } })) }] });
const benchmark = (n, ns = 50, count = 10) =>
  Array(count).fill(name + "-1 " + n + " " + ns + " ns/op 0 B/op 0 allocs/op").join("\n");
const phase = (seconds) => ({ hyperfine: hyperfine(seconds), corroboration: benchmark(iterations) });
const phases = () => ({ baseline: phase(2.2), candidate: phase(2), integration: phase(2) });
const events = (...rows) => rows.map((row) => JSON.stringify(row)).join("\n");

test("requires every integration row to reach the unchanged 1.05 gate", () => {
  const good = assessRow(name, iterations, phases());
  assert.equal(good.status, "latency-pass");
  const bad = phases();
  bad.integration = phase(2.15);
  const rejected = assessRow(name, iterations, bad);
  assert.equal(rejected.status, "latency-rejected");
  assert.equal(verdict([good, rejected]), "not-accepted");
});

test("blocks noise instead of selecting favorable samples", () => {
  const input = phases();
  input.candidate.hyperfine.results[0].measurements[9].time_wall_clock.value = 8;
  assert.equal(assessRow(name, iterations, input).status, "measurement-blocked");
});

test("does not accept short work hidden by process overhead", () => {
  const input = phases();
  input.candidate.corroboration = benchmark(iterations, 10);
  assert.equal(assessRow(name, iterations, input).status, "measurement-blocked");
});

test("rejects different work counts, omitted rows, and failed timed commands", () => {
  assert.throws(() => benchmarkRows(benchmark(iterations + 1), name, iterations, 10), /work count/);
  assert.throws(() => benchmarkRows(benchmark(iterations, 50, 9), name, iterations, 10), /exact benchmark/);
  const failed = hyperfine(2);
  failed.results[0].measurements[4].exit_code = 1;
  assert.throws(() => timing(failed), /timed command failed/);
  assert.throws(() => timing({ schema_version: 2, primary_metric: "time_wall_clock",
    results: [] }), /exactly one command/);
});

test("rejects invalid samples and missing allocation metrics", () => {
  for (const value of [0, -1, NaN, Infinity]) {
    const input = hyperfine(2);
    input.results[0].measurements[0].time_wall_clock.value = value;
    assert.throws(() => timing(input), /finite and positive/);
  }
  assert.throws(() => benchmarkRows(name + " 4 2 ns/op", name, 4, 1), /metric/);
});

test("recomputes statistics from the raw samples", () => {
  const input = hyperfine(2);
  input.results[0].summary = { time_wall_clock: { median: 1000 } };
  assert.equal(timing(input).median, 2);
});

test("a passing package with no required tests is not coverage", () => {
  assert.throws(() => goResults(events({ Action: "pass", Package: "runtime" }),
    { prefix: "TestLogosCgo" }), /did not execute/);
});

test("enabled runtime tests cannot silently skip", () => {
  const output = events({ Action: "run", Test: "TestLogosCgoShape" },
    { Action: "skip", Test: "TestLogosCgoShape" });
  assert.throws(() => goResults(output, { prefix: "TestLogosCgo" }), /ended as skip/);
  assert.equal(goResults(output, { prefix: "TestLogosCgo", allowSkipped: true }).TestLogosCgoShape, "skip");
});

test("RED requires the named running attack and its predicted diagnostic", () => {
  const output = events({ Action: "run", Test: "TestCgoOptEmission" },
    { Action: "output", Test: "TestCgoOptEmission", Output: "stack-top lookups = 2, want 0\n" },
    { Action: "fail", Test: "TestCgoOptEmission" });
  expectedRed(output, "TestCgoOptEmission", "stack-top lookups = 2, want 0");
  assert.throws(() => expectedRed(output, "TestOther", "stack-top"), /did not execute/);
  assert.throws(() => expectedRed(output, "TestCgoOptEmission", "wrong"), /wrong reason/);
  assert.throws(() => expectedRed(output.replace('"Test":"TestCgoOptEmission","Output"',
    '"Test":"TestUnrelated","Output"'), "TestCgoOptEmission", "stack-top"), /wrong reason/);
  assert.throws(() => expectedRed(events({ Action: "build-fail" }),
    "TestCgoOptEmission", "stack-top"), /build failure/);
});

test("probe replacements fail on absent or duplicated targets", () => {
  assert.equal(replaceOnce("before target after", "target", "new"), "before new after");
  assert.throws(() => replaceOnce("none", "target", "new"), /exactly once/);
  assert.throws(() => replaceOnce("target target", "target", "new"), /exactly once/);
});


test("manifest requirements are explicit and cannot be silently duplicated", () => {
  const base = { version: 1, runtimeFamilies: [], benchmarks: [{ name, role: "target" }],
    fixtures: [{ name: "ordinary", path: "src/cmd/cgo/internal/testlogos/testdata/cgoopt",
      args: [], stdout: "ok" }] };
  assert.equal(manifest(base), base);
  assert.throws(() => manifest({ ...base, benchmarks: [] }), /no target/);
  assert.throws(() => manifest({ ...base, fixtures: [] }), /no direct fixture/);
  assert.throws(() => manifest({ ...base, benchmarks: [...base.benchmarks, ...base.benchmarks] }),
    /duplicate benchmark/);
  const family = { package: "runtime", prefix: "TestLogosCgoShape",
    tests: ["TestLogosCgoShape"], allowSkippedOff: false };
  assert.equal(manifest({ ...base, runtimeFamilies: [family] }).runtimeFamilies.length, 1);
  assert.throws(() => manifest({ ...base, runtimeFamilies: [{ ...family, tests: ["TestUnrelated"] }] }),
    /runtime test requirement/);
});

test("a terminal pass event alone does not prove the required test ran", () => {
  assert.throws(() => goResults(events({ Action: "pass", Test: "TestLogosCgoShape" }),
    { prefix: "TestLogosCgo" }), /did not execute/);
});


test("an early failed diagnostic survives a long passing suffix", () => {
  const early = events(
    { Action: "run", Package: "runtime", Test: "TestLogosCgoAddress/eligible" },
    { Action: "output", Package: "runtime", Test: "TestLogosCgoAddress/eligible",
      Output: "    address_test.go:42: pointer-free address was rejected\n" },
    { Action: "fail", Package: "runtime", Test: "TestLogosCgoAddress/eligible" });
  const suffix = Array(2000).fill(events(
    { Action: "output", Package: "runtime", Test: "TestOther", Output: "passing output\n" },
    { Action: "pass", Package: "runtime", Test: "TestOther" })).join("\n");
  const summary = goFailureSummary(early + "\n" + suffix);
  assert.equal(summary.length, 1);
  assert.equal(summary[0].test, "TestLogosCgoAddress/eligible");
  assert.equal(summary[0].diagnostics[0], "address_test.go:42: pointer-free address was rejected");
});

test("failure summaries retain each failed test and bound assembly diagnostics", () => {
  const summary = goFailureSummary("not JSON\n" + events(
    { Action: "output", Package: "p", Test: "TestOne/sub", Output: "=== RUN   TestOne/sub\n" },
    { Action: "output", Package: "p", Test: "TestOne/sub",
      Output: "file.go:4: wrong lookup count\n" + "x".repeat(1000) + "\nassembly 2\nassembly 3\n" },
    { Action: "fail", Package: "p", Test: "TestOne/sub" },
    { Action: "fail", Package: "p", Test: "TestOne" },
    { Action: "output", Package: "p", Test: "TestTwo", Output: "other.go:9: wrong eligibility\n" },
    { Action: "fail", Package: "p", Test: "TestTwo" }));
  assert.equal(summary.length, 3);
  assert.equal(summary[0].diagnostics.length, 3);
  assert.equal(summary[0].diagnostics[1].length, 200);
  assert.equal(summary[1].test, "TestOne");
  assert.equal(summary[1].diagnostics.length, 0);
  assert.equal(summary[2].diagnostics[0], "other.go:9: wrong eligibility");
});
