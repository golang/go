// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

function insist(condition, message) {
  if (!condition) throw new Error(message);
}

function positive(value, label) {
  insist(typeof value === "number" && Number.isFinite(value) && value > 0,
    label + " must be finite and positive");
  return value;
}

export function timing(document) {
  insist(document && document.schema_version === 2 &&
    document.primary_metric === "time_wall_clock",
    "hyperfine must export schema 2 wall-clock timings");
  insist(Array.isArray(document.results) && document.results.length === 1,
    "hyperfine must contain exactly one command");
  const result = document.results[0];
  insist(Array.isArray(result.measurements) && result.measurements.length >= 10,
    "hyperfine must contain at least ten measured samples");
  const times = result.measurements.map((sample) => {
    insist(sample && sample.exit_code === 0,
      "a timed command failed or lacks an exit code");
    insist(sample.time_wall_clock && sample.time_wall_clock.unit === "second",
      "a timed sample is not a wall-clock second");
    return positive(sample.time_wall_clock.value, "sample");
  });
  const sorted = [...times].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
  const mean = times.reduce((sum, value) => sum + value, 0) / times.length;
  const variance = times.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (times.length - 1);
  const standardDeviation = Math.sqrt(variance);
  return { samples: times.length, median, standardDeviation, minimum: sorted[0],
    maximum: sorted.at(-1), noise: standardDeviation / median };
}

export function benchmarkRows(output, name, iterations, count) {
  insist(/^BenchmarkCgoOpt[A-Za-z0-9_]*$/.test(name), "invalid exact benchmark name");
  const rows = [];
  for (const line of output.split("\n")) {
    const fields = line.trim().split(/\s+/);
    if (fields[0].replace(/-\d+$/, "") !== name) continue;
    const n = Number(fields[1]);
    insist(Number.isSafeInteger(n) && n > 0, "invalid benchmark iteration count");
    insist(iterations === null || n === iterations, "benchmark work count changed");
    const metric = (unit) => {
      const at = fields.indexOf(unit);
      insist(at >= 3 && at % 2 === 1, "missing benchmark metric " + unit);
      const value = Number(fields[at - 1]);
      insist(Number.isFinite(value) && value >= 0, "invalid benchmark metric " + unit);
      return value;
    };
    const nanoseconds = positive(metric("ns/op"), "ns/op");
    rows.push({ iterations: n, nanoseconds, bytes: metric("B/op"),
      allocations: metric("allocs/op"), usefulSeconds: n * nanoseconds / 1e9 });
  }
  insist(rows.length === count, "expected " + count + " exact benchmark rows, got " + rows.length);
  return rows;
}

export function assessRow(name, iterations, phases) {
  insist(Number.isSafeInteger(iterations) && iterations > 0, "invalid frozen work count");
  const blocked = [];
  const statistics = {};
  for (const phase of ["baseline", "candidate", "integration"]) {
    const data = phases[phase];
    insist(data, "missing phase " + phase);
    statistics[phase] = timing(data.hyperfine);
    const rows = benchmarkRows(data.corroboration, name, iterations, 10);
    if (statistics[phase].noise > 0.20) blocked.push(phase + ": noise exceeds 20% of median");
    if (statistics[phase].minimum < 1 || rows.some((row) => row.usefulSeconds < 1)) {
      blocked.push(phase + ": less than one second of work; a new frozen campaign is required");
    }
    statistics[phase].goBenchmarks = rows;
  }
  const candidateRatio = statistics.baseline.median / statistics.candidate.median;
  const integrationRatio = statistics.baseline.median / statistics.integration.median;
  const status = blocked.length > 0 ? "measurement-blocked"
    : candidateRatio >= 1.05 && integrationRatio >= 1.05 ? "latency-pass" : "latency-rejected";
  return { name, iterations, status, blocked, candidateRatio, integrationRatio, statistics,
    promotion: "not established: latency evidence does not replace parity, admission, or a justified floor" };
}

export function goResults(output, { names = [], prefix = "", allowSkipped = false }) {
  const events = output.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  const selected = new Map();
  const ran = new Set();
  for (const event of events) {
    if (typeof event.Test !== "string" || event.Test.includes("/")) continue;
    if (!names.includes(event.Test) && !(prefix && event.Test.startsWith(prefix))) continue;
    if (event.Action === "run") {
      selected.set(event.Test, "running");
      ran.add(event.Test);
    }
    if (["pass", "fail", "skip"].includes(event.Action)) selected.set(event.Test, event.Action);
  }
  insist(selected.size > 0, "required test family did not execute");
  for (const name of names) insist(selected.has(name), "required test did not execute: " + name);
  for (const [name, action] of selected) {
    insist(ran.has(name), "required test did not execute: " + name);
    insist(action === "pass" || (allowSkipped && action === "skip"),
      "required test " + name + " ended as " + action);
  }
  return Object.fromEntries(selected);
}

export function expectedRed(output, test, diagnostic) {
  const events = output.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  insist(!events.some((event) => event.Action === "build-fail"), "build failure is not an attack failure");
  insist(events.some((event) => event.Action === "run" && event.Test === test),
    "RED test did not execute");
  insist(events.some((event) => event.Action === "fail" && event.Test === test),
    "RED test did not fail");
  const text = events.filter((event) => typeof event.Output === "string" &&
    (event.Test === test || event.Test?.startsWith(test + "/")))
    .map((event) => event.Output).join("");
  insist(text.includes(diagnostic), "RED failed for the wrong reason");
}

export function replaceOnce(source, before, after) {
  insist(typeof before === "string" && before.length > 0 && typeof after === "string",
    "invalid probe replacement");
  const index = source.indexOf(before);
  insist(index >= 0 && source.indexOf(before, index + before.length) < 0,
    "probe target must occur exactly once");
  return source.slice(0, index) + after + source.slice(index + before.length);
}

export function verdict(rows) {
  insist(rows.length > 0, "no benchmark rows were measured");
  return rows.every((row) => row.status === "latency-pass") ? "latency-pass" : "not-accepted";
}

export function manifest(value) {
  insist(value && value.version === 1 && Array.isArray(value.runtimeFamilies) &&
    Array.isArray(value.benchmarks) && Array.isArray(value.fixtures),
    "invalid cgo manifest version or lists");
  const tests = new Set();
  for (const family of value.runtimeFamilies) {
    insist(family.package === "runtime" && /^TestLogosCgo[A-Za-z0-9_]*$/.test(family.prefix) &&
      typeof family.allowSkippedOff === "boolean" && Array.isArray(family.tests) &&
      family.tests.length > 0, "invalid runtime test family");
    for (const name of family.tests) {
      insist(/^TestLogosCgo[A-Za-z0-9_]*$/.test(name) && name.startsWith(family.prefix) &&
        !tests.has(name), "invalid or duplicate runtime test requirement");
      tests.add(name);
    }
  }
  const names = new Set();
  for (const row of value.benchmarks) {
    insist(/^BenchmarkCgoOpt[A-Za-z0-9_]*$/.test(row.name) &&
      ["target", "reference"].includes(row.role) && !names.has(row.name),
      "invalid or duplicate benchmark requirement");
    names.add(row.name);
  }
  insist(value.benchmarks.some((row) => row.role === "target"), "no target benchmark configured");
  const fixtures = new Set();
  for (const fixture of value.fixtures) {
    insist(/^[a-z0-9-]+$/.test(fixture.name) && !fixtures.has(fixture.name) &&
      /^src\/cmd\/cgo\/internal\/testlogos\/testdata\/[A-Za-z0-9_/-]+$/.test(fixture.path) &&
      Array.isArray(fixture.args) && fixture.args.every((arg) => typeof arg === "string") &&
      typeof fixture.stdout === "string" && fixture.stdout.length > 0,
      "invalid or duplicate direct fixture");
    fixtures.add(fixture.name);
  }
  insist(fixtures.size > 0, "no direct fixture configured");
  return value;
}


// Keep diagnostics independently of the output tail: later passing cases may
// otherwise hide the reason an early test or subtest failed.
export function goFailureSummary(output) {
  const records = new Map();
  for (const line of output.split("\n")) {
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      continue; // Non-JSON command output still has its complete raw log.
    }
    if (!event || typeof event !== "object" || typeof event.Action !== "string") continue;
    const pkg = typeof event.Package === "string" ? event.Package : "";
    const test = typeof event.Test === "string" ? event.Test : "";
    const key = JSON.stringify([pkg, test]);
    if (!records.has(key)) records.set(key, { package: pkg, test, diagnostics: [], failed: false });
    const record = records.get(key);
    if (typeof event.Output === "string" && record.diagnostics.length < 3) {
      for (const text of event.Output.split("\n")) {
        const diagnostic = text.trim();
        if (!diagnostic || /^(=== |--- (PASS|FAIL|SKIP):|PASS$|FAIL(?:\s|$)|ok\s)/.test(diagnostic)) continue;
        record.diagnostics.push(diagnostic.slice(0, 200));
        if (record.diagnostics.length === 3) break;
      }
    }
    if (event.Action === "fail" || event.Action === "build-fail") record.failed = true;
  }
  return [...records.values()].filter((record) => record.failed)
    .map(({ package: pkg, test, diagnostics }) => ({ package: pkg, test, diagnostics }));
}
