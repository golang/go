// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { goResults, manifest } from "./check.mjs";
import { root, go, experiments, goEnv, run, saveJSON, snapshot } from "./common.mjs";

const directory = resolve(process.argv[2] || "");
if (process.argv.length !== 3) throw new Error("usage: node verify.mjs EVIDENCE_DIRECTORY");
snapshot(join(directory, "environment"));
const observed = [];
const config = manifest(JSON.parse(readFileSync(join(root, "misc/cgo/manifest.json"), "utf8")));
try {
  const suites = [
    { pkg: "cmd/cgo/internal/testlogos", pattern: "^TestCgoOpt",
      requirement: { prefix: "TestCgoOpt", names: ["TestCgoOptEmission", "TestCgoOptLateDirective",
        "TestCgoOptGccgoUnchanged", "TestCgoOptRuntime"] } },
    ...config.runtimeFamilies.map((family) => ({ pkg: family.package,
      pattern: "^" + family.prefix, allowSkippedOff: family.allowSkippedOff,
      requirement: { prefix: family.prefix, names: family.tests } })),
    { pkg: "cmd/cgo/internal/testerrors", pattern: "^TestPointerChecks$",
      requirement: { names: ["TestPointerChecks"] } },
  ];
  // Run the enabled normal cases first so a test-first RED stops before expensive modes.
  for (const setting of ["on", "off"]) {
    const experiment = experiments[setting];
    for (const mode of ["normal", "race", "cgocheck2"]) {
      const env = goEnv(experiment + (mode === "cgocheck2" ? ",cgocheck2" : ""));
      for (const suite of suites) {
        const label = setting + "-" + mode + "-" + suite.pkg.replaceAll("/", "_") + "-" +
          suite.pattern.replaceAll(/[^A-Za-z0-9_]/g, "");
        const args = ["test", "-json", "-p=2", "-count=1", "-timeout=30m",
          "-run=" + suite.pattern];
        if (mode === "race") args.push("-race");
        args.push(suite.pkg);
        const result = run(go, args, { env, cwd: join(root, "src"), log: join(directory, label) });
        const tests = goResults(result.stdout, { ...suite.requirement,
          allowSkipped: setting === "off" && suite.allowSkippedOff === true });
        observed.push({ label, tests });
        saveJSON(join(directory, "test-results.json"), observed);
      }
    }
  }

  if (!process.env.UPROOT1274) throw new Error("UPROOT1274 is required for the stock oracle");
  const oracle = process.env.UPROOT1274;
  for (const fixture of config.fixtures) {
    const cwd = join(root, fixture.path);
    for (const [setting, experiment] of Object.entries(experiments)) {
      for (const sanitizer of ["asan", "msan"]) {
        const env = goEnv(experiment, sanitizer === "msan" ? { CC: "clang", CXX: "clang++" } : {});
        const label = fixture.name + "-" + setting + "-" + sanitizer;
        const result = run(go, ["run", "-" + sanitizer, ".", ...fixture.args],
          { env, cwd, log: join(directory, label) });
        if (result.stdout.trim() !== fixture.stdout) throw new Error(label + ": unexpected fixture output");
        observed.push({ label, output: result.stdout.trim() });
        saveJSON(join(directory, "test-results.json"), observed);
      }
    }
    const label = fixture.name + "-stock-oracle";
    const result = run(join(oracle, "bin/go"), ["run", ".", ...fixture.args], {
      env: goEnv("", { GOROOT: oracle }), cwd, log: join(directory, label) });
    if (result.stdout.trim() !== fixture.stdout) throw new Error(label + ": stock oracle output differs");
    observed.push({ label, output: result.stdout.trim() });
    saveJSON(join(directory, "test-results.json"), observed);
  }
} catch (error) {
  saveJSON(join(directory, "failure.json"), { message: error.message });
  throw error;
}
