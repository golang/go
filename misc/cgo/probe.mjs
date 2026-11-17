// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { expectedRed, goResults, replaceOnce } from "./check.mjs";
import { root, go, experiments, goEnv, run, saveJSON } from "./common.mjs";

if (process.argv.length !== 3) throw new Error("usage: node probe.mjs EVIDENCE_DIRECTORY");
const directory = resolve(process.argv[2]);
const probes = JSON.parse(readFileSync(join(root, "misc/cgo/probes.json"), "utf8"));
if (!Array.isArray(probes) || probes.length === 0) throw new Error("no probes configured");
const env = goEnv(experiments.off);
const initial = run("git", ["status", "--porcelain"], { env, log: join(directory, "initial-status") });
if (initial.stdout.trim()) throw new Error("probes require an unchanged checkout");
for (const probe of probes) {
  if (!/^[a-z0-9-]+$/.test(probe.id) ||
      !/^src\/(?:cmd\/cgo|runtime)\/[a-zA-Z0-9_]+\.go$/.test(probe.path) ||
      !/^Test[A-Za-z0-9_]+$/.test(probe.test) ||
      !["cgo", "toolchain"].includes(probe.rebuild) ||
      !["counterfactual-red", "mutation"].includes(probe.kind) ||
      !["off", "on"].includes(probe.setting ?? "off")) {
    throw new Error("invalid probe: " + JSON.stringify(probe));
  }
  const env = goEnv(experiments[probe.setting ?? "off"]);
  const path = join(root, probe.path);
  const original = readFileSync(path, "utf8");
  const replacement = replaceOnce(original, probe.before, probe.after);
  const out = join(directory, probe.id);
  const rebuild = (phase) => {
    if (probe.rebuild === "cgo") {
      run(go, ["install", "cmd/cgo"], { env, cwd: join(root, "src"),
        log: join(out, phase + "-install") });
    } else {
      run("bash", ["./make.bash"], { env: { ...env, GOROOT_BOOTSTRAP: process.env.UPROOT1274 },
        cwd: join(root, "src"), log: join(out, phase + "-build") });
    }
  };
  const args = ["test", "-json", "-p=2", "-count=1", "-timeout=10m",
    "-run=^" + probe.test + "$", probe.package];
  let failure;
  try {
    writeFileSync(path, replacement);
    rebuild("probe");
    const result = run(go, args, { env, cwd: join(root, "src"),
      log: join(out, "red"), allowFailure: true });
    if (result.status === 0) throw new Error(probe.id + ": probe survived");
    expectedRed(result.stdout, probe.test, probe.diagnostic);
    saveJSON(join(out, "red-proof.json"), { id: probe.id, kind: probe.kind,
      diagnostic: probe.diagnostic, status: "caught",
      scope: "checkout-only control; not a claim that CI ran before the implementation was written" });
  } catch (error) {
    failure = error;
  } finally {
    writeFileSync(path, original);
    rebuild("restored");
  }
  if (failure) throw failure;
  const green = run(go, args, { env, cwd: join(root, "src"), log: join(out, "green") });
  goResults(green.stdout, { names: [probe.test] });
}
const final = run("git", ["status", "--porcelain"], { env, log: join(directory, "final-status") });
if (final.stdout.trim()) throw new Error("probe restoration left checkout changes");
