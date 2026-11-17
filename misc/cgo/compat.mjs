// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import { join, resolve } from "node:path";
import { goResults } from "./check.mjs";
import { root, go, goEnv, run, saveJSON } from "./common.mjs";

if (process.argv.length !== 3) throw new Error("usage: node compat.mjs EVIDENCE_DIRECTORY");
const directory = resolve(process.argv[2]);
const observed = [];
const configurations = [
  { name: "both-on", experiment: "logoscgoopt,logosfastcgo" },
  { name: "opt-only", experiment: "logoscgoopt,nologosfastcgo" },
  { name: "fast-only", experiment: "nologoscgoopt,logosfastcgo" },
  { name: "both-off", experiment: "nologoscgoopt,nologosfastcgo" },
];
try {
  for (const configuration of configurations) {
    const modes = configuration.name === "both-on"
      ? ["normal", "race", "cgocheck2", "asan", "msan"] : ["normal"];
    for (const mode of modes) {
      const names = ["TestSelfBuildConfiguration", "TestCgoOptPointerFreeAddressInline",
        "TestCgoOptSpanCodegen", "TestCgoOptSpanPointerRules"];
      if (mode === "normal" || mode === "race") {
        names.push("TestFastcallOneTrampoline", "TestFastcallStructure", "TestFastcallNoClockInFastBody");
      }
      const experiment = configuration.experiment + (mode === "cgocheck2" ? ",cgocheck2" : "");
      const env = goEnv(experiment, mode === "msan" ? { CC: "clang", CXX: "clang++" } : {});
      const label = configuration.name + "-" + mode;
      const args = ["test", "-json", "-p=2", "-count=1", "-timeout=30m",
        "-run=^(" + names.join("|") + ")$"];
      if (["race", "asan", "msan"].includes(mode)) args.push("-" + mode);
      args.push("cmd/cgo/internal/testlogos");
      const result = run(go, args, { env, cwd: join(root, "src"), log: join(directory, label) });
      observed.push({ label, experiment, tests: goResults(result.stdout, { names }) });
      saveJSON(join(directory, "test-results.json"), observed);
    }
  }
} catch (error) {
  saveJSON(join(directory, "failure.json"), { message: error.message });
  throw error;
}
