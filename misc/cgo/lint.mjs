// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import { join, resolve } from "node:path";
import { root, go, experiments, goEnv, run } from "./common.mjs";

if (process.argv.length !== 3) throw new Error("usage: node lint.mjs EVIDENCE_DIRECTORY");
const directory = resolve(process.argv[2]);
const env = goEnv(experiments.off);
// Restrict runtime and generator checks to real package sources. Their other
// testdata directories contain deliberately invalid Go programs.
const tracked = run("git", ["ls-files", "-z", "--",
  ":(glob)src/runtime/*.go", ":(glob)src/cmd/cgo/*.go",
  ":(glob)src/internal/buildcfg/*.go", ":(glob)src/internal/goexperiment/*.go",
  "src/cmd/cgo/internal/testlogos"], { env, log: join(directory, "tracked-files") });
const files = tracked.stdout.split("\0").filter((path) => path.endsWith(".go"));
if (files.length === 0) throw new Error("no tracked cgo/runtime Go sources selected");
const formatted = run(join(root, "bin/gofmt"), ["-l", ...files],
  { env, log: join(directory, "gofmt") });
if (formatted.stdout.trim()) {
  console.error(formatted.stdout);
  throw new Error("gofmt reported unformatted tracked source; CI does not rewrite files");
}
for (const [setting, experiment] of Object.entries(experiments)) {
  run(go, ["vet", "-p=2", "runtime", "cmd/cgo", "cmd/cgo/internal/testlogos",
    "internal/buildcfg", "internal/goexperiment"], {
    env: goEnv(experiment), cwd: join(root, "src"), log: join(directory, setting + "-vet") });
}
