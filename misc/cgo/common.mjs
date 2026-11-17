// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { goFailureSummary } from "./check.mjs";

export const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
export const go = join(root, "bin/go");
export const experiments = {
  off: "nologoscgoopt,nologosfastcgo",
  on: "logoscgoopt,nologosfastcgo",
};

export function save(path, data) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, data);
}

export function saveJSON(path, data) {
  save(path, JSON.stringify(data, null, 2) + "\n");
}

export function goEnv(experiment, extra = {}) {
  for (const name of ["HOME", "PATH", "RUNNER_TEMP"]) {
    if (!process.env[name]) throw new Error("required environment variable is missing: " + name);
  }
  const env = {
    HOME: process.env.HOME,
    PATH: join(root, "bin") + ":" + join(root, "lib/wasm") + ":" + process.env.PATH,
    TMPDIR: process.env.RUNNER_TEMP,
    LANG: "C.UTF-8",
    TZ: "UTC",
    GOROOT: root,
    GOENV: "off",
    GOTOOLCHAIN: "local",
    CGO_ENABLED: "1",
    GOEXPERIMENT: experiment,
    GOMAXPROCS: "2",
    GOFLAGS: "-p=2",
    CC: "gcc",
    CXX: "g++",
    GCCGO: process.env.GCCGO || "gccgo",
  };
  for (const name of ["UPROOT1272", "UPROOT1273", "UPROOT1274"]) {
    if (process.env[name]) env[name] = process.env[name];
  }
  return { ...env, ...extra };
}

export function toolPath(name) {
  const value = process.env[name];
  if (!value) throw new Error("required environment variable is missing: " + name);
  const resolved = resolve(value);
  const base = resolve(process.env.RUNNER_TEMP || "");
  if (resolved !== value || !resolved.startsWith(base + "/")) {
    throw new Error(name + " must be an absolute path inside RUNNER_TEMP");
  }
  try {
    if (!statSync(resolved).isFile()) throw new Error("not a file");
  } catch (e) {
    throw new Error(name + " does not name an installed tool: " + resolved);
  }
  return resolved;
}

export function run(command, args, { env, cwd = root, log, allowFailure = false, timeout = 1800000 } = {}) {
  if (!env || !log) throw new Error("commands require an explicit environment and log path");
  console.log("Running " + command + " " + args.join(" "));
  const start = new Date().toISOString();
  const result = spawnSync(command, args, { cwd, env, shell: false, encoding: "utf8", timeout,
    maxBuffer: 128 * 1024 * 1024 });
  save(log + ".stdout", result.stdout || "");
  save(log + ".stderr", result.stderr || "");
  saveJSON(log + ".json", { command, args, cwd, env, start, end: new Date().toISOString(),
    status: result.status, signal: result.signal, error: result.error?.message || null });
  if (result.error) throw result.error;
  if (result.status !== 0 && !allowFailure) {
    const failures = goFailureSummary(result.stdout || "");
    if (failures.length > 0) {
      console.error("Go test failures:\n" + JSON.stringify(failures, null, 2));
      if (result.stderr) console.error(result.stderr.slice(-6000));
    } else {
      console.error((result.stdout || "").slice(-12000) + (result.stderr || "").slice(-12000));
    }
    throw new Error(command + " exited " + result.status + "; see " + log);
  }
  return { status: result.status, stdout: result.stdout || "", stderr: result.stderr || "" };
}

export function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

export function snapshot(directory) {
  const env = goEnv(experiments.off);
  const git = (args, label) => run("git", args, { env, log: join(directory, label) }).stdout;
  const status = git(["status", "--porcelain"], "source-status");
  if (status.trim() !== "") throw new Error("validation requires an unchanged checkout");
  const head = git(["rev-parse", "HEAD"], "revision").trim();
  const index = git(["ls-files", "-s"], "tracked-content");
  const versions = {};
  for (const tool of ["gcc", "g++", "gccgo", "clang", "node"]) {
    versions[tool] = run(tool, ["--version"], { env, log: join(directory, "version-" + tool) }).stdout;
  }
  versions.go = run(go, ["version"], { env, log: join(directory, "version-go") }).stdout;
  const system = run("uname", ["-a"], { env, log: join(directory, "kernel") }).stdout;
  for (const [path, name] of [["/proc/cpuinfo", "cpuinfo"], ["/proc/meminfo", "meminfo"]]) {
    save(join(directory, name + ".txt"), readFileSync(path));
  }
  const goSettings = JSON.parse(run(go, ["env", "-json", "GOARCH", "GOOS", "GOAMD64",
    "GOARM64", "CC", "CXX", "CGO_ENABLED", "GOVERSION"],
    { env, log: join(directory, "go-settings") }).stdout);
  const fingerprint = { head, sourceIndexSHA256: createHash("sha256").update(index).digest("hex"),
    versions, system, goSettings };
  saveJSON(join(directory, "fingerprint.json"), fingerprint);
  return fingerprint;
}
