// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { run } from "./common.mjs";

test("command paths and arguments remain literal across the process boundary", (t) => {
  const directory = mkdtempSync(join(tmpdir(), "logos-cgo 'quoted' $()-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const command = join(directory, "node with spaces;literal");
  symlinkSync(process.execPath, command);
  const marker = join(directory, "must-not-exist");
  const values = ["a b", "'quoted'", "\"quoted\"", "$HOME", "$(touch " + marker + ")",
    "; touch " + marker, "`touch " + marker + "`", "line\nbreak"];
  const log = join(directory, "command");
  const result = run(command,
    ["-e", "process.stdout.write(JSON.stringify(process.argv.slice(1)))", "--", ...values],
    { env: { ...process.env }, cwd: directory, log });
  assert.equal(result.status, 0);
  assert.deepEqual(JSON.parse(result.stdout), values);
  assert.equal(existsSync(marker), false);
  const recorded = JSON.parse(readFileSync(log + ".json", "utf8"));
  assert.equal(recorded.command, command);
  assert.deepEqual(recorded.args.slice(3), values);
});
