// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godebug_test

import (
	"bytes"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

// Test that using a GODEBUG setting does not link in the runtime metrics implementation.
func TestLinkerGC(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	dir := t.TempDir()
	src := filepath.Join(dir, "main.go")
	program := []byte(`package main

import _ "net"

func main() {}
`)
	if err := os.WriteFile(src, program, 0644); err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(dir, "main.exe")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}

	cmd = testenv.Command(t, testenv.GoToolPath(t), "tool", "nm", exe)
	nm, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("nm: %v\n%s", err, nm)
	}
	if want := []byte("internal/godebug.(*Setting).Value"); !bytes.Contains(nm, want) {
		t.Fatalf("expected symbol %q not found", want)
	}
	if bad := []byte("runtime.initMetrics"); bytes.Contains(nm, bad) {
		t.Fatalf("unexpected symbol %q found", bad)
	}
}
