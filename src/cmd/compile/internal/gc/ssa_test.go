// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"internal/testenv"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TODO: move all these tests elsewhere?
// Perhaps teach test/run.go how to run them with a new action verb.
func runTest(t *testing.T, filename string) {
	if runtime.GOARCH != "amd64" {
		t.Skipf("skipping SSA tests on %s for now", runtime.GOARCH)
	}
	testenv.MustHaveGoBuild(t)
	var stdout, stderr bytes.Buffer
	cmd := exec.Command("go", "run", filepath.Join("testdata", filename))
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	// TODO: set GOGC=off until we have stackmaps
	if err := cmd.Run(); err != nil {
		t.Fatalf("Failed: %v:\nOut: %s\nStderr: %s\n", err, &stdout, &stderr)
	}
	if s := stdout.String(); s != "" {
		t.Errorf("Stdout = %s\nWant empty", s)
	}
	if s := stderr.String(); strings.Contains(s, "SSA unimplemented") {
		t.Errorf("Unimplemented message found in stderr:\n%s", s)
	}
}

// TestShortCircuit tests OANDAND and OOROR expressions and short circuiting.
func TestShortCircuit(t *testing.T) { runTest(t, "short_ssa.go") }

// TestBreakContinue tests that continue and break statements do what they say.
func TestBreakContinue(t *testing.T) { runTest(t, "break_ssa.go") }
