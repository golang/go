// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"internal/testenv"
	"os"
	"os/exec"
	"testing"
)

// Test that the generated code for the lock rank graph is up-to-date.
func TestLockRankGenerated(t *testing.T) {
	testenv.MustHaveGoRun(t)
	cmd := testenv.CleanCmdEnv(testenv.Command(t, testenv.GoToolPath(t), "run", "mklockrank.go"))
	want, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
			t.Fatalf("%v: %v\n%s", cmd, err, ee.Stderr)
		}
		t.Fatalf("%v: %v", cmd, err)
	}
	got, err := os.ReadFile("lockrank.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(want, got) {
		t.Fatalf("lockrank.go is out of date. Please run go generate.")
	}
}
