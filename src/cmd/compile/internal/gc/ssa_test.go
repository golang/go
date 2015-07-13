// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"internal/testenv"
	"os/exec"
	"runtime"
	"strings"
	"testing"
)

// Tests OANDAND and OOROR expressions and short circuiting.
// TODO: move these tests elsewhere? perhaps teach test/run.go how to run them
// with a new action verb.
func TestShortCircuit(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skipf("skipping SSA tests on %s for now", runtime.GOARCH)
	}
	testenv.MustHaveGoBuild(t)
	var stdout, stderr bytes.Buffer
	cmd := exec.Command("go", "run", "testdata/short_ssa.go")
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
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
