// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package verylongtest

import (
	"bytes"
	"internal/testenv"
	"os"
	"os/exec"
	"runtime"
	"testing"
)

// Regression test for golang.org/issue/34499: version command should not crash
// when executed in a deleted directory on Linux.
func TestExecInDeletedDir(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9",
		"aix",                // Fails with "device busy".
		"solaris", "illumos": // Fails with "invalid argument".
		t.Skipf("%v does not support removing the current working directory", runtime.GOOS)
	}
	gotool := testenv.GoToolPath(t)

	tmpdir := t.TempDir()
	t.Chdir(tmpdir)

	if err := os.Remove(tmpdir); err != nil {
		t.Fatal(err)
	}

	// `go version` should not fail
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(gotool, "version")
	cmd.Env = append(os.Environ(), "GO111MODULE=off") // This behavior doesn't apply with GO111MODULE != off because we need to know the module to check the version.
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("running go version: %v\n[stdout]: %s\n[stderr]: %s", err, stdout.Bytes(), stderr.Bytes())
	}
}
