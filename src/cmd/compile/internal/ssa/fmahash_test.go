// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestFmaHash checks that the hash-test machinery works properly for a single case.
// It does not check or run the generated code.
// The test file is however a useful example of fused-vs-cascaded multiply-add.
func TestFmaHash(t *testing.T) {
	if testing.Short() {
		t.Skip("Slow test, usually avoid it, testing.Short")
	}
	switch runtime.GOOS {
	case "linux", "darwin":
	default:
		t.Skipf("Slow test, usually avoid it, os=%s not linux or darwin", runtime.GOOS)
	}
	switch runtime.GOARCH {
	case "amd64", "arm64":
	default:
		t.Skipf("Slow test, usually avoid it, arch=%s not amd64 or arm64", runtime.GOARCH)
	}

	testenv.MustHaveGoBuild(t)
	gocmd := testenv.GoToolPath(t)
	tmpdir, err := os.MkdirTemp("", "x")
	if err != nil {
		t.Error(err)
	}
	defer os.RemoveAll(tmpdir)
	source := filepath.Join("testdata", "fma.go")
	output := filepath.Join(tmpdir, "fma.exe")
	cmd := exec.Command(gocmd, "build", "-o", output, source)
	cmd.Env = append(cmd.Env, "GOCOMPILEDEBUG=fmahash=101111101101111001110110", "GOOS=linux", "GOARCH=arm64", "HOME="+tmpdir)
	t.Logf("%v", cmd)
	t.Logf("%v", cmd.Env)
	b, e := cmd.CombinedOutput()
	if e != nil {
		t.Error(e)
	}
	s := string(b) // Looking for "GOFMAHASH triggered main.main:24"
	if !strings.Contains(s, "fmahash triggered main.main:24") {
		t.Errorf("Expected to see 'fmahash triggered main.main:24' in \n-----\n%s-----", s)
	}
}
