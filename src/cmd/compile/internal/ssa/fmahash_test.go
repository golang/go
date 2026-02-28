// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	"internal/testenv"
	"path/filepath"
	"regexp"
	"runtime"
	"testing"
)

// TestFmaHash checks that the hash-test machinery works properly for a single case.
// It also runs ssa/check and gccheck to be sure that those are checked at least a
// little in each run.bash.  It does not check or run the generated code.
// The test file is however a useful example of fused-vs-cascaded multiply-add.
func TestFmaHash(t *testing.T) {
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
	tmpdir := t.TempDir()
	source := filepath.Join("testdata", "fma.go")
	output := filepath.Join(tmpdir, "fma.exe")
	cmd := testenv.Command(t, gocmd, "build", "-o", output, source)
	// The hash-dependence on file path name is dodged by specifying "all hashes ending in 1" plus "all hashes ending in 0"
	// i.e., all hashes.  This will print all the FMAs; this test is only interested in one of them (that should appear near the end).
	cmd.Env = append(cmd.Env, "GOCOMPILEDEBUG=fmahash=1/0", "GOOS=linux", "GOARCH=arm64", "HOME="+tmpdir)
	t.Logf("%v", cmd)
	t.Logf("%v", cmd.Env)
	b, e := cmd.CombinedOutput()
	if e != nil {
		t.Errorf("build failed: %v\n%s", e, b)
	}
	s := string(b) // Looking for "GOFMAHASH triggered main.main:24"
	re := "fmahash(0?) triggered .*fma.go:29:..;.*fma.go:18:.."
	match := regexp.MustCompile(re)
	if !match.MatchString(s) {
		t.Errorf("Expected to match '%s' with \n-----\n%s-----", re, s)
	}
}
