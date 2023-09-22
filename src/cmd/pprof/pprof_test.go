// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// TestMain executes the test binary as the pprof command if
// GO_PPROFTEST_IS_PPROF is set, and runs the tests otherwise.
func TestMain(m *testing.M) {
	if os.Getenv("GO_PPROFTEST_IS_PPROF") != "" {
		main()
		os.Exit(0)
	}

	os.Setenv("GO_PPROFTEST_IS_PPROF", "1") // Set for subprocesses to inherit.
	os.Exit(m.Run())
}

// pprofPath returns the path to the "pprof" binary to run.
func pprofPath(t testing.TB) string {
	t.Helper()
	testenv.MustHaveExec(t)

	pprofPathOnce.Do(func() {
		pprofExePath, pprofPathErr = os.Executable()
	})
	if pprofPathErr != nil {
		t.Fatal(pprofPathErr)
	}
	return pprofExePath
}

var (
	pprofPathOnce sync.Once
	pprofExePath  string
	pprofPathErr  error
)

// See also runtime/pprof.cpuProfilingBroken.
func mustHaveCPUProfiling(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping on %s, unimplemented", runtime.GOOS)
	case "aix":
		t.Skipf("skipping on %s, issue 45170", runtime.GOOS)
	case "ios", "dragonfly", "netbsd", "illumos", "solaris":
		t.Skipf("skipping on %s, issue 13841", runtime.GOOS)
	case "openbsd":
		if runtime.GOARCH == "arm" || runtime.GOARCH == "arm64" {
			t.Skipf("skipping on %s/%s, issue 13841", runtime.GOOS, runtime.GOARCH)
		}
	}
}

func mustHaveDisasm(t *testing.T) {
	switch runtime.GOARCH {
	case "loong64":
		t.Skipf("skipping on %s.", runtime.GOARCH)
	case "mips", "mipsle", "mips64", "mips64le":
		t.Skipf("skipping on %s, issue 12559", runtime.GOARCH)
	case "riscv64":
		t.Skipf("skipping on %s, issue 36738", runtime.GOARCH)
	case "s390x":
		t.Skipf("skipping on %s, issue 15255", runtime.GOARCH)
	}

	// pprof can only disassemble PIE on some platforms.
	// Skip the ones it can't handle yet.
	if runtime.GOOS == "android" && runtime.GOARCH == "arm" {
		t.Skipf("skipping on %s/%s, issue 46639", runtime.GOOS, runtime.GOARCH)
	}
}

// TestDisasm verifies that cmd/pprof can successfully disassemble functions.
//
// This is a regression test for issue 46636.
func TestDisasm(t *testing.T) {
	mustHaveCPUProfiling(t)
	mustHaveDisasm(t)
	testenv.MustHaveGoBuild(t)

	tmpdir := t.TempDir()
	cpuExe := filepath.Join(tmpdir, "cpu.exe")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", cpuExe, "cpu.go")
	cmd.Dir = "testdata/"
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}

	profile := filepath.Join(tmpdir, "cpu.pprof")
	cmd = testenv.Command(t, cpuExe, "-output", profile)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("cpu failed: %v\n%s", err, out)
	}

	cmd = testenv.Command(t, pprofPath(t), "-disasm", "main.main", cpuExe, profile)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Errorf("pprof -disasm failed: %v\n%s", err, out)

		// Try to print out profile content for debugging.
		cmd = testenv.Command(t, pprofPath(t), "-raw", cpuExe, profile)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Logf("pprof -raw failed: %v\n%s", err, out)
		} else {
			t.Logf("profile content:\n%s", out)
		}
		return
	}

	sout := string(out)
	want := "ROUTINE ======================== main.main"
	if !strings.Contains(sout, want) {
		t.Errorf("pprof -disasm got %s want contains %q", sout, want)
	}
}
