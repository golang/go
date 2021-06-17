// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var tmp, pprofExe string // populated by buildPprof

func TestMain(m *testing.M) {
	if !testenv.HasGoBuild() {
		return
	}

	var exitcode int
	if err := buildPprof(); err == nil {
		exitcode = m.Run()
	} else {
		fmt.Println(err)
		exitcode = 1
	}
	os.RemoveAll(tmp)
	os.Exit(exitcode)
}

func buildPprof() error {
	var err error
	tmp, err = os.MkdirTemp("", "TestPprof")
	if err != nil {
		return fmt.Errorf("TempDir failed: %v", err)
	}

	pprofExe = filepath.Join(tmp, "testpprof.exe")
	gotool, err := testenv.GoTool()
	if err != nil {
		return err
	}
	out, err := exec.Command(gotool, "build", "-o", pprofExe, "cmd/pprof").CombinedOutput()
	if err != nil {
		os.RemoveAll(tmp)
		return fmt.Errorf("go build -o %v cmd/pprof: %v\n%s", pprofExe, err, string(out))
	}

	return nil
}

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
	case "mips", "mipsle", "mips64", "mips64le":
		t.Skipf("skipping on %s, issue 12559", runtime.GOARCH)
	case "riscv64":
		t.Skipf("skipping on %s, issue 36738", runtime.GOARCH)
	case "s390x":
		t.Skipf("skipping on %s, issue 15255", runtime.GOARCH)
	}

	// Skip PIE platforms, pprof can't disassemble PIE.
	if runtime.GOOS == "windows" {
		t.Skipf("skipping on %s, issue 46639", runtime.GOOS)
	}
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
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
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", cpuExe, "cpu.go")
	cmd.Dir = "testdata/"
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}

	profile := filepath.Join(tmpdir, "cpu.pprof")
	cmd = exec.Command(cpuExe, "-output", profile)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("cpu failed: %v\n%s", err, out)
	}

	cmd = exec.Command(pprofExe, "-disasm", "main.main", cpuExe, profile)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("pprof failed: %v\n%s", err, out)
	}

	sout := string(out)
	want := "ROUTINE ======================== main.main"
	if !strings.Contains(sout, want) {
		t.Errorf("pprof disasm got %s want contains %q", sout, want)
	}
}
