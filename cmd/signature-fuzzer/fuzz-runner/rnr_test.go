// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func canRace(t *testing.T) bool {
	_, err := exec.Command("go", "run", "-race", "./testdata/himom.go").CombinedOutput()
	return err == nil
}

// buildRunner builds the fuzz-runner executable, returning its path.
func buildRunner(t *testing.T) string {
	bindir := filepath.Join(t.TempDir(), "bin")
	err := os.Mkdir(bindir, os.ModePerm)
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(bindir, "runner")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", binary)
	if err := cmd.Run(); err != nil {
		t.Fatalf("Building fuzz-runner: %v", err)
	}
	return binary
}

// TestRunner builds the binary, then kicks off a collection of sub-tests that invoke it.
func TestRunner(t *testing.T) {
	testenv.NeedsTool(t, "go")
	if runtime.GOOS == "android" {
		t.Skipf("the dependencies are not available on android")
	}
	binaryPath := buildRunner(t)

	// Sub-tests using the binary built above.
	t.Run("Basic", func(t *testing.T) { testBasic(t, binaryPath) })
	t.Run("Race", func(t *testing.T) { testRace(t, binaryPath) })
	t.Run("Minimization1", func(t *testing.T) { testMinimization1(t, binaryPath) })
	t.Run("Minimization2", func(t *testing.T) { testMinimization2(t, binaryPath) })
}

func testBasic(t *testing.T, binaryPath string) {
	t.Parallel()
	args := []string{"-numit=1", "-numfcns=1", "-numpkgs=1", "-seed=103", "-cleancache=0"}
	c := exec.Command(binaryPath, args...)
	b, err := c.CombinedOutput()
	t.Logf("%s\n", b)
	if err != nil {
		t.Fatalf("error invoking fuzz-runner: %v", err)
	}
}

func testRace(t *testing.T, binaryPath string) {
	t.Parallel()
	// For this test to work, the current test platform has to support the
	// race detector. Check to see if that is the case by running a very
	// simple Go program through it.
	if !canRace(t) {
		t.Skip("current platform does not appear to support the race detector")
	}

	args := []string{"-v=1", "-numit=1", "-race", "-numfcns=3", "-numpkgs=3", "-seed=987", "-cleancache=0"}
	c := exec.Command(binaryPath, args...)
	b, err := c.CombinedOutput()
	t.Logf("%s\n", b)
	if err != nil {
		t.Fatalf("error invoking fuzz-runner: %v", err)
	}
}

func testMinimization1(t *testing.T, binaryPath string) {
	if binaryPath == "" {
		t.Skipf("No runner binary")
	}
	t.Parallel()
	// Fire off the runner passing it -emitbad=1, so that the generated code
	// contains illegal Go code (which will force the build to fail). Verify that
	// it does fail, that the error reflects the nature of the failure, and that
	// we can minimize the error down to a single package.
	args := []string{"-emitbad=1", "-badfcnidx=2", "-badpkgidx=2",
		"-forcetmpclean", "-cleancache=0",
		"-numit=1", "-numfcns=3", "-numpkgs=3", "-seed=909"}
	invocation := fmt.Sprintf("%s %v", binaryPath, args)
	c := exec.Command(binaryPath, args...)
	b, err := c.CombinedOutput()
	t.Logf("%s\n", b)
	if err == nil {
		t.Fatalf("unexpected pass of fuzz-runner (invocation %q): %v", invocation, err)
	}
	result := string(b)
	if !strings.Contains(result, "syntax error") {
		t.Fatalf("-emitbad=1 did not trigger syntax error (invocation %q): output: %s", invocation, result)
	}
	if !strings.Contains(result, "package minimization succeeded: found bad pkg 2") {
		t.Fatalf("failed to minimize package (invocation %q): output: %s", invocation, result)
	}
	if !strings.Contains(result, "function minimization succeeded: found bad fcn 2") {
		t.Fatalf("failed to minimize package (invocation %q): output: %s", invocation, result)
	}
}

func testMinimization2(t *testing.T, binaryPath string) {
	if binaryPath == "" {
		t.Skipf("No runner binary")
	}
	t.Parallel()
	// Fire off the runner passing it -emitbad=2, so that the
	// generated code forces a runtime error. Verify that it does
	// fail, and that the error is reflective.
	args := []string{"-emitbad=2", "-badfcnidx=1", "-badpkgidx=1",
		"-forcetmpclean", "-cleancache=0",
		"-numit=1", "-numfcns=3", "-numpkgs=3", "-seed=55909"}
	invocation := fmt.Sprintf("%s %v", binaryPath, args)
	c := exec.Command(binaryPath, args...)
	b, err := c.CombinedOutput()
	t.Logf("%s\n", b)
	if err == nil {
		t.Fatalf("unexpected pass of fuzz-runner (invocation %q): %v", invocation, err)
	}
	result := string(b)
	if !strings.Contains(result, "Error: fail") || !strings.Contains(result, "Checker1.Test1") {
		t.Fatalf("-emitbad=2 did not trigger runtime error (invocation %q): output: %s", invocation, result)
	}
	if !strings.Contains(result, "package minimization succeeded: found bad pkg 1") {
		t.Fatalf("failed to minimize package (invocation %q): output: %s", invocation, result)
	}
	if !strings.Contains(result, "function minimization succeeded: found bad fcn 1") {
		t.Fatalf("failed to minimize package (invocation %q): output: %s", invocation, result)
	}
}
