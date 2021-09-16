// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

// buildDriver builds the fuzz-driver executable, returning its path.
func buildDriver(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "android" {
		t.Skipf("the dependencies are not available on android")
		return ""
	}
	bindir := filepath.Join(t.TempDir(), "bin")
	err := os.Mkdir(bindir, os.ModePerm)
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(bindir, "driver")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", binary)
	if err := cmd.Run(); err != nil {
		t.Fatalf("Building fuzz-driver: %v", err)
	}
	return binary
}

func TestEndToEndIntegration(t *testing.T) {
	testenv.NeedsTool(t, "go")
	td := t.TempDir()

	// Build the fuzz-driver binary.
	// Note: if more tests are added to this package, move this to single setup fcn, so
	// that we don't have to redo the build each time.
	binary := buildDriver(t)

	// Kick off a run.
	gendir := filepath.Join(td, "gen")
	args := []string{"-numfcns", "3", "-numpkgs", "1", "-seed", "101", "-outdir", gendir}
	c := exec.Command(binary, args...)
	b, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("error invoking fuzz-driver: %v\n%s", err, b)
	}

	found := ""
	walker := func(path string, info os.FileInfo, err error) error {
		found = found + ":" + info.Name()
		return nil
	}

	// Make sure it emits something.
	err2 := filepath.Walk(gendir, walker)
	if err2 != nil {
		t.Fatalf("error from filepath.Walk: %v", err2)
	}
	const expected = ":gen:genCaller0:genCaller0.go:genChecker0:genChecker0.go:genMain.go:genUtils:genUtils.go:go.mod"
	if found != expected {
		t.Errorf("walk of generated code: got %s want %s", found, expected)
	}
}
