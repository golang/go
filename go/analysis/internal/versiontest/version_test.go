// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.21

// Check that GoVersion propagates through to checkers.
// Depends on Go 1.21 go/types.

package versiontest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/multichecker"
	"golang.org/x/tools/go/analysis/singlechecker"
	"golang.org/x/tools/internal/testenv"
)

var analyzer = &analysis.Analyzer{
	Name: "versiontest",
	Doc:  "off",
	Run: func(pass *analysis.Pass) (interface{}, error) {
		pass.Reportf(pass.Files[0].Package, "goversion=%s", pass.Pkg.GoVersion())
		return nil, nil
	},
}

func init() {
	if os.Getenv("VERSIONTEST_MULTICHECKER") == "1" {
		multichecker.Main(analyzer)
		os.Exit(0)
	}
	if os.Getenv("VERSIONTEST_SINGLECHECKER") == "1" {
		singlechecker.Main(analyzer)
		os.Exit(0)
	}
}

func testDir(t *testing.T) (dir string) {
	dir = t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("go 1.20\nmodule m\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "x.go"), []byte("package main // want \"goversion=go1.20\"\n"), 0666); err != nil {
		t.Fatal(err)
	}
	return dir
}

// There are many ways to run analyzers. Test all the ones here in x/tools.

func TestAnalysistest(t *testing.T) {
	analysistest.Run(t, testDir(t), analyzer)
}

func TestMultichecker(t *testing.T) {
	testenv.NeedsGoPackages(t)

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, ".")
	cmd.Dir = testDir(t)
	cmd.Env = append(os.Environ(), "VERSIONTEST_MULTICHECKER=1")
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "x.go:1:1: goversion=go1.20\n") {
		t.Fatalf("multichecker: %v\n%s", err, out)
	}
}

func TestSinglechecker(t *testing.T) {
	testenv.NeedsGoPackages(t)

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, ".")
	cmd.Dir = testDir(t)
	cmd.Env = append(os.Environ(), "VERSIONTEST_SINGLECHECKER=1")
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "x.go:1:1: goversion=go1.20\n") {
		t.Fatalf("multichecker: %v\n%s", err, out)
	}
}

func TestVettool(t *testing.T) {
	testenv.NeedsGoPackages(t)

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "vet", "-vettool="+exe, ".")
	cmd.Dir = testDir(t)
	cmd.Env = append(os.Environ(), "VERSIONTEST_MULTICHECKER=1")
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "x.go:1:1: goversion=go1.20\n") {
		t.Fatalf("vettool: %v\n%s", err, out)
	}
}
