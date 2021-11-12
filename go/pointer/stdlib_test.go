// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incomplete source tree on Android.

//go:build !android
// +build !android

package pointer

// This file runs the pointer analysis on all packages and tests beneath
// $GOROOT.  It provides a "smoke test" that the analysis doesn't crash
// on a large input, and a benchmark for performance measurement.
//
// Because it is relatively slow, the --stdlib flag must be enabled for
// this test to run:
//    % go test -v golang.org/x/tools/go/pointer --stdlib

import (
	"flag"
	"go/token"
	"testing"
	"time"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

var runStdlibTest = flag.Bool("stdlib", false, "Run the (slow) stdlib test")

func TestStdlib(t *testing.T) {
	if !*runStdlibTest {
		t.Skip("skipping (slow) stdlib test (use --stdlib)")
	}

	cfg := &packages.Config{
		Mode: packages.LoadAllSyntax,
		// Create test main packages with a main function.
		Tests: true,
	}
	pkgs, err := packages.Load(cfg, "std")
	if err != nil || packages.PrintErrors(pkgs) > 0 {
		t.Fatalf("Load failed: %v", err)
	}

	// Create SSA packages.
	prog, _ := ssautil.AllPackages(pkgs, 0)
	prog.Build()

	numPkgs := len(prog.AllPackages())
	if want := 240; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	// Determine the set of packages/tests to analyze.
	var mains []*ssa.Package
	for _, ssapkg := range prog.AllPackages() {
		if ssapkg.Pkg.Name() == "main" && ssapkg.Func("main") != nil {
			mains = append(mains, ssapkg)
		}
	}
	if mains == nil {
		t.Fatal("no tests found in analysis scope")
	}

	// Run the analysis.
	config := &Config{
		Reflection:     false, // TODO(adonovan): fix remaining bug in rVCallConstraint, then enable.
		BuildCallGraph: true,
		Mains:          mains,
	}
	// TODO(adonovan): add some query values (affects track bits).

	t0 := time.Now()

	result, err := Analyze(config)
	if err != nil {
		t.Fatal(err) // internal error in pointer analysis
	}
	_ = result // TODO(adonovan): measure something

	t1 := time.Now()

	// Dump some statistics.
	allFuncs := ssautil.AllFunctions(prog)
	var numInstrs int
	for fn := range allFuncs {
		for _, b := range fn.Blocks {
			numInstrs += len(b.Instrs)
		}
	}

	// determine line count
	var lineCount int
	prog.Fset.Iterate(func(f *token.File) bool {
		lineCount += f.LineCount()
		return true
	})

	t.Log("#Source lines:          ", lineCount)
	t.Log("#Instructions:          ", numInstrs)
	t.Log("Pointer analysis:       ", t1.Sub(t0))
}
