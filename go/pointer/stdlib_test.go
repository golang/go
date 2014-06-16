// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file runs the pointer analysis on all packages and tests beneath
// $GOROOT.  It provides a "smoke test" that the analysis doesn't crash
// on a large input, and a benchmark for performance measurement.
//
// Because it is relatively slow, the --stdlib flag must be enabled for
// this test to run:
//    % go test -v code.google.com/p/go.tools/go/pointer --stdlib

import (
	"flag"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"code.google.com/p/go.tools/go/loader"
	"code.google.com/p/go.tools/go/ssa"
	"code.google.com/p/go.tools/go/ssa/ssautil"
)

var runStdlibTest = flag.Bool("stdlib", false, "Run the (slow) stdlib test")

// TODO(adonovan): move this to go/buildutil package since we have four copies:
// go/{loader,pointer,ssa}/stdlib_test.go and godoc/analysis/analysis.go.
func allPackages() []string {
	var pkgs []string
	root := filepath.Join(runtime.GOROOT(), "src/pkg") + string(os.PathSeparator)
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		// Prune the search if we encounter any of these names:
		switch filepath.Base(path) {
		case "testdata", ".hg":
			return filepath.SkipDir
		}
		if info.IsDir() {
			pkg := filepath.ToSlash(strings.TrimPrefix(path, root))
			switch pkg {
			case "builtin", "pkg":
				return filepath.SkipDir // skip these subtrees
			case "":
				return nil // ignore root of tree
			}
			pkgs = append(pkgs, pkg)
		}

		return nil
	})
	return pkgs
}

func TestStdlib(t *testing.T) {
	if !*runStdlibTest {
		t.Skip("skipping (slow) stdlib test (use --stdlib)")
	}

	// Load, parse and type-check the program.
	var conf loader.Config
	conf.SourceImports = true
	if _, err := conf.FromArgs(allPackages(), true); err != nil {
		t.Errorf("FromArgs failed: %v", err)
		return
	}

	iprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Create SSA packages.
	prog := ssa.Create(iprog, 0)
	prog.BuildAll()

	numPkgs := len(prog.AllPackages())
	if want := 140; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	// Determine the set of packages/tests to analyze.
	var testPkgs []*ssa.Package
	for _, info := range iprog.InitialPackages() {
		testPkgs = append(testPkgs, prog.Package(info.Pkg))
	}
	testmain := prog.CreateTestMainPackage(testPkgs...)
	if testmain == nil {
		t.Fatal("analysis scope has tests")
	}

	// Run the analysis.
	config := &Config{
		Reflection:     false, // TODO(adonovan): fix remaining bug in rVCallConstraint, then enable.
		BuildCallGraph: true,
		Mains:          []*ssa.Package{testmain},
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
