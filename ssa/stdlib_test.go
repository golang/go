// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

// This file runs the SSA builder in sanity-checking mode on all
// packages beneath $GOROOT and prints some summary information.
//
// Run test with GOMAXPROCS=8.

import (
	"go/build"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
	"code.google.com/p/go.tools/ssa/ssautil"
)

const debugMode = false

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
			pkg := strings.TrimPrefix(path, root)
			switch pkg {
			case "builtin", "pkg", "code.google.com":
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
	impctx := importer.Config{Build: &build.Default}

	// Load, parse and type-check the program.
	t0 := time.Now()

	imp := importer.New(&impctx)

	if _, _, err := imp.LoadInitialPackages(allPackages()); err != nil {
		t.Errorf("LoadInitialPackages failed: %s", err)
		return
	}

	t1 := time.Now()

	runtime.GC()
	var memstats runtime.MemStats
	runtime.ReadMemStats(&memstats)
	alloc := memstats.Alloc

	// Create SSA packages.
	prog := ssa.NewProgram(imp.Fset, ssa.SanityCheckFunctions)
	if err := prog.CreatePackages(imp); err != nil {
		t.Errorf("CreatePackages failed: %s", err)
		return
	}
	// Enable debug mode globally.
	for _, info := range imp.AllPackages() {
		prog.Package(info.Pkg).SetDebugMode(debugMode)
	}

	t2 := time.Now()

	// Build SSA IR... if it's safe.
	prog.BuildAll()

	t3 := time.Now()

	runtime.GC()
	runtime.ReadMemStats(&memstats)

	numPkgs := len(prog.AllPackages())
	if want := 140; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

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
	imp.Fset.Iterate(func(f *token.File) bool {
		lineCount += f.LineCount()
		return true
	})

	// NB: when benchmarking, don't forget to clear the debug +
	// sanity builder flags for better performance.

	t.Log("GOMAXPROCS:           ", runtime.GOMAXPROCS(0))
	t.Log("#Source lines:        ", lineCount)
	t.Log("Load/parse/typecheck: ", t1.Sub(t0))
	t.Log("SSA create:           ", t2.Sub(t1))
	t.Log("SSA build:            ", t3.Sub(t2))

	// SSA stats:
	t.Log("#Packages:            ", numPkgs)
	t.Log("#Functions:           ", len(allFuncs))
	t.Log("#Instructions:        ", numInstrs)
	t.Log("#MB:                  ", (memstats.Alloc-alloc)/1000000)
}
