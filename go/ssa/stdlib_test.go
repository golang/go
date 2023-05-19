// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incomplete source tree on Android.

//go:build !android
// +build !android

package ssa_test

// This file runs the SSA builder in sanity-checking mode on all
// packages beneath $GOROOT and prints some summary information.
//
// Run with "go test -cpu=8 to" set GOMAXPROCS.

import (
	"go/ast"
	"go/token"
	"runtime"
	"testing"
	"time"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/internal/testenv"
)

func bytesAllocated() uint64 {
	runtime.GC()
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return stats.Alloc
}

func TestStdlib(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode; too slow (https://golang.org/issue/14113)") // ~5s
	}
	testenv.NeedsTool(t, "go")

	// Load, parse and type-check the program.
	t0 := time.Now()
	alloc0 := bytesAllocated()

	cfg := &packages.Config{Mode: packages.LoadSyntax}
	pkgs, err := packages.Load(cfg, "std", "cmd")
	if err != nil {
		t.Fatal(err)
	}

	t1 := time.Now()
	alloc1 := bytesAllocated()

	// Create SSA packages.
	var mode ssa.BuilderMode
	// Comment out these lines during benchmarking.  Approx SSA build costs are noted.
	mode |= ssa.SanityCheckFunctions // + 2% space, + 4% time
	mode |= ssa.GlobalDebug          // +30% space, +18% time
	mode |= ssa.InstantiateGenerics  // + 0% space, + 2% time (unlikely to reproduce outside of stdlib)
	prog, _ := ssautil.Packages(pkgs, mode)

	t2 := time.Now()

	// Build SSA.
	prog.Build()

	t3 := time.Now()
	alloc3 := bytesAllocated()

	numPkgs := len(prog.AllPackages())
	if want := 140; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	// Keep pkgs reachable until after we've measured memory usage.
	if len(pkgs) == 0 {
		panic("unreachable")
	}

	allFuncs := ssautil.AllFunctions(prog)

	// The assertion below is not valid if the program contains
	// variants of the same package, such as the test variants
	// (e.g. package p as compiled for test executable x) obtained
	// when cfg.Tests=true. Profile-guided optimization may
	// lead to similar variation for non-test executables.
	//
	// Ideally, the test would assert that all functions within
	// each executable (more generally: within any singly rooted
	// transitively closed subgraph of the import graph) have
	// distinct names, but that isn't so easy to compute efficiently.
	// Disabling for now.
	if false {
		// Check that all non-synthetic functions have distinct names.
		// Synthetic wrappers for exported methods should be distinct too,
		// except for unexported ones (explained at (*Function).RelString).
		byName := make(map[string]*ssa.Function)
		for fn := range allFuncs {
			if fn.Synthetic == "" || ast.IsExported(fn.Name()) {
				str := fn.String()
				prev := byName[str]
				byName[str] = fn
				if prev != nil {
					t.Errorf("%s: duplicate function named %s",
						prog.Fset.Position(fn.Pos()), str)
					t.Errorf("%s:   (previously defined here)",
						prog.Fset.Position(prev.Pos()))
				}
			}
		}
	}

	// Dump some statistics.
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
	t.Log("#MB AST+types:        ", int64(alloc1-alloc0)/1e6)
	t.Log("#MB SSA:              ", int64(alloc3-alloc1)/1e6)
}
