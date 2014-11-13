// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

// This file runs the SSA builder in sanity-checking mode on all
// packages beneath $GOROOT and prints some summary information.
//
// Run with "go test -cpu=8 to" set GOMAXPROCS.

import (
	"go/build"
	"go/token"
	"runtime"
	"testing"
	"time"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

func bytesAllocated() uint64 {
	runtime.GC()
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return stats.Alloc
}

func TestStdlib(t *testing.T) {
	// Load, parse and type-check the program.
	t0 := time.Now()
	alloc0 := bytesAllocated()

	// Load, parse and type-check the program.
	ctxt := build.Default // copy
	ctxt.GOPATH = ""      // disable GOPATH
	conf := loader.Config{
		SourceImports: true,
		Build:         &ctxt,
	}
	if _, err := conf.FromArgs(buildutil.AllPackages(conf.Build), true); err != nil {
		t.Errorf("FromArgs failed: %v", err)
		return
	}

	iprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	t1 := time.Now()
	alloc1 := bytesAllocated()

	// Create SSA packages.
	var mode ssa.BuilderMode
	// Comment out these lines during benchmarking.  Approx SSA build costs are noted.
	mode |= ssa.SanityCheckFunctions // + 2% space, + 4% time
	mode |= ssa.GlobalDebug          // +30% space, +18% time
	prog := ssa.Create(iprog, mode)

	t2 := time.Now()

	// Build SSA.
	prog.BuildAll()

	t3 := time.Now()
	alloc3 := bytesAllocated()

	numPkgs := len(prog.AllPackages())
	if want := 140; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	// Keep iprog reachable until after we've measured memory usage.
	if len(iprog.AllPackages) == 0 {
		print() // unreachable
	}

	allFuncs := ssautil.AllFunctions(prog)

	// Check that all non-synthetic functions have distinct names.
	byName := make(map[string]*ssa.Function)
	for fn := range allFuncs {
		if fn.Synthetic == "" {
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
