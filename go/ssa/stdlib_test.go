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
	"runtime"
	"testing"
	"time"

	"code.google.com/p/go.tools/go/buildutil"
	"code.google.com/p/go.tools/go/loader"
	"code.google.com/p/go.tools/go/ssa"
	"code.google.com/p/go.tools/go/ssa/ssautil"
)

func TestStdlib(t *testing.T) {
	// Load, parse and type-check the program.
	t0 := time.Now()

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

	runtime.GC()
	var memstats runtime.MemStats
	runtime.ReadMemStats(&memstats)
	alloc := memstats.Alloc

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

	runtime.GC()
	runtime.ReadMemStats(&memstats)

	numPkgs := len(prog.AllPackages())
	if want := 140; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
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
	t.Log("#MB:                  ", int64(memstats.Alloc-alloc)/1000000)
}
