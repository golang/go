// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"log"
	"runtime"
	"time"
)

var Debug int

// Compile is the main entry point for this package.
// Compile modifies f so that on return:
//   · all Values in f map to 0 or 1 assembly instructions of the target architecture
//   · the order of f.Blocks is the order to emit the Blocks
//   · the order of b.Values is the order to emit the Values in each Block
//   · f has a non-nil regAlloc field
func Compile(f *Func) {
	// TODO: debugging - set flags to control verbosity of compiler,
	// which phases to dump IR before/after, etc.
	if f.Log() {
		f.Logf("compiling %s\n", f.Name)
	}

	// hook to print function & phase if panic happens
	phaseName := "init"
	defer func() {
		if phaseName != "" {
			err := recover()
			stack := make([]byte, 16384)
			n := runtime.Stack(stack, false)
			stack = stack[:n]
			f.Fatalf("panic during %s while compiling %s:\n\n%v\n\n%s\n", phaseName, f.Name, err, stack)
		}
	}()

	// Run all the passes
	printFunc(f)
	f.Config.HTML.WriteFunc("start", f)
	checkFunc(f)
	const logMemStats = false
	for _, p := range passes {
		if !f.Config.optimize && !p.required {
			continue
		}
		phaseName = p.name
		if f.Log() {
			f.Logf("  pass %s begin\n", p.name)
		}
		// TODO: capture logging during this pass, add it to the HTML
		var mStart runtime.MemStats
		if logMemStats {
			runtime.ReadMemStats(&mStart)
		}

		tStart := time.Now()
		p.fn(f)

		if f.Log() || f.Config.HTML != nil {
			tEnd := time.Now()

			time := tEnd.Sub(tStart).Nanoseconds()
			var stats string
			if logMemStats {
				var mEnd runtime.MemStats
				runtime.ReadMemStats(&mEnd)
				nBytes := mEnd.TotalAlloc - mStart.TotalAlloc
				nAllocs := mEnd.Mallocs - mStart.Mallocs
				stats = fmt.Sprintf("[%d ns %d allocs %d bytes]", time, nAllocs, nBytes)
			} else {
				stats = fmt.Sprintf("[%d ns]", time)
			}

			f.Logf("  pass %s end %s\n", p.name, stats)
			printFunc(f)
			f.Config.HTML.WriteFunc(fmt.Sprintf("after %s <span class=\"stats\">%s</span>", phaseName, stats), f)
		}
		checkFunc(f)
	}

	// Squash error printing defer
	phaseName = ""
}

type pass struct {
	name     string
	fn       func(*Func)
	required bool
}

// list of passes for the compiler
var passes = [...]pass{
	// TODO: combine phielim and copyelim into a single pass?
	{"early phielim", phielim, false},
	{"early copyelim", copyelim, false},
	{"early deadcode", deadcode, false}, // remove generated dead code to avoid doing pointless work during opt
	{"short circuit", shortcircuit, false},
	{"decompose user", decomposeUser, true},
	{"decompose builtin", decomposeBuiltIn, true},
	{"opt", opt, true},                // TODO: split required rules and optimizing rules
	{"zero arg cse", zcse, true},      // required to merge OpSB values
	{"opt deadcode", deadcode, false}, // remove any blocks orphaned during opt
	{"generic cse", cse, false},
	{"nilcheckelim", nilcheckelim, false},
	{"generic deadcode", deadcode, false},
	{"fuse", fuse, false},
	{"dse", dse, false},
	{"tighten", tighten, false}, // move values closer to their uses
	{"lower", lower, true},
	{"lowered cse", cse, false},
	{"lowered deadcode", deadcode, true},
	{"checkLower", checkLower, true},
	{"late phielim", phielim, false},
	{"late copyelim", copyelim, false},
	{"late deadcode", deadcode, false},
	{"critical", critical, true},   // remove critical edges
	{"layout", layout, true},       // schedule blocks
	{"schedule", schedule, true},   // schedule values
	{"flagalloc", flagalloc, true}, // allocate flags register
	{"regalloc", regalloc, true},   // allocate int & float registers + stack slots
	{"trim", trim, false},          // remove empty blocks
}

// Double-check phase ordering constraints.
// This code is intended to document the ordering requirements
// between different phases.  It does not override the passes
// list above.
type constraint struct {
	a, b string // a must come before b
}

var passOrder = [...]constraint{
	// common-subexpression before dead-store elim, so that we recognize
	// when two address expressions are the same.
	{"generic cse", "dse"},
	// cse substantially improves nilcheckelim efficacy
	{"generic cse", "nilcheckelim"},
	// allow deadcode to clean up after nilcheckelim
	{"nilcheckelim", "generic deadcode"},
	// nilcheckelim generates sequences of plain basic blocks
	{"nilcheckelim", "fuse"},
	// nilcheckelim relies on opt to rewrite user nil checks
	{"opt", "nilcheckelim"},
	// tighten should happen before lowering to avoid splitting naturally paired instructions such as CMP/SET
	{"tighten", "lower"},
	// tighten will be most effective when as many values have been removed as possible
	{"generic deadcode", "tighten"},
	{"generic cse", "tighten"},
	// don't run optimization pass until we've decomposed builtin objects
	{"decompose builtin", "opt"},
	// don't layout blocks until critical edges have been removed
	{"critical", "layout"},
	// regalloc requires the removal of all critical edges
	{"critical", "regalloc"},
	// regalloc requires all the values in a block to be scheduled
	{"schedule", "regalloc"},
	// checkLower must run after lowering & subsequent dead code elim
	{"lower", "checkLower"},
	{"lowered deadcode", "checkLower"},
	// flagalloc needs instructions to be scheduled.
	{"schedule", "flagalloc"},
	// regalloc needs flags to be allocated first.
	{"flagalloc", "regalloc"},
	// trim needs regalloc to be done first.
	{"regalloc", "trim"},
}

func init() {
	for _, c := range passOrder {
		a, b := c.a, c.b
		i := -1
		j := -1
		for k, p := range passes {
			if p.name == a {
				i = k
			}
			if p.name == b {
				j = k
			}
		}
		if i < 0 {
			log.Panicf("pass %s not found", a)
		}
		if j < 0 {
			log.Panicf("pass %s not found", b)
		}
		if i >= j {
			log.Panicf("passes %s and %s out of order", a, b)
		}
	}
}
