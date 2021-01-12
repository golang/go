// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"internal/race"
	"math/rand"
	"sort"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/liveness"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/walk"
)

// "Portable" code generation.

var (
	compilequeue []*ir.Func // functions waiting to be compiled
)

func enqueueFunc(fn *ir.Func) {
	if ir.CurFunc != nil {
		base.FatalfAt(fn.Pos(), "enqueueFunc %v inside %v", fn, ir.CurFunc)
	}

	if ir.FuncName(fn) == "_" {
		// Skip compiling blank functions.
		// Frontend already reported any spec-mandated errors (#29870).
		return
	}

	if len(fn.Body) == 0 {
		// Initialize ABI wrappers if necessary.
		ssagen.InitLSym(fn, false)
		liveness.WriteFuncMap(fn)
		return
	}

	errorsBefore := base.Errors()
	prepareFunc(fn)
	if base.Errors() > errorsBefore {
		return
	}

	compilequeue = append(compilequeue, fn)
}

// prepareFunc handles any remaining frontend compilation tasks that
// aren't yet safe to perform concurrently.
func prepareFunc(fn *ir.Func) {
	// Set up the function's LSym early to avoid data races with the assemblers.
	// Do this before walk, as walk needs the LSym to set attributes/relocations
	// (e.g. in markTypeUsedInInterface).
	ssagen.InitLSym(fn, true)

	// Calculate parameter offsets.
	types.CalcSize(fn.Type())

	typecheck.DeclContext = ir.PAUTO
	ir.CurFunc = fn
	walk.Walk(fn)
	ir.CurFunc = nil // enforce no further uses of CurFunc
	typecheck.DeclContext = ir.PEXTERN

	// Make sure type syms are declared for all types that might
	// be types of stack objects. We need to do this here
	// because symbols must be allocated before the parallel
	// phase of the compiler.
	for _, n := range fn.Dcl {
		switch n.Class {
		case ir.PPARAM, ir.PPARAMOUT, ir.PAUTO:
			if liveness.ShouldTrack(n) && n.Addrtaken() {
				reflectdata.WriteType(n.Type())
				// Also make sure we allocate a linker symbol
				// for the stack object data, for the same reason.
				if fn.LSym.Func().StackObjects == nil {
					fn.LSym.Func().StackObjects = base.Ctxt.Lookup(fn.LSym.Name + ".stkobj")
				}
			}
		}
	}
}

// compileFunctions compiles all functions in compilequeue.
// It fans out nBackendWorkers to do the work
// and waits for them to complete.
func compileFunctions() {
	if len(compilequeue) == 0 {
		return
	}

	types.CalcSizeDisabled = true // not safe to calculate sizes concurrently
	if race.Enabled {
		// Randomize compilation order to try to shake out races.
		tmp := make([]*ir.Func, len(compilequeue))
		perm := rand.Perm(len(compilequeue))
		for i, v := range perm {
			tmp[v] = compilequeue[i]
		}
		copy(compilequeue, tmp)
	} else {
		// Compile the longest functions first,
		// since they're most likely to be the slowest.
		// This helps avoid stragglers.
		sort.Slice(compilequeue, func(i, j int) bool {
			return len(compilequeue[i].Body) > len(compilequeue[j].Body)
		})
	}
	var wg sync.WaitGroup
	base.Ctxt.InParallel = true
	c := make(chan *ir.Func, base.Flag.LowerC)
	for i := 0; i < base.Flag.LowerC; i++ {
		wg.Add(1)
		go func(worker int) {
			for fn := range c {
				ssagen.Compile(fn, worker)
			}
			wg.Done()
		}(i)
	}
	for _, fn := range compilequeue {
		c <- fn
	}
	close(c)
	compilequeue = nil
	wg.Wait()
	base.Ctxt.InParallel = false
	types.CalcSizeDisabled = false
}
