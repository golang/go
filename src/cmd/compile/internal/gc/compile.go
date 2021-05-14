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
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/walk"
	"cmd/internal/obj"
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

	if clo := fn.OClosure; clo != nil && !ir.IsTrivialClosure(clo) {
		return // we'll get this as part of its enclosing function
	}

	if len(fn.Body) == 0 {
		// Initialize ABI wrappers if necessary.
		ssagen.InitLSym(fn, false)
		types.CalcSize(fn.Type())
		a := ssagen.AbiForBodylessFuncStackMap(fn)
		abiInfo := a.ABIAnalyzeFuncType(fn.Type().FuncType()) // abiInfo has spill/home locations for wrapper
		liveness.WriteFuncMap(fn, abiInfo)
		if fn.ABI == obj.ABI0 {
			x := ssagen.EmitArgInfo(fn, abiInfo)
			objw.Global(x, int32(len(x.P)), obj.RODATA|obj.LOCAL)
		}
		return
	}

	errorsBefore := base.Errors()

	todo := []*ir.Func{fn}
	for len(todo) > 0 {
		next := todo[len(todo)-1]
		todo = todo[:len(todo)-1]

		prepareFunc(next)
		todo = append(todo, next.Closures...)
	}

	if base.Errors() > errorsBefore {
		return
	}

	// Enqueue just fn itself. compileFunctions will handle
	// scheduling compilation of its closures after it's done.
	compilequeue = append(compilequeue, fn)
}

// prepareFunc handles any remaining frontend compilation tasks that
// aren't yet safe to perform concurrently.
func prepareFunc(fn *ir.Func) {
	// Set up the function's LSym early to avoid data races with the assemblers.
	// Do this before walk, as walk needs the LSym to set attributes/relocations
	// (e.g. in MarkTypeUsedInInterface).
	ssagen.InitLSym(fn, true)

	// Calculate parameter offsets.
	types.CalcSize(fn.Type())

	typecheck.DeclContext = ir.PAUTO
	ir.CurFunc = fn
	walk.Walk(fn)
	ir.CurFunc = nil // enforce no further uses of CurFunc
	typecheck.DeclContext = ir.PEXTERN
}

// compileFunctions compiles all functions in compilequeue.
// It fans out nBackendWorkers to do the work
// and waits for them to complete.
func compileFunctions() {
	if len(compilequeue) == 0 {
		return
	}

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

	// By default, we perform work right away on the current goroutine
	// as the solo worker.
	queue := func(work func(int)) {
		work(0)
	}

	if nWorkers := base.Flag.LowerC; nWorkers > 1 {
		// For concurrent builds, we create a goroutine per task, but
		// require them to hold a unique worker ID while performing work
		// to limit parallelism.
		workerIDs := make(chan int, nWorkers)
		for i := 0; i < nWorkers; i++ {
			workerIDs <- i
		}

		queue = func(work func(int)) {
			go func() {
				worker := <-workerIDs
				work(worker)
				workerIDs <- worker
			}()
		}
	}

	var wg sync.WaitGroup
	var compile func([]*ir.Func)
	compile = func(fns []*ir.Func) {
		wg.Add(len(fns))
		for _, fn := range fns {
			fn := fn
			queue(func(worker int) {
				ssagen.Compile(fn, worker)
				compile(fn.Closures)
				wg.Done()
			})
		}
	}

	types.CalcSizeDisabled = true // not safe to calculate sizes concurrently
	base.Ctxt.InParallel = true

	compile(compilequeue)
	compilequeue = nil
	wg.Wait()

	base.Ctxt.InParallel = false
	types.CalcSizeDisabled = false
}
