// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmp"
	"internal/race"
	"math/rand"
	"slices"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/liveness"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/pgoir"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/staticinit"
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

	if fn.IsClosure() {
		return // we'll get this as part of its enclosing function
	}

	if ssagen.CreateWasmImportWrapper(fn) {
		return
	}

	if len(fn.Body) == 0 {
		// Initialize ABI wrappers if necessary.
		ir.InitLSym(fn, false)
		types.CalcSize(fn.Type())
		a := ssagen.AbiForBodylessFuncStackMap(fn)
		abiInfo := a.ABIAnalyzeFuncType(fn.Type()) // abiInfo has spill/home locations for wrapper
		if fn.ABI == obj.ABI0 {
			// The current args_stackmap generation assumes the function
			// is ABI0, and only ABI0 assembly function can have a FUNCDATA
			// reference to args_stackmap (see cmd/internal/obj/plist.go:Flushplist).
			// So avoid introducing an args_stackmap if the func is not ABI0.
			liveness.WriteFuncMap(fn, abiInfo)

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
	ir.InitLSym(fn, true)

	// If this function is a compiler-generated outlined global map
	// initializer function, register its LSym for later processing.
	if staticinit.MapInitToVar != nil {
		if _, ok := staticinit.MapInitToVar[fn]; ok {
			ssagen.RegisterMapInitLsym(fn.Linksym())
		}
	}

	// Calculate parameter offsets.
	types.CalcSize(fn.Type())

	// Generate wrappers between Go ABI and Wasm ABI, for a wasmexport
	// function.
	// Must be done after InitLSym and CalcSize.
	ssagen.GenWasmExportWrapper(fn)

	ir.CurFunc = fn
	walk.Walk(fn)
	ir.CurFunc = nil // enforce no further uses of CurFunc
}

// compileFunctions compiles all functions in compilequeue.
// It fans out nBackendWorkers to do the work
// and waits for them to complete.
func compileFunctions(profile *pgoir.Profile) {
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
		slices.SortFunc(compilequeue, func(a, b *ir.Func) int {
			return cmp.Compare(len(b.Body), len(a.Body))
		})
	}

	// By default, we perform work right away on the current goroutine
	// as the solo worker.
	queue := func(work func(int)) {
		work(0)
	}

	if nWorkers := base.Flag.LowerC; nWorkers > 1 {
		// For concurrent builds, we allow the work queue
		// to grow arbitrarily large, but only nWorkers work items
		// can be running concurrently.
		workq := make(chan func(int))
		done := make(chan int)
		go func() {
			ids := make([]int, nWorkers)
			for i := range ids {
				ids[i] = i
			}
			var pending []func(int)
			for {
				select {
				case work := <-workq:
					pending = append(pending, work)
				case id := <-done:
					ids = append(ids, id)
				}
				for len(pending) > 0 && len(ids) > 0 {
					work := pending[len(pending)-1]
					id := ids[len(ids)-1]
					pending = pending[:len(pending)-1]
					ids = ids[:len(ids)-1]
					go func() {
						work(id)
						done <- id
					}()
				}
			}
		}()
		queue = func(work func(int)) {
			workq <- work
		}
	}

	var wg sync.WaitGroup
	var compile func([]*ir.Func)
	compile = func(fns []*ir.Func) {
		wg.Add(len(fns))
		for _, fn := range fns {
			fn := fn
			queue(func(worker int) {
				ssagen.Compile(fn, worker, profile)
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
