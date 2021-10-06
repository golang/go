// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkginit

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/staticinit"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
)

// MakeInit creates a synthetic init function to handle any
// package-scope initialization statements.
//
// TODO(mdempsky): Move into noder, so that the types2-based frontends
// can use Info.InitOrder instead.
func MakeInit() {
	nf := initOrder(typecheck.Target.Decls)
	if len(nf) == 0 {
		return
	}

	// Make a function that contains all the initialization statements.
	base.Pos = nf[0].Pos() // prolog/epilog gets line number of first init stmt
	initializers := typecheck.Lookup("init")
	fn := typecheck.DeclFunc(initializers, ir.NewFuncType(base.Pos, nil, nil, nil))
	for _, dcl := range typecheck.InitTodoFunc.Dcl {
		dcl.Curfn = fn
	}
	fn.Dcl = append(fn.Dcl, typecheck.InitTodoFunc.Dcl...)
	typecheck.InitTodoFunc.Dcl = nil

	// Suppress useless "can inline" diagnostics.
	// Init functions are only called dynamically.
	fn.SetInlinabilityChecked(true)

	fn.Body = nf
	typecheck.FinishFuncBody()

	typecheck.Func(fn)
	ir.WithFunc(fn, func() {
		typecheck.Stmts(nf)
	})
	typecheck.Target.Decls = append(typecheck.Target.Decls, fn)

	// Prepend to Inits, so it runs first, before any user-declared init
	// functions.
	typecheck.Target.Inits = append([]*ir.Func{fn}, typecheck.Target.Inits...)

	if typecheck.InitTodoFunc.Dcl != nil {
		// We only generate temps using InitTodoFunc if there
		// are package-scope initialization statements, so
		// something's weird if we get here.
		base.Fatalf("InitTodoFunc still has declarations")
	}
	typecheck.InitTodoFunc = nil
}

// Task makes and returns an initialization record for the package.
// See runtime/proc.go:initTask for its layout.
// The 3 tasks for initialization are:
//   1) Initialize all of the packages the current package depends on.
//   2) Initialize all the variables that have initializers.
//   3) Run any init functions.
func Task() *ir.Name {
	var deps []*obj.LSym // initTask records for packages the current package depends on
	var fns []*obj.LSym  // functions to call for package initialization

	// Find imported packages with init tasks.
	for _, pkg := range typecheck.Target.Imports {
		n := typecheck.Resolve(ir.NewIdent(base.Pos, pkg.Lookup(".inittask")))
		if n.Op() == ir.ONONAME {
			continue
		}
		if n.Op() != ir.ONAME || n.(*ir.Name).Class != ir.PEXTERN {
			base.Fatalf("bad inittask: %v", n)
		}
		deps = append(deps, n.(*ir.Name).Linksym())
	}

	// Record user init functions.
	for _, fn := range typecheck.Target.Inits {
		if fn.Sym().Name == "init" {
			// Synthetic init function for initialization of package-scope
			// variables. We can use staticinit to optimize away static
			// assignments.
			s := staticinit.Schedule{
				Plans: make(map[ir.Node]*staticinit.Plan),
				Temps: make(map[ir.Node]*ir.Name),
			}
			for _, n := range fn.Body {
				s.StaticInit(n)
			}
			fn.Body = s.Out
			ir.WithFunc(fn, func() {
				typecheck.Stmts(fn.Body)
			})

			if len(fn.Body) == 0 {
				fn.Body = []ir.Node{ir.NewBlockStmt(src.NoXPos, nil)}
			}
		}

		// Skip init functions with empty bodies.
		if len(fn.Body) == 1 {
			if stmt := fn.Body[0]; stmt.Op() == ir.OBLOCK && len(stmt.(*ir.BlockStmt).List) == 0 {
				continue
			}
		}
		fns = append(fns, fn.Nname.Linksym())
	}

	if len(deps) == 0 && len(fns) == 0 && types.LocalPkg.Name != "main" && types.LocalPkg.Name != "runtime" {
		return nil // nothing to initialize
	}

	// Make an .inittask structure.
	sym := typecheck.Lookup(".inittask")
	task := typecheck.NewName(sym)
	task.SetType(types.Types[types.TUINT8]) // fake type
	task.Class = ir.PEXTERN
	sym.Def = task
	lsym := task.Linksym()
	ot := 0
	ot = objw.Uintptr(lsym, ot, 0) // state: not initialized yet
	ot = objw.Uintptr(lsym, ot, uint64(len(deps)))
	ot = objw.Uintptr(lsym, ot, uint64(len(fns)))
	for _, d := range deps {
		ot = objw.SymPtr(lsym, ot, d, 0)
	}
	for _, f := range fns {
		ot = objw.SymPtr(lsym, ot, f, 0)
	}
	// An initTask has pointers, but none into the Go heap.
	// It's not quite read only, the state field must be modifiable.
	objw.Global(lsym, int32(ot), obj.NOPTR)
	return task
}
