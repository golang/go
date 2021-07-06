// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkginit

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/deadcode"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
)

// Task makes and returns an initialization record for the package.
// See runtime/proc.go:initTask for its layout.
// The 3 tasks for initialization are:
//   1) Initialize all of the packages the current package depends on.
//   2) Initialize all the variables that have initializers.
//   3) Run any init functions.
func Task() *ir.Name {
	nf := initOrder(typecheck.Target.Decls)

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

	// Make a function that contains all the initialization statements.
	if len(nf) > 0 {
		base.Pos = nf[0].Pos() // prolog/epilog gets line number of first init stmt
		initializers := typecheck.Lookup("init")
		fn := typecheck.DeclFunc(initializers, ir.NewFuncType(base.Pos, nil, nil, nil))
		for _, dcl := range typecheck.InitTodoFunc.Dcl {
			dcl.Curfn = fn
		}
		fn.Dcl = append(fn.Dcl, typecheck.InitTodoFunc.Dcl...)
		typecheck.InitTodoFunc.Dcl = nil

		fn.Body = nf
		typecheck.FinishFuncBody()

		typecheck.Func(fn)
		ir.CurFunc = fn
		typecheck.Stmts(nf)
		ir.CurFunc = nil
		typecheck.Target.Decls = append(typecheck.Target.Decls, fn)
		fns = append(fns, fn.Linksym())
	}
	if typecheck.InitTodoFunc.Dcl != nil {
		// We only generate temps using InitTodoFunc if there
		// are package-scope initialization statements, so
		// something's weird if we get here.
		base.Fatalf("InitTodoFunc still has declarations")
	}
	typecheck.InitTodoFunc = nil

	// Record user init functions.
	for _, fn := range typecheck.Target.Inits {
		// Must happen after initOrder; see #43444.
		deadcode.Func(fn)

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
