// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
)

// A function named init is a special case.
// It is called by the initialization before main is run.
// To make it unique within a package and also uncallable,
// the name, normally "pkg.init", is altered to "pkg.init.0".
var renameinitgen int

// Function collecting autotmps generated during typechecking,
// to be included in the package-level init function.
var initTodo = ir.NewFunc(base.Pos)

func renameinit() *types.Sym {
	s := lookupN("init.", renameinitgen)
	renameinitgen++
	return s
}

// fninit makes and returns an initialization record for the package.
// See runtime/proc.go:initTask for its layout.
// The 3 tasks for initialization are:
//   1) Initialize all of the packages the current package depends on.
//   2) Initialize all the variables that have initializers.
//   3) Run any init functions.
func fninit() *ir.Name {
	nf := initOrder(Target.Decls)

	var deps []*obj.LSym // initTask records for packages the current package depends on
	var fns []*obj.LSym  // functions to call for package initialization

	// Find imported packages with init tasks.
	for _, pkg := range Target.Imports {
		n := resolve(ir.NewIdent(base.Pos, pkg.Lookup(".inittask")))
		if n.Op() == ir.ONONAME {
			continue
		}
		if n.Op() != ir.ONAME || n.(*ir.Name).Class_ != ir.PEXTERN {
			base.Fatalf("bad inittask: %v", n)
		}
		deps = append(deps, n.(*ir.Name).Sym().Linksym())
	}

	// Make a function that contains all the initialization statements.
	if len(nf) > 0 {
		base.Pos = nf[0].Pos() // prolog/epilog gets line number of first init stmt
		initializers := lookup("init")
		fn := dclfunc(initializers, ir.NewFuncType(base.Pos, nil, nil, nil))
		for _, dcl := range initTodo.Dcl {
			dcl.Curfn = fn
		}
		fn.Dcl = append(fn.Dcl, initTodo.Dcl...)
		initTodo.Dcl = nil

		fn.Body.Set(nf)
		funcbody()

		typecheckFunc(fn)
		Curfn = fn
		typecheckslice(nf, ctxStmt)
		Curfn = nil
		Target.Decls = append(Target.Decls, fn)
		fns = append(fns, initializers.Linksym())
	}
	if initTodo.Dcl != nil {
		// We only generate temps using initTodo if there
		// are package-scope initialization statements, so
		// something's weird if we get here.
		base.Fatalf("initTodo still has declarations")
	}
	initTodo = nil

	// Record user init functions.
	for _, fn := range Target.Inits {
		// Skip init functions with empty bodies.
		if fn.Body.Len() == 1 {
			if stmt := fn.Body.First(); stmt.Op() == ir.OBLOCK && stmt.(*ir.BlockStmt).List.Len() == 0 {
				continue
			}
		}
		fns = append(fns, fn.Nname.Sym().Linksym())
	}

	if len(deps) == 0 && len(fns) == 0 && types.LocalPkg.Name != "main" && types.LocalPkg.Name != "runtime" {
		return nil // nothing to initialize
	}

	// Make an .inittask structure.
	sym := lookup(".inittask")
	task := NewName(sym)
	task.SetType(types.Types[types.TUINT8]) // fake type
	task.Class_ = ir.PEXTERN
	sym.Def = task
	lsym := sym.Linksym()
	ot := 0
	ot = duintptr(lsym, ot, 0) // state: not initialized yet
	ot = duintptr(lsym, ot, uint64(len(deps)))
	ot = duintptr(lsym, ot, uint64(len(fns)))
	for _, d := range deps {
		ot = dsymptr(lsym, ot, d, 0)
	}
	for _, f := range fns {
		ot = dsymptr(lsym, ot, f, 0)
	}
	// An initTask has pointers, but none into the Go heap.
	// It's not quite read only, the state field must be modifiable.
	ggloblsym(lsym, int32(ot), obj.NOPTR)
	return task
}
