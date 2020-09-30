// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
)

// A function named init is a special case.
// It is called by the initialization before main is run.
// To make it unique within a package and also uncallable,
// the name, normally "pkg.init", is altered to "pkg.init.0".
var renameinitgen int

// Dummy function for autotmps generated during typechecking.
var dummyInitFn = nod(ODCLFUNC, nil, nil)

func renameinit() *types.Sym {
	s := lookupN("init.", renameinitgen)
	renameinitgen++
	return s
}

// fninit makes an initialization record for the package.
// See runtime/proc.go:initTask for its layout.
// The 3 tasks for initialization are:
//   1) Initialize all of the packages the current package depends on.
//   2) Initialize all the variables that have initializers.
//   3) Run any init functions.
func fninit(n []*Node) {
	nf := initOrder(n)

	var deps []*obj.LSym // initTask records for packages the current package depends on
	var fns []*obj.LSym  // functions to call for package initialization

	// Find imported packages with init tasks.
	for _, s := range types.InitSyms {
		deps = append(deps, s.Linksym())
	}

	// Make a function that contains all the initialization statements.
	if len(nf) > 0 {
		lineno = nf[0].Pos // prolog/epilog gets line number of first init stmt
		initializers := lookup("init")
		fn := dclfunc(initializers, nod(OTFUNC, nil, nil))
		for _, dcl := range dummyInitFn.Func.Dcl {
			dcl.Name.Curfn = fn
		}
		fn.Func.Dcl = append(fn.Func.Dcl, dummyInitFn.Func.Dcl...)
		dummyInitFn.Func.Dcl = nil

		fn.Nbody.Set(nf)
		funcbody()

		fn = typecheck(fn, ctxStmt)
		Curfn = fn
		typecheckslice(nf, ctxStmt)
		Curfn = nil
		xtop = append(xtop, fn)
		fns = append(fns, initializers.Linksym())
	}
	if dummyInitFn.Func.Dcl != nil {
		// We only generate temps using dummyInitFn if there
		// are package-scope initialization statements, so
		// something's weird if we get here.
		Fatalf("dummyInitFn still has declarations")
	}
	dummyInitFn = nil

	// Record user init functions.
	for i := 0; i < renameinitgen; i++ {
		s := lookupN("init.", i)
		fn := asNode(s.Def).Name.Defn
		// Skip init functions with empty bodies.
		if fn.Nbody.Len() == 1 && fn.Nbody.First().Op == OEMPTY {
			continue
		}
		fns = append(fns, s.Linksym())
	}

	if len(deps) == 0 && len(fns) == 0 && localpkg.Name != "main" && localpkg.Name != "runtime" {
		return // nothing to initialize
	}

	// Make an .inittask structure.
	sym := lookup(".inittask")
	nn := newname(sym)
	nn.Type = types.Types[TUINT8] // dummy type
	nn.SetClass(PEXTERN)
	sym.Def = asTypesNode(nn)
	exportsym(nn)
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
}
