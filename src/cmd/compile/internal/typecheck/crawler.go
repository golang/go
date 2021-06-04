// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// crawlExports crawls the type/object graph rooted at the given list
// of exported objects. Any functions that are found to be potentially
// callable by importers are marked with ExportInline so that
// iexport.go knows to re-export their inline body.
func crawlExports(exports []*ir.Name) {
	p := crawler{marked: make(map[*types.Type]bool)}
	for _, n := range exports {
		p.markObject(n)
	}
}

type crawler struct {
	marked map[*types.Type]bool // types already seen by markType
}

// markObject visits a reachable object.
func (p *crawler) markObject(n *ir.Name) {
	if n.Op() == ir.ONAME && n.Class == ir.PFUNC {
		p.markInlBody(n)
	}

	p.markType(n.Type())
}

// markType recursively visits types reachable from t to identify
// functions whose inline bodies may be needed.
func (p *crawler) markType(t *types.Type) {
	if t.IsInstantiatedGeneric() {
		// Re-instantiated types don't add anything new, so don't follow them.
		return
	}
	if p.marked[t] {
		return
	}
	p.marked[t] = true

	// If this is a named type, mark all of its associated
	// methods. Skip interface types because t.Methods contains
	// only their unexpanded method set (i.e., exclusive of
	// interface embeddings), and the switch statement below
	// handles their full method set.
	if t.Sym() != nil && t.Kind() != types.TINTER {
		for _, m := range t.Methods().Slice() {
			if types.IsExported(m.Sym.Name) {
				p.markObject(m.Nname.(*ir.Name))
			}
		}
	}

	// Recursively mark any types that can be produced given a
	// value of type t: dereferencing a pointer; indexing or
	// iterating over an array, slice, or map; receiving from a
	// channel; accessing a struct field or interface method; or
	// calling a function.
	//
	// Notably, we don't mark function parameter types, because
	// the user already needs some way to construct values of
	// those types.
	switch t.Kind() {
	case types.TPTR, types.TARRAY, types.TSLICE:
		p.markType(t.Elem())

	case types.TCHAN:
		if t.ChanDir().CanRecv() {
			p.markType(t.Elem())
		}

	case types.TMAP:
		p.markType(t.Key())
		p.markType(t.Elem())

	case types.TSTRUCT:
		for _, f := range t.FieldSlice() {
			if types.IsExported(f.Sym.Name) || f.Embedded != 0 {
				p.markType(f.Type)
			}
		}

	case types.TFUNC:
		for _, f := range t.Results().FieldSlice() {
			p.markType(f.Type)
		}

	case types.TINTER:
		// TODO(danscales) - will have to deal with the types in interface
		// elements here when implemented in types2 and represented in types1.
		for _, f := range t.AllMethods().Slice() {
			if types.IsExported(f.Sym.Name) {
				p.markType(f.Type)
			}
		}

	case types.TTYPEPARAM:
		// No other type that needs to be followed.
	}
}

// markInlBody marks n's inline body for export and recursively
// ensures all called functions are marked too.
func (p *crawler) markInlBody(n *ir.Name) {
	if n == nil {
		return
	}
	if n.Op() != ir.ONAME || n.Class != ir.PFUNC {
		base.Fatalf("markInlBody: unexpected %v, %v, %v", n, n.Op(), n.Class)
	}
	fn := n.Func
	if fn == nil {
		base.Fatalf("markInlBody: missing Func on %v", n)
	}
	if fn.Inl == nil {
		return
	}

	if fn.ExportInline() {
		return
	}
	fn.SetExportInline(true)

	ImportedBody(fn)

	var doFlood func(n ir.Node)
	doFlood = func(n ir.Node) {
		switch n.Op() {
		case ir.OMETHEXPR, ir.ODOTMETH:
			p.markInlBody(ir.MethodExprName(n))

		case ir.ONAME:
			n := n.(*ir.Name)
			switch n.Class {
			case ir.PFUNC:
				p.markInlBody(n)
				Export(n)
			case ir.PEXTERN:
				Export(n)
			}

		case ir.OCALLPART:
			// Okay, because we don't yet inline indirect
			// calls to method values.
		case ir.OCLOSURE:
			// VisitList doesn't visit closure bodies, so force a
			// recursive call to VisitList on the body of the closure.
			ir.VisitList(n.(*ir.ClosureExpr).Func.Body, doFlood)
		}
	}

	// Recursively identify all referenced functions for
	// reexport. We want to include even non-called functions,
	// because after inlining they might be callable.
	ir.VisitList(fn.Inl.Body, doFlood)
}
