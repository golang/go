// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// crawlExports crawls the type/object graph rooted at the given list of exported
// objects (which are variables, functions, and types). It descends through all parts
// of types and follows methods on defined types. Any functions that are found to be
// potentially callable by importers directly or after inlining are marked with
// ExportInline, so that iexport.go knows to export their inline body.
//
// The overall purpose of crawlExports is to AVOID exporting inlineable methods
// that cannot actually be referenced, thereby reducing the size of the exports
// significantly.
//
// For non-generic defined types reachable from global variables, we only set
// ExportInline for exported methods. For defined types that are directly named or are
// embedded recursively in such a type, we set ExportInline for all methods, since
// these types can be embedded in another local type. For instantiated types that are
// used anywhere in a inlineable function, we set ExportInline on all methods of the
// base generic type, since all methods will be needed for creating any instantiated
// type.
func crawlExports(exports []*ir.Name) {
	p := crawler{
		marked:         make(map[*types.Type]bool),
		embedded:       make(map[*types.Type]bool),
		generic:        make(map[*types.Type]bool),
		checkFullyInst: make(map[*types.Type]bool),
	}
	for _, n := range exports {
		p.markObject(n)
	}
}

type crawler struct {
	marked         map[*types.Type]bool // types already seen by markType
	embedded       map[*types.Type]bool // types already seen by markEmbed
	generic        map[*types.Type]bool // types already seen by markGeneric
	checkFullyInst map[*types.Type]bool // types already seen by checkForFullyInst
}

// markObject visits a reachable object (function, method, global type, or global variable)
func (p *crawler) markObject(n *ir.Name) {
	if n.Op() == ir.ONAME && n.Class == ir.PFUNC {
		p.markInlBody(n)
	}

	// If a declared type name is reachable, users can embed it in their
	// own types, which makes even its unexported methods reachable.
	if n.Op() == ir.OTYPE {
		p.markEmbed(n.Type())
	}

	p.markType(n.Type())
}

// markType recursively visits types reachable from t to identify functions whose
// inline bodies may be needed. For instantiated generic types, it visits the base
// generic type, which has the relevant methods.
func (p *crawler) markType(t *types.Type) {
	if orig := t.OrigType(); orig != nil {
		// Convert to the base generic type.
		t = orig
	}
	if p.marked[t] {
		return
	}
	p.marked[t] = true

	// If this is a defined type, mark all of its associated
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
		if t.IsFuncArgStruct() {
			break
		}
		for _, f := range t.FieldSlice() {
			// Mark the type of a unexported field if it is a
			// fully-instantiated type, since we create and instantiate
			// the methods of any fully-instantiated type that we see
			// during import (see end of typecheck.substInstType).
			if types.IsExported(f.Sym.Name) || f.Embedded != 0 ||
				isPtrFullyInstantiated(f.Type) {
				p.markType(f.Type)
			}
		}

	case types.TFUNC:
		for _, f := range t.Results().FieldSlice() {
			p.markType(f.Type)
		}

	case types.TINTER:
		for _, f := range t.AllMethods().Slice() {
			if types.IsExported(f.Sym.Name) {
				p.markType(f.Type)
			}
		}

	case types.TTYPEPARAM:
		// No other type that needs to be followed.
	}
}

// markEmbed is similar to markType, but handles finding methods that
// need to be re-exported because t can be embedded in user code
// (possibly transitively).
func (p *crawler) markEmbed(t *types.Type) {
	if t.IsPtr() {
		// Defined pointer type; not allowed to embed anyway.
		if t.Sym() != nil {
			return
		}
		t = t.Elem()
	}

	if orig := t.OrigType(); orig != nil {
		// Convert to the base generic type.
		t = orig
	}

	if p.embedded[t] {
		return
	}
	p.embedded[t] = true

	// If t is a defined type, then re-export all of its methods. Unlike
	// in markType, we include even unexported methods here, because we
	// still need to generate wrappers for them, even if the user can't
	// refer to them directly.
	if t.Sym() != nil && t.Kind() != types.TINTER {
		for _, m := range t.Methods().Slice() {
			p.markObject(m.Nname.(*ir.Name))
		}
	}

	// If t is a struct, recursively visit its embedded fields.
	if t.IsStruct() {
		for _, f := range t.FieldSlice() {
			if f.Embedded != 0 {
				p.markEmbed(f.Type)
			}
		}
	}
}

// markGeneric takes an instantiated type or a base generic type t, and marks all the
// methods of the base generic type of t. If a base generic type is written out for
// export, even if not explicitly marked for export, then all of its methods need to
// be available for instantiation, since we always create all methods of a specified
// instantiated type. Non-exported methods must generally be instantiated, since they may
// be called by the exported methods or other generic function in the same package.
func (p *crawler) markGeneric(t *types.Type) {
	if t.IsPtr() {
		t = t.Elem()
	}
	if orig := t.OrigType(); orig != nil {
		// Convert to the base generic type.
		t = orig
	}
	if p.generic[t] {
		return
	}
	p.generic[t] = true

	if t.Sym() != nil && t.Kind() != types.TINTER {
		for _, m := range t.Methods().Slice() {
			p.markObject(m.Nname.(*ir.Name))
		}
	}
}

// checkForFullyInst looks for fully-instantiated types in a type (at any nesting
// level). If it finds a fully-instantiated type, it ensures that the necessary
// dictionary and shape methods are exported. It updates p.checkFullyInst, so it
// traverses each particular type only once.
func (p *crawler) checkForFullyInst(t *types.Type) {
	if p.checkFullyInst[t] {
		return
	}
	p.checkFullyInst[t] = true

	if t.IsFullyInstantiated() && !t.HasShape() && !t.IsInterface() && t.Methods().Len() > 0 {
		// For any fully-instantiated type, the relevant
		// dictionaries and shape instantiations will have
		// already been created or are in the import data.
		// Make sure that they are exported, so that any
		// other package that inlines this function will have
		// them available for import, and so will not need
		// another round of method and dictionary
		// instantiation after inlining.
		baseType := t.OrigType()
		shapes := make([]*types.Type, len(t.RParams()))
		for i, t1 := range t.RParams() {
			shapes[i] = Shapify(t1, i, baseType.RParams()[i])
		}
		for j := range t.Methods().Slice() {
			baseNname := baseType.Methods().Slice()[j].Nname.(*ir.Name)
			dictsym := MakeDictSym(baseNname.Sym(), t.RParams(), true)
			if dictsym.Def == nil {
				in := Resolve(ir.NewIdent(src.NoXPos, dictsym))
				dictsym = in.Sym()
			}
			Export(dictsym.Def.(*ir.Name))
			methsym := MakeFuncInstSym(baseNname.Sym(), shapes, false, true)
			if methsym.Def == nil {
				in := Resolve(ir.NewIdent(src.NoXPos, methsym))
				methsym = in.Sym()
			}
			methNode := methsym.Def.(*ir.Name)
			Export(methNode)
			if HaveInlineBody(methNode.Func) {
				// Export the body as well if
				// instantiation is inlineable.
				ImportedBody(methNode.Func)
				methNode.Func.SetExportInline(true)
			}
		}
	}

	// Descend into the type. We descend even if it is a fully-instantiated type,
	// since the instantiated type may have other instantiated types inside of
	// it (in fields, methods, etc.).
	switch t.Kind() {
	case types.TPTR, types.TARRAY, types.TSLICE:
		p.checkForFullyInst(t.Elem())

	case types.TCHAN:
		p.checkForFullyInst(t.Elem())

	case types.TMAP:
		p.checkForFullyInst(t.Key())
		p.checkForFullyInst(t.Elem())

	case types.TSTRUCT:
		if t.IsFuncArgStruct() {
			break
		}
		for _, f := range t.FieldSlice() {
			p.checkForFullyInst(f.Type)
		}

	case types.TFUNC:
		if recv := t.Recv(); recv != nil {
			p.checkForFullyInst(t.Recv().Type)
		}
		for _, f := range t.Params().FieldSlice() {
			p.checkForFullyInst(f.Type)
		}
		for _, f := range t.Results().FieldSlice() {
			p.checkForFullyInst(f.Type)
		}

	case types.TINTER:
		for _, f := range t.AllMethods().Slice() {
			p.checkForFullyInst(f.Type)
		}
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
	if !HaveInlineBody(fn) {
		return
	}

	if fn.ExportInline() {
		return
	}
	fn.SetExportInline(true)

	ImportedBody(fn)

	var doFlood func(n ir.Node)
	doFlood = func(n ir.Node) {
		t := n.Type()
		if t != nil {
			if t.HasTParam() {
				// If any generic types are used, then make sure that
				// the methods of the generic type are exported and
				// scanned for other possible exports.
				p.markGeneric(t)
			} else {
				p.checkForFullyInst(t)
			}
			if base.Debug.Unified == 0 {
				// If a method of un-exported type is promoted and accessible by
				// embedding in an exported type, it makes that type reachable.
				//
				// Example:
				//
				//     type t struct {}
				//     func (t) M() {}
				//
				//     func F() interface{} { return struct{ t }{} }
				//
				// We generate the wrapper for "struct{ t }".M, and inline call
				// to "struct{ t }".M, which makes "t.M" reachable.
				if t.IsStruct() {
					for _, f := range t.FieldSlice() {
						if f.Embedded != 0 {
							p.markEmbed(f.Type)
						}
					}
				}
			}
		}

		switch n.Op() {
		case ir.OMETHEXPR, ir.ODOTMETH:
			p.markInlBody(ir.MethodExprName(n))
		case ir.ONAME:
			n := n.(*ir.Name)
			switch n.Class {
			case ir.PFUNC:
				p.markInlBody(n)
				// Note: this Export() and the one below seem unneeded,
				// since any function/extern name encountered in an
				// exported function body will be exported
				// automatically via qualifiedIdent() in iexport.go.
				Export(n)
			case ir.PEXTERN:
				Export(n)
			}
		case ir.OMETHVALUE:
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

// isPtrFullyInstantiated returns true if t is a fully-instantiated type, or it is a
// pointer to a fully-instantiated type.
func isPtrFullyInstantiated(t *types.Type) bool {
	return t.IsPtr() && t.Elem().IsFullyInstantiated() ||
		t.IsFullyInstantiated()
}
