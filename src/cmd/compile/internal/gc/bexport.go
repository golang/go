// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

type exporter struct {
	marked map[*types.Type]bool // types already seen by markType
}

// markObject visits a reachable object.
func (p *exporter) markObject(n ir.Node) {
	if n.Op() == ir.ONAME && n.Class() == ir.PFUNC {
		inlFlood(n.(*ir.Name))
	}

	p.markType(n.Type())
}

// markType recursively visits types reachable from t to identify
// functions whose inline bodies may be needed.
func (p *exporter) markType(t *types.Type) {
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
				p.markObject(ir.AsNode(m.Nname))
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
		for _, f := range t.FieldSlice() {
			if types.IsExported(f.Sym.Name) {
				p.markType(f.Type)
			}
		}
	}
}

// ----------------------------------------------------------------------------
// Export format

// Tags. Must be < 0.
const (
	// Objects
	packageTag = -(iota + 1)
	constTag
	typeTag
	varTag
	funcTag
	endTag

	// Types
	namedTag
	arrayTag
	sliceTag
	dddTag
	structTag
	pointerTag
	signatureTag
	interfaceTag
	mapTag
	chanTag

	// Values
	falseTag
	trueTag
	int64Tag
	floatTag
	fractionTag // not used by gc
	complexTag
	stringTag
	nilTag
	unknownTag // not used by gc (only appears in packages with errors)

	// Type aliases
	aliasTag
)

var predecl []*types.Type // initialized lazily

func predeclared() []*types.Type {
	if predecl == nil {
		// initialize lazily to be sure that all
		// elements have been initialized before
		predecl = []*types.Type{
			// basic types
			types.Types[types.TBOOL],
			types.Types[types.TINT],
			types.Types[types.TINT8],
			types.Types[types.TINT16],
			types.Types[types.TINT32],
			types.Types[types.TINT64],
			types.Types[types.TUINT],
			types.Types[types.TUINT8],
			types.Types[types.TUINT16],
			types.Types[types.TUINT32],
			types.Types[types.TUINT64],
			types.Types[types.TUINTPTR],
			types.Types[types.TFLOAT32],
			types.Types[types.TFLOAT64],
			types.Types[types.TCOMPLEX64],
			types.Types[types.TCOMPLEX128],
			types.Types[types.TSTRING],

			// basic type aliases
			types.ByteType,
			types.RuneType,

			// error
			types.ErrorType,

			// untyped types
			types.UntypedBool,
			types.UntypedInt,
			types.UntypedRune,
			types.UntypedFloat,
			types.UntypedComplex,
			types.UntypedString,
			types.Types[types.TNIL],

			// package unsafe
			types.Types[types.TUNSAFEPTR],

			// invalid type (package contains errors)
			types.Types[types.Txxx],

			// any type, for builtin export data
			types.Types[types.TANY],
		}
	}
	return predecl
}
