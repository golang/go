// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
)

type exporter struct {
	marked map[*types.Type]bool // types already seen by markType
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
	if t.Sym != nil && t.Etype != TINTER {
		for _, m := range t.Methods().Slice() {
			if types.IsExported(m.Sym.Name) {
				p.markType(m.Type)
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
	switch t.Etype {
	case TPTR, TARRAY, TSLICE:
		p.markType(t.Elem())

	case TCHAN:
		if t.ChanDir().CanRecv() {
			p.markType(t.Elem())
		}

	case TMAP:
		p.markType(t.Key())
		p.markType(t.Elem())

	case TSTRUCT:
		for _, f := range t.FieldSlice() {
			if types.IsExported(f.Sym.Name) || f.Embedded != 0 {
				p.markType(f.Type)
			}
		}

	case TFUNC:
		// If t is the type of a function or method, then
		// t.Nname() is its ONAME. Mark its inline body and
		// any recursively called functions for export.
		inlFlood(asNode(t.Nname()))

		for _, f := range t.Results().FieldSlice() {
			p.markType(f.Type)
		}

	case TINTER:
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
			types.Types[TBOOL],
			types.Types[TINT],
			types.Types[TINT8],
			types.Types[TINT16],
			types.Types[TINT32],
			types.Types[TINT64],
			types.Types[TUINT],
			types.Types[TUINT8],
			types.Types[TUINT16],
			types.Types[TUINT32],
			types.Types[TUINT64],
			types.Types[TUINTPTR],
			types.Types[TFLOAT32],
			types.Types[TFLOAT64],
			types.Types[TCOMPLEX64],
			types.Types[TCOMPLEX128],
			types.Types[TSTRING],

			// basic type aliases
			types.Bytetype,
			types.Runetype,

			// error
			types.Errortype,

			// untyped types
			types.UntypedBool,
			types.UntypedInt,
			types.UntypedRune,
			types.UntypedFloat,
			types.UntypedComplex,
			types.UntypedString,
			types.Types[TNIL],

			// package unsafe
			types.Types[TUNSAFEPTR],

			// invalid type (package contains errors)
			types.Types[Txxx],

			// any type, for builtin export data
			types.Types[TANY],
		}
	}
	return predecl
}
