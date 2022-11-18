// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// Utilities for dealing with core types.

// isBytestring returns true if T has the same terms as interface{[]byte | string}.
// These act like a core type for some operations: slice expressions, append and copy.
//
// See https://go.dev/ref/spec#Core_types for the details on bytestring.
func isBytestring(T types.Type) bool {
	U := T.Underlying()
	if _, ok := U.(*types.Interface); !ok {
		return false
	}

	tset := typeSetOf(U)
	if tset.Len() != 2 {
		return false
	}
	hasBytes, hasString := false, false
	underIs(tset, func(t types.Type) bool {
		switch {
		case isString(t):
			hasString = true
		case isByteSlice(t):
			hasBytes = true
		}
		return hasBytes || hasString
	})
	return hasBytes && hasString
}

// termList is a list of types.
type termList []*typeparams.Term       // type terms of the type set
func (s termList) Len() int            { return len(s) }
func (s termList) At(i int) types.Type { return s[i].Type() }

// typeSetOf returns the type set of typ. Returns an empty typeset on an error.
func typeSetOf(typ types.Type) termList {
	// This is a adaptation of x/exp/typeparams.NormalTerms which x/tools cannot depend on.
	var terms []*typeparams.Term
	var err error
	switch typ := typ.(type) {
	case *typeparams.TypeParam:
		terms, err = typeparams.StructuralTerms(typ)
	case *typeparams.Union:
		terms, err = typeparams.UnionTermSet(typ)
	case *types.Interface:
		terms, err = typeparams.InterfaceTermSet(typ)
	default:
		// Common case.
		// Specializing the len=1 case to avoid a slice
		// had no measurable space/time benefit.
		terms = []*typeparams.Term{typeparams.NewTerm(false, typ)}
	}

	if err != nil {
		return termList(nil)
	}
	return termList(terms)
}

// underIs calls f with the underlying types of the specific type terms
// of s and reports whether all calls to f returned true. If there are
// no specific terms, underIs returns the result of f(nil).
func underIs(s termList, f func(types.Type) bool) bool {
	if s.Len() == 0 {
		return f(nil)
	}
	for i := 0; i < s.Len(); i++ {
		u := s.At(i).Underlying()
		if !f(u) {
			return false
		}
	}
	return true
}

// indexType returns the element type and index mode of a IndexExpr over a type.
// It returns (nil, invalid) if the type is not indexable; this should never occur in a well-typed program.
func indexType(typ types.Type) (types.Type, indexMode) {
	switch U := typ.Underlying().(type) {
	case *types.Array:
		return U.Elem(), ixArrVar
	case *types.Pointer:
		if arr, ok := U.Elem().Underlying().(*types.Array); ok {
			return arr.Elem(), ixVar
		}
	case *types.Slice:
		return U.Elem(), ixVar
	case *types.Map:
		return U.Elem(), ixMap
	case *types.Basic:
		return tByte, ixValue // must be a string
	case *types.Interface:
		tset := typeSetOf(U)
		if tset.Len() == 0 {
			return nil, ixInvalid // no underlying terms or error is empty.
		}

		elem, mode := indexType(tset.At(0))
		for i := 1; i < tset.Len() && mode != ixInvalid; i++ {
			e, m := indexType(tset.At(i))
			if !types.Identical(elem, e) { // if type checked, just a sanity check
				return nil, ixInvalid
			}
			// Update the mode to the most constrained address type.
			mode = mode.meet(m)
		}
		if mode != ixInvalid {
			return elem, mode
		}
	}
	return nil, ixInvalid
}

// An indexMode specifies the (addressing) mode of an index operand.
//
// Addressing mode of an index operation is based on the set of
// underlying types.
// Hasse diagram of the indexMode meet semi-lattice:
//
//	ixVar     ixMap
//	  |          |
//	ixArrVar     |
//	  |          |
//	ixValue      |
//	   \        /
//	  ixInvalid
type indexMode byte

const (
	ixInvalid indexMode = iota // index is invalid
	ixValue                    // index is a computed value (not addressable)
	ixArrVar                   // like ixVar, but index operand contains an array
	ixVar                      // index is an addressable variable
	ixMap                      // index is a map index expression (acts like a variable on lhs, commaok on rhs of an assignment)
)

// meet is the address type that is constrained by both x and y.
func (x indexMode) meet(y indexMode) indexMode {
	if (x == ixMap || y == ixMap) && x != y {
		return ixInvalid
	}
	// Use int representation and return min.
	if x < y {
		return y
	}
	return x
}
