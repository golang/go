// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// Utilities for dealing with core types.

// coreType returns the core type of T or nil if T does not have a core type.
//
// See https://go.dev/ref/spec#Core_types for the definition of a core type.
func coreType(T types.Type) types.Type {
	U := T.Underlying()
	if _, ok := U.(*types.Interface); !ok {
		return U // for non-interface types,
	}

	terms, err := _NormalTerms(U)
	if len(terms) == 0 || err != nil {
		// len(terms) -> empty type set of interface.
		// err != nil => U is invalid, exceeds complexity bounds, or has an empty type set.
		return nil // no core type.
	}

	U = terms[0].Type().Underlying()
	var identical int // i in [0,identical) => Identical(U, terms[i].Type().Underlying())
	for identical = 1; identical < len(terms); identical++ {
		if !types.Identical(U, terms[identical].Type().Underlying()) {
			break
		}
	}

	if identical == len(terms) {
		// https://go.dev/ref/spec#Core_types
		// "There is a single type U which is the underlying type of all types in the type set of T"
		return U
	}
	ch, ok := U.(*types.Chan)
	if !ok {
		return nil // no core type as identical < len(terms) and U is not a channel.
	}
	// https://go.dev/ref/spec#Core_types
	// "the type chan E if T contains only bidirectional channels, or the type chan<- E or
	// <-chan E depending on the direction of the directional channels present."
	for chans := identical; chans < len(terms); chans++ {
		curr, ok := terms[chans].Type().Underlying().(*types.Chan)
		if !ok {
			return nil
		}
		if !types.Identical(ch.Elem(), curr.Elem()) {
			return nil // channel elements are not identical.
		}
		if ch.Dir() == types.SendRecv {
			// ch is bidirectional. We can safely always use curr's direction.
			ch = curr
		} else if curr.Dir() != types.SendRecv && ch.Dir() != curr.Dir() {
			// ch and curr are not bidirectional and not the same direction.
			return nil
		}
	}
	return ch
}

// isBytestring returns true if T has the same terms as interface{[]byte | string}.
// These act like a coreType for some operations: slice expressions, append and copy.
//
// See https://go.dev/ref/spec#Core_types for the details on bytestring.
func isBytestring(T types.Type) bool {
	U := T.Underlying()
	if _, ok := U.(*types.Interface); !ok {
		return false
	}

	tset := typeSetOf(U)
	if len(tset) != 2 {
		return false
	}
	hasBytes, hasString := false, false
	tset.underIs(func(t types.Type) bool {
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

// _NormalTerms returns a slice of terms representing the normalized structural
// type restrictions of a type, if any.
//
// For all types other than *types.TypeParam, *types.Interface, and
// *types.Union, this is just a single term with Tilde() == false and
// Type() == typ. For *types.TypeParam, *types.Interface, and *types.Union, see
// below.
//
// Structural type restrictions of a type parameter are created via
// non-interface types embedded in its constraint interface (directly, or via a
// chain of interface embeddings). For example, in the declaration type
// T[P interface{~int; m()}] int the structural restriction of the type
// parameter P is ~int.
//
// With interface embedding and unions, the specification of structural type
// restrictions may be arbitrarily complex. For example, consider the
// following:
//
//	type A interface{ ~string|~[]byte }
//
//	type B interface{ int|string }
//
//	type C interface { ~string|~int }
//
//	type T[P interface{ A|B; C }] int
//
// In this example, the structural type restriction of P is ~string|int: A|B
// expands to ~string|~[]byte|int|string, which reduces to ~string|~[]byte|int,
// which when intersected with C (~string|~int) yields ~string|int.
//
// _NormalTerms computes these expansions and reductions, producing a
// "normalized" form of the embeddings. A structural restriction is normalized
// if it is a single union containing no interface terms, and is minimal in the
// sense that removing any term changes the set of types satisfying the
// constraint. It is left as a proof for the reader that, modulo sorting, there
// is exactly one such normalized form.
//
// Because the minimal representation always takes this form, _NormalTerms
// returns a slice of tilde terms corresponding to the terms of the union in
// the normalized structural restriction. An error is returned if the type is
// invalid, exceeds complexity bounds, or has an empty type set. In the latter
// case, _NormalTerms returns ErrEmptyTypeSet.
//
// _NormalTerms makes no guarantees about the order of terms, except that it
// is deterministic.
//
// This is a copy of x/exp/typeparams.NormalTerms which x/tools cannot depend on.
// TODO(taking): Remove this copy when possible.
func _NormalTerms(typ types.Type) ([]*typeparams.Term, error) {
	switch typ := typ.(type) {
	case *typeparams.TypeParam:
		return typeparams.StructuralTerms(typ)
	case *typeparams.Union:
		return typeparams.UnionTermSet(typ)
	case *types.Interface:
		return typeparams.InterfaceTermSet(typ)
	default:
		return []*typeparams.Term{typeparams.NewTerm(false, typ)}, nil
	}
}

// typeSetOf returns the type set of typ. Returns an empty typeset on an error.
func typeSetOf(typ types.Type) typeSet {
	terms, err := _NormalTerms(typ)
	if err != nil {
		return nil
	}
	return terms
}

type typeSet []*typeparams.Term // type terms of the type set

// underIs calls f with the underlying types of the specific type terms
// of s and reports whether all calls to f returned true. If there are
// no specific terms, underIs returns the result of f(nil).
func (s typeSet) underIs(f func(types.Type) bool) bool {
	if len(s) == 0 {
		return f(nil)
	}
	for _, t := range s {
		u := t.Type().Underlying()
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
		terms, err := _NormalTerms(U)
		if len(terms) == 0 || err != nil {
			return nil, ixInvalid // no underlying terms or error is empty.
		}

		elem, mode := indexType(terms[0].Type())
		for i := 1; i < len(terms) && mode != ixInvalid; i++ {
			e, m := indexType(terms[i].Type())
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
