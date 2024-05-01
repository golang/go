// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"fmt"
	"go/types"

	"golang.org/x/tools/internal/aliases"
)

// CoreType returns the core type of T or nil if T does not have a core type.
//
// See https://go.dev/ref/spec#Core_types for the definition of a core type.
func CoreType(T types.Type) types.Type {
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
func _NormalTerms(typ types.Type) ([]*types.Term, error) {
	switch typ := aliases.Unalias(typ).(type) {
	case *types.TypeParam:
		return StructuralTerms(typ)
	case *types.Union:
		return UnionTermSet(typ)
	case *types.Interface:
		return InterfaceTermSet(typ)
	default:
		return []*types.Term{types.NewTerm(false, typ)}, nil
	}
}

// Deref returns the type of the variable pointed to by t,
// if t's core type is a pointer; otherwise it returns t.
//
// Do not assume that Deref(T)==T implies T is not a pointer:
// consider "type T *T", for example.
//
// TODO(adonovan): ideally this would live in typesinternal, but that
// creates an import cycle. Move there when we melt this package down.
func Deref(t types.Type) types.Type {
	if ptr, ok := CoreType(t).(*types.Pointer); ok {
		return ptr.Elem()
	}
	return t
}

// MustDeref returns the type of the variable pointed to by t.
// It panics if t's core type is not a pointer.
//
// TODO(adonovan): ideally this would live in typesinternal, but that
// creates an import cycle. Move there when we melt this package down.
func MustDeref(t types.Type) types.Type {
	if ptr, ok := CoreType(t).(*types.Pointer); ok {
		return ptr.Elem()
	}
	panic(fmt.Sprintf("%v is not a pointer", t))
}
