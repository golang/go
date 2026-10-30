// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"go/types"
	"iter"
	"unsafe"
)

// A TypeSet is the type set of a type, represented as a normalized
// list of specific terms (types T or ~T). It is immutable.
//
// If [TypeSet.Unrestricted], the set is constrained only by methods,
// if at all (e.g. any, comparable, interface{ M() }), and has no terms.
// This is the zero value.
//
// Otherwise, the set is exactly the union of its [TypeSet.Len] disjoint
// terms. Zero terms denotes the empty set (e.g. interface{ int; string }).
//
// Terms are reported in a deterministic but otherwise unspecified order.
//
// TODO(adonovan): promote to go/types (go.dev/issue/61013).
type TypeSet struct {
	terms termlist // normalized specific terms; nil => unrestricted
}

// TypeSetOf returns the type set of t:
//
//   - For a type parameter, it is the type set of its constraint.
//   - For an interface or union type (or a named type whose underlying
//     type is an interface), it is the set implied by its embedded
//     non-interface types and unions, ignoring methods.
//   - For any other type t, it is {t}: a single non-tilde term holding t
//     as provided (including any alias).
//
// The specific terms are normalized: the type set is expressed as a single
// union containing no interface terms, minimal in the sense that removing
// any term changes the set. For example, given
//
//	type A interface{ ~string|~[]byte }
//	type B interface{ int|string }
//	type C interface{ ~string|~int }
//	type T[P interface{ A|B; C }] int
//
// the type set of P is ~string|int: A|B expands to
// ~string|~[]byte|int|string, which reduces to ~string|~[]byte|int,
// which when intersected with C (~string|~int) yields ~string|int.
//
// TypeSetOf never fails. If t is ill-formed (e.g. it has an invalid
// embedded type or cyclic declaration) or exceeds complexity bounds,
// the result is Unrestricted. This cannot happen in well-typed code,
// and for ill-typed code it is the conservative answer: "does every
// type in the set satisfy P?" yields false (whereas the empty set
// would vacuously yield true, licensing unsound transformations), and
// "does some type in the set satisfy P?" finds no witness (so no
// spurious diagnostic is reported).
//
// Performance notes (non-normative): the singleton case {t} allocates
// a [types.Term]; callers may avoid this by first checking
// !types.IsInterface(t), which holds exactly when the type set of t
// is {t}. (IsInterface reports true for type parameters.) Interfaces
// and type parameters allocate only if they have type terms.
func TypeSetOf(t types.Type) TypeSet {
	// typeSetOf(T) == typeSetOf(Unalias(T))
	typ := types.Unalias(t)
	if named, ok := typ.(*types.Named); ok {
		typ = named.Underlying()
	}
	switch typ := typ.(type) {
	case *types.TypeParam:
		constraint := typ.Constraint()
		if constraint == nil {
			return TypeSet{} // ill-formed; treat as unrestricted
		}
		iface, ok := constraint.Underlying().(*types.Interface)
		if !ok {
			return TypeSet{} // ill-formed; treat as unrestricted
		}
		return termSetOf(iface)

	case *types.Interface, *types.Union:
		return termSetOf(typ)

	default:
		// Allocate the term and the list that holds it together.
		single := &struct {
			term term
			list [1]*term
		}{term: term{false, t}}
		single.list[0] = &single.term
		return TypeSet{terms: single.list[:]}
	}
}

// termSetOf returns the normalized type set of an interface or union.
func termSetOf(typ types.Type) TypeSet {
	// Fast path: an interface with no type terms (e.g. any, error,
	// io.Reader) is unrestricted. go/types caches this information,
	// so this check avoids the allocation of the general algorithm.
	if iface, ok := typ.(*types.Interface); ok && iface.IsMethodSet() {
		return TypeSet{}
	}

	tset, err := computeTermSetInternal(typ, make(map[types.Type]*termSet), 0)
	switch {
	case err != nil:
		return TypeSet{} // ill-formed or too complex; treat as unrestricted (see TypeSetOf)
	case tset.terms.isAll():
		return TypeSet{}
	case tset.terms.isEmpty():
		return TypeSet{terms: termlist{}} // non-nil
	default:
		return TypeSet{terms: tset.terms}
	}
}

// Unrestricted reports whether the type set is not restricted by
// specific terms, for example any, comparable, or interface{ M() }.
// It also reports true for ill-formed types; see [TypeSetOf].
func (s TypeSet) Unrestricted() bool { return s.terms == nil }

// Len returns the number of specific terms.
// Zero means either Unrestricted (all types) or empty (no types).
func (s TypeSet) Len() int { return len(s.terms) }

// Term returns the ith specific term.
// It panics unless 0 <= i < s.Len().
func (s TypeSet) Term(i int) *types.Term { return asTypesTerm(s.terms[i]) }

// Terms returns an iterator over the specific terms.
// It yields nothing if the set is Unrestricted or empty.
func (s TypeSet) Terms() iter.Seq[*types.Term] {
	return func(yield func(*types.Term) bool) {
		for _, t := range s.terms {
			if !yield(asTypesTerm(t)) {
				return
			}
		}
	}
}

// UnderlyingTypes returns an iterator over the underlying types of the
// specific terms. (For a term ~T, this is T itself.) It yields nothing
// if the set is Unrestricted or empty.
//
// Duplicates are not eliminated: for example, MyInt1|MyInt2
// yields int twice if both have underlying type int.
func (s TypeSet) UnderlyingTypes() iter.Seq[types.Type] {
	return func(yield func(types.Type) bool) {
		for _, t := range s.terms {
			if !yield(t.typ.Underlying()) {
				return
			}
		}
	}
}

// String returns a representation of the type set for debugging:
// "𝓤" if unrestricted, "∅" if empty, or a union of terms.
func (s TypeSet) String() string {
	switch {
	case s.terms == nil:
		return "𝓤"
	case len(s.terms) == 0:
		return "∅"
	default:
		return s.terms.String()
	}
}

// asTypesTerm converts a *term to a *types.Term without allocation.
//
// This is sound because the two types have identical layout:
// go/types declares "type Term term", where its term is
// struct{ tilde bool; typ Type }, like ours. The init function below
// verifies this assumption. Terms are never mutated after creation,
// and types.Term has no mutators, so sharing is safe.
//
// TODO(adonovan): this hack can be removed when we move this API to
// go/types (go.dev/issue/61013).
func asTypesTerm(t *term) *types.Term {
	return (*types.Term)(unsafe.Pointer(t))
}

func init() {
	// Check the layout assumption of asTypesTerm.
	if unsafe.Sizeof(term{}) != unsafe.Sizeof(types.Term{}) {
		panic("internal error: typeparams.term and types.Term differ in size")
	}
	for _, tilde := range []bool{false, true} {
		typ := types.Typ[types.Int]
		got := asTypesTerm(&term{tilde, typ})
		if got.Tilde() != tilde || got.Type() != typ {
			panic("internal error: typeparams.term and types.Term differ in layout")
		}
	}
}
