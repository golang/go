// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// TODO(gri) use a different symbol instead of ⊤ for the set of all types
//           (⊤ is hard to distinguish from T in some fonts)

// A term describes elementary type sets:
//
//   ∅:  (*term)(nil)     == ∅                      // set of no types (empty set)
//   ⊤:  &term{}          == ⊤                      // set of all types
//   T:  &term{false, T}  == {T}                    // set of type T
//  ~t:  &term{true, t}   == {t' | under(t') == t}  // set of types with underlying type t
//
type term struct {
	tilde bool // valid if typ != nil
	typ   Type
}

func (x *term) String() string {
	switch {
	case x == nil:
		return "∅"
	case x.typ == nil:
		return "⊤"
	case x.tilde:
		return "~" + x.typ.String()
	default:
		return x.typ.String()
	}
}

// equal reports whether x and y represent the same type set.
func (x *term) equal(y *term) bool {
	// easy cases
	switch {
	case x == nil || y == nil:
		return x == y
	case x.typ == nil || y.typ == nil:
		return x.typ == y.typ
	}
	// ∅ ⊂ x, y ⊂ ⊤

	return x.tilde == y.tilde && Identical(x.typ, y.typ)
}

// union returns the union x ∪ y: zero, one, or two non-nil terms.
func (x *term) union(y *term) (_, _ *term) {
	// easy cases
	switch {
	case x == nil && y == nil:
		return nil, nil // ∅ ∪ ∅ == ∅
	case x == nil:
		return y, nil // ∅ ∪ y == y
	case y == nil:
		return x, nil // x ∪ ∅ == x
	case x.typ == nil:
		return x, nil // ⊤ ∪ y == ⊤
	case y.typ == nil:
		return y, nil // x ∪ ⊤ == ⊤
	}
	// ∅ ⊂ x, y ⊂ ⊤

	if x.disjoint(y) {
		return x, y // x ∪ y == (x, y) if x ∩ y == ∅
	}
	// x.typ == y.typ

	// ~t ∪ ~t == ~t
	// ~t ∪  T == ~t
	//  T ∪ ~t == ~t
	//  T ∪  T ==  T
	if x.tilde || !y.tilde {
		return x, nil
	}
	return y, nil
}

// intersect returns the intersection x ∩ y.
func (x *term) intersect(y *term) *term {
	// easy cases
	switch {
	case x == nil || y == nil:
		return nil // ∅ ∩ y == ∅ and ∩ ∅ == ∅
	case x.typ == nil:
		return y // ⊤ ∩ y == y
	case y.typ == nil:
		return x // x ∩ ⊤ == x
	}
	// ∅ ⊂ x, y ⊂ ⊤

	if x.disjoint(y) {
		return nil // x ∩ y == ∅ if x ∩ y == ∅
	}
	// x.typ == y.typ

	// ~t ∩ ~t == ~t
	// ~t ∩  T ==  T
	//  T ∩ ~t ==  T
	//  T ∩  T ==  T
	if !x.tilde || y.tilde {
		return x
	}
	return y
}

// includes reports whether t ∈ x.
func (x *term) includes(t Type) bool {
	// easy cases
	switch {
	case x == nil:
		return false // t ∈ ∅ == false
	case x.typ == nil:
		return true // t ∈ ⊤ == true
	}
	// ∅ ⊂ x ⊂ ⊤

	u := t
	if x.tilde {
		u = under(u)
	}
	return Identical(x.typ, u)
}

// subsetOf reports whether x ⊆ y.
func (x *term) subsetOf(y *term) bool {
	// easy cases
	switch {
	case x == nil:
		return true // ∅ ⊆ y == true
	case y == nil:
		return false // x ⊆ ∅ == false since x != ∅
	case y.typ == nil:
		return true // x ⊆ ⊤ == true
	case x.typ == nil:
		return false // ⊤ ⊆ y == false since y != ⊤
	}
	// ∅ ⊂ x, y ⊂ ⊤

	if x.disjoint(y) {
		return false // x ⊆ y == false if x ∩ y == ∅
	}
	// x.typ == y.typ

	// ~t ⊆ ~t == true
	// ~t ⊆ T == false
	//  T ⊆ ~t == true
	//  T ⊆  T == true
	return !x.tilde || y.tilde
}

// disjoint reports whether x ∩ y == ∅.
// x.typ and y.typ must not be nil.
func (x *term) disjoint(y *term) bool {
	ux := x.typ
	if y.tilde {
		ux = under(ux)
	}
	uy := y.typ
	if x.tilde {
		uy = under(uy)
	}
	return !Identical(ux, uy)
}
