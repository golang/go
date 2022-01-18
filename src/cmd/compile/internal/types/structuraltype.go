// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// Implementation of structural type computation for types.

// TODO: we would like to depend only on the types2 computation of structural type,
// but we can only do that the next time we change the export format and export
// structural type info along with each constraint type, since the compiler imports
// types directly into types1 format.

// A term describes elementary type sets:
//
// term{false, T}  set of type T
// term{true, T}   set of types with underlying type t
// term{}          empty set (we specifically check for typ == nil)
type term struct {
	tilde bool
	typ   *Type
}

// StructuralType returns the structural type of an interface, or nil if it has no
// structural type.
func (t *Type) StructuralType() *Type {
	sts, _ := specificTypes(t)
	var su *Type
	for _, st := range sts {
		u := st.typ.Underlying()
		if su != nil {
			u = match(su, u)
			if u == nil {
				return nil
			}
		}
		// su == nil || match(su, u) != nil
		su = u
	}
	return su
}

// If x and y are identical, match returns x.
// If x and y are identical channels but for their direction
// and one of them is unrestricted, match returns the channel
// with the restricted direction.
// In all other cases, match returns nil.
// x and y are assumed to be underlying types, hence are not named types.
func match(x, y *Type) *Type {
	if IdenticalStrict(x, y) {
		return x
	}

	if x.IsChan() && y.IsChan() && IdenticalStrict(x.Elem(), y.Elem()) {
		// We have channels that differ in direction only.
		// If there's an unrestricted channel, select the restricted one.
		// If both have the same direction, return x (either is fine).
		switch {
		case x.ChanDir().CanSend() && x.ChanDir().CanRecv():
			return y
		case y.ChanDir().CanSend() && y.ChanDir().CanRecv():
			return x
		}
	}
	return nil
}

// specificTypes returns the list of specific types of an interface type or nil if
// there are none. It also returns a flag that indicates, for an empty term list
// result, whether it represents the empty set, or the infinite set of all types (in
// both cases, there are no specific types).
func specificTypes(t *Type) (list []term, inf bool) {
	t.wantEtype(TINTER)

	// We have infinite term list before processing any type elements
	// (or if there are no type elements).
	inf = true
	for _, m := range t.Methods().Slice() {
		var r2 []term
		inf2 := false

		switch {
		case m.IsMethod():
			inf2 = true

		case m.Type.IsUnion():
			nt := m.Type.NumTerms()
			for i := 0; i < nt; i++ {
				t, tilde := m.Type.Term(i)
				if t.IsInterface() {
					r3, r3inf := specificTypes(t)
					if r3inf {
						// Union with an infinite set of types is
						// infinite, so skip remaining terms.
						r2 = nil
						inf2 = true
						break
					}
					// Add the elements of r3 to r2.
					for _, r3e := range r3 {
						r2 = insertType(r2, r3e)
					}
				} else {
					r2 = insertType(r2, term{tilde, t})
				}
			}

		case m.Type.IsInterface():
			r2, inf2 = specificTypes(m.Type)

		default:
			// m.Type is a single non-interface type, so r2 is just a
			// one-element list, inf2 is false.
			r2 = []term{{false, m.Type}}
		}

		if inf2 {
			// If the current type element has infinite types,
			// its intersection with r is just r, so skip this type element.
			continue
		}

		if inf {
			// If r is infinite, then the intersection of r and r2 is just r2.
			list = r2
			inf = false
			continue
		}

		// r and r2 are finite, so intersect r and r2.
		var r3 []term
		for _, re := range list {
			for _, r2e := range r2 {
				if tm := intersect(re, r2e); tm.typ != nil {
					r3 = append(r3, tm)
				}
			}
		}
		list = r3
	}
	return
}

// insertType adds t to the returned list if it is not already in list.
func insertType(list []term, tm term) []term {
	for i, elt := range list {
		if new := union(elt, tm); new.typ != nil {
			// Replace existing elt with the union of elt and new.
			list[i] = new
			return list
		}
	}
	return append(list, tm)
}

// If x and y are disjoint, return term with nil typ (which means the union should
// include both types). If x and y are not disjoint, return the single type which is
// the union of x and y.
func union(x, y term) term {
	if disjoint(x, y) {
		return term{false, nil}
	}
	if x.tilde || !y.tilde {
		return x
	}
	return y
}

// intersect returns the intersection x ∩ y.
func intersect(x, y term) term {
	if disjoint(x, y) {
		return term{false, nil}
	}
	if !x.tilde || y.tilde {
		return x
	}
	return y
}

// disjoint reports whether x ∩ y == ∅.
func disjoint(x, y term) bool {
	ux := x.typ
	if y.tilde {
		ux = ux.Underlying()
	}
	uy := y.typ
	if x.tilde {
		uy = uy.Underlying()
	}
	return !IdenticalStrict(ux, uy)
}
