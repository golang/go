// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various field and method lookup functions.

package types

import "go/ast"

// LookupFieldOrMethod looks up a field or method with given package and name
// in typ and returns the corresponding *Field or *Func, and an index sequence.
//
// The last index entry is the field or method index in the (possibly embedded)
// type where the entry was found, either:
//
//	1) the list of declared methods of a named type; or
//	2) the list of all methods (method set) of an interface type; or
//	3) the list of fields of a struct type.
//
// The earlier index entries are the indices of the embedded fields traversed
// to get to the found entry, starting at depth 0.
//
// If no entry is found, a nil object is returned. In this case, the returned
// index sequence points to an ambiguous entry if it exists, or it is nil.
//
func LookupFieldOrMethod(typ Type, pkg *Package, name string) (obj Object, index []int) {
	if name == "_" {
		return // empty fields/methods are never found
	}

	isPtr := false
	if p, ok := typ.Underlying().(*Pointer); ok {
		typ = p.base
		isPtr = true
	}

	// TODO(gri) consult isPtr for precise method set computation
	_ = isPtr

	// named types that we have seen already
	seen := make(map[*Named]bool)

	// embedded represents an embedded named type
	type embedded struct {
		typ       *Named // nil means use the outer typ variable instead
		index     []int  // field/method indices, starting with index at depth 0
		multiples bool   // if set, type appears multiple times at its depth
	}

	// Start with typ as single entry at lowest depth.
	// If typ is not a named type, insert a nil type instead.
	t, _ := typ.(*Named)
	current := []embedded{{t, nil, false}}

	// search current depth if there's work to do
	for len(current) > 0 {
		var next []embedded // embedded types found at current depth

		// look for (pkg, name) in all types at this depth
		for _, e := range current {
			// The very first time only, e.typ may be nil.
			// In this case, we don't have a named type and
			// we simply continue with the underlying type.
			if e.typ != nil {
				if seen[e.typ] {
					continue
				}
				seen[e.typ] = true

				// look for a matching attached method
				if i, m := lookupMethod(e.typ.methods, pkg, name); m != nil {
					// potential match
					assert(m.typ != nil)
					index = concat(e.index, i)
					if obj != nil || e.multiples {
						obj = nil // collision
						return
					}
					obj = m
				}

				// continue with underlying type
				typ = e.typ.underlying
			}

			switch t := typ.(type) {
			case *Struct:
				// look for a matching field and collect embedded types
				for i, f := range t.fields {
					if f.isMatch(pkg, name) {
						assert(f.typ != nil)
						index = concat(e.index, i)
						if obj != nil || e.multiples {
							obj = nil // collision
							return
						}
						obj = f
						continue
					}
					// Collect embedded struct fields for searching the next
					// lower depth, but only if we have not seen a match yet
					// (if we have a match it is either the desired field or
					// we have a name collision on the same depth; in either
					// case we don't need to look further).
					// Embedded fields are always of the form T or *T where
					// T is a named type. If e.typ appeared multiple times at
					// this depth, f.typ appears multiple times at the next
					// depth.
					if obj == nil && f.anonymous {
						// Ignore embedded basic types - only user-defined
						// named types can have methods or have struct fields.
						if t, _ := f.typ.Deref().(*Named); t != nil {
							next = append(next, embedded{t, concat(e.index, i), e.multiples})
						}
					}
				}

			case *Interface:
				// look for a matching method
				if i, m := lookupMethod(t.methods, pkg, name); m != nil {
					assert(m.typ != nil)
					index = concat(e.index, i)
					if obj != nil || e.multiples {
						obj = nil // collision
						return
					}
					obj = m
				}
			}
		}

		if obj != nil {
			return // found a match
		}

		// Consolidate next: collect multiple entries with the same
		// type into a single entry marked as containing multiples.
		n := len(next)
		if n > 1 {
			n := 0                       // number of entries w/ unique type
			prev := make(map[*Named]int) // index at which type was previously seen
			for _, e := range next {
				if i, found := prev[e.typ]; found {
					next[i].multiples = true
					// ignore this entry
				} else {
					prev[e.typ] = n
					next[n] = e
					n++
				}
			}
		}
		current = next[:n]
	}

	index = nil
	return // not found
}

// concat returns the result of concatenating list and i.
// The result does not share its underlying array with list.
func concat(list []int, i int) []int {
	var t []int
	t = append(t, list...)
	return append(t, i)
}

// MissingMethod returns (nil, false) if typ implements T, otherwise
// it returns the first missing method required by T and whether it
// is missing or simply has the wrong type.
//
func MissingMethod(typ Type, T *Interface) (method *Func, wrongType bool) {
	// TODO(gri): distinguish pointer and non-pointer receivers
	// an interface type implements T if it has no methods with conflicting signatures
	// Note: This is stronger than the current spec. Should the spec require this?

	// fast path for common case
	if T.IsEmpty() {
		return
	}

	if ityp, _ := typ.Underlying().(*Interface); ityp != nil {
		for _, m := range T.methods {
			_, obj := lookupMethod(ityp.methods, m.pkg, m.name)
			if obj != nil && !IsIdentical(obj.Type(), m.typ) {
				return m, true
			}
		}
		return
	}

	// a concrete type implements T if it implements all methods of T.
	for _, m := range T.methods {
		obj, _ := LookupFieldOrMethod(typ, m.pkg, m.name)
		if obj == nil {
			return m, false
		}
		if !IsIdentical(obj.Type(), m.typ) {
			return m, true
		}
	}
	return
}

// fieldIndex returns the index for the field with matching package and name, or a value < 0.
func fieldIndex(fields []*Field, pkg *Package, name string) int {
	if name == "_" {
		return -1 // blank identifiers are never found
	}
	for i, f := range fields {
		// spec:
		// "Two identifiers are different if they are spelled differently,
		// or if they appear in different packages and are not exported.
		// Otherwise, they are the same."
		if f.name == name && (ast.IsExported(name) || f.pkg.path == pkg.path) {
			return i
		}
	}
	return -1
}

// lookupMethod returns the index of and method with matching package and name, or (-1, nil).
func lookupMethod(methods []*Func, pkg *Package, name string) (int, *Func) {
	assert(name != "_")
	for i, m := range methods {
		// spec:
		// "Two identifiers are different if they are spelled differently,
		// or if they appear in different packages and are not exported.
		// Otherwise, they are the same."
		if m.name == name && (ast.IsExported(name) || m.pkg.path == pkg.path) {
			return i, m
		}
	}
	return -1, nil
}
