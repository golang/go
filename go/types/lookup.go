// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various field and method lookup functions.

package types

import "go/ast"

// LookupFieldOrMethod looks up a field or method with given package and name in typ.
// If an entry is found, obj is the corresponding *Field or *Func. For fields, index
// is the index sequence to reach the (possibly embedded) field; for methods, index
// is nil; and collision is false. If no entry is found, obj is nil, index is undefined,
// and collision indicates if the reason for not finding an entry was a name collision.
//
func LookupFieldOrMethod(typ Type, pkg *Package, name string) (obj Object, index []int, collision bool) {
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

	// We treat the top-most level separately because it's simpler
	// (no incoming multiples) and because it's the common case.

	if t, _ := typ.(*Named); t != nil {
		seen[t] = true
		if m := lookupMethod(t.methods, pkg, name); m != nil {
			assert(m.typ != nil)
			return m, nil, false
		}
		typ = t.underlying
	}

	// embedded named types at the current and next lower depth
	type embedded struct {
		typ       *Named
		index     []int // field index sequence
		multiples bool
	}
	var current, next []embedded

	switch t := typ.(type) {
	case *Struct:
		for i, f := range t.fields {
			if f.isMatch(pkg, name) {
				assert(f.typ != nil)
				return f, []int{i}, false
			}
			if f.anonymous {
				// Ignore embedded basic types - only user-defined
				// named types can have methods or struct fields.
				if t, _ := f.typ.Deref().(*Named); t != nil {
					next = append(next, embedded{t, []int{i}, false})
				}
			}
		}

	case *Interface:
		if m := lookupMethod(t.methods, pkg, name); m != nil {
			assert(m.typ != nil)
			return m, nil, false
		}
	}

	// search the next depth if we don't have a match yet and there's work to do
	for obj == nil && len(next) > 0 {
		// Consolidate next: collect multiple entries with the same
		// type into a single entry marked as containing multiples.
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
		// next[:n] is the list of embedded entries to process

		// The underlying arrays of current and next are different, thus
		// swapping is safe and they never share the same underlying array.
		current, next = next[:n], current[:0] // don't waste underlying array

		// look for name in all types at this depth
		for _, e := range current {
			if seen[e.typ] {
				continue
			}
			seen[e.typ] = true

			// look for a matching attached method
			if m := lookupMethod(e.typ.methods, pkg, name); m != nil {
				// potential match
				assert(m.typ != nil)
				if obj != nil || e.multiples {
					return nil, nil, true
				}
				obj = m
				index = nil
			}

			switch t := e.typ.underlying.(type) {
			case *Struct:
				// look for a matching field and collect embedded types
				for i, f := range t.fields {
					if f.isMatch(pkg, name) {
						assert(f.typ != nil)
						if obj != nil || e.multiples {
							return nil, nil, true
						}
						obj = f
						index = append(index[:0], e.index...)
						index = append(index, i)
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
							var copy []int
							copy = append(copy, e.index...)
							copy = append(copy, i)
							next = append(next, embedded{t, copy, e.multiples})
						}
					}
				}

			case *Interface:
				// look for a matching method
				if m := lookupMethod(t.methods, pkg, name); m != nil {
					assert(m.typ != nil)
					if obj != nil || e.multiples {
						return nil, nil, true
					}
					obj = m
					index = nil
				}
			}
		}
	}

	return
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
			obj := lookupMethod(ityp.methods, m.pkg, m.name)
			if obj != nil && !IsIdentical(obj.Type(), m.typ) {
				return m, true
			}
		}
		return
	}

	// a concrete type implements T if it implements all methods of T.
	for _, m := range T.methods {
		obj, _, _ := LookupFieldOrMethod(typ, m.pkg, m.name)
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

// lookupMethod returns the method with matching package and name, or nil.
func lookupMethod(methods []*Func, pkg *Package, name string) *Func {
	assert(name != "_")
	for _, m := range methods {
		// spec:
		// "Two identifiers are different if they are spelled differently,
		// or if they appear in different packages and are not exported.
		// Otherwise, they are the same."
		if m.name == name && (ast.IsExported(name) || m.pkg.path == pkg.path) {
			return m
		}
	}
	return nil
}
