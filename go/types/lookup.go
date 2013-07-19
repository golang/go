// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various field and method lookup functions.

package types

// TODO(gri) The named type consolidation and seen maps below must be
//           indexed by unique keys for a given type. Verify that named
//           types always have only one representation (even when imported
//           indirectly via different packages.)

// TODO(gri) Move Field to objects.go?

// A Field represents a struct field x.f and corresponding path.
type Field struct {
	*Var
	selectorPath
}

func lookupResult(typ Type, obj Object, index []int, indirect bool) Object {
	switch obj := obj.(type) {
	case nil:
		return nil
	case *Var:
		return &Field{obj, selectorPath{typ, index, indirect}}
	case *Func:
		return &Method{obj, selectorPath{typ, index, indirect}}
	}
	panic("unreachable")
}

// LookupFieldOrMethod looks up a field or method with given package and name
// in typ and returns the corresponding *Var or *Func, an index sequence,
// and a bool indicating if there were any pointer indirections on the path
// to the field or method.
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
func LookupFieldOrMethod(typ Type, pkg *Package, name string) (obj Object, index []int, indirect bool) {
	obj, index, indirect = lookupFieldOrMethod(typ, pkg, name)
	if obj != nil {
		return
	}

	// TODO(gri) The code below is not needed if we are looking for methods only,
	//           and it can be done always if we look for fields only. Consider
	//           providing LookupField and LookupMethod as well.

	// If we didn't find anything, we still might have a field p.x as in:
	//
	//	type S struct{ x int }
	//      func (*S) m() {}
	//	type P *S
	//	var p P
	//
	// which requires that we start the search with the underlying type
	// of P (i.e., *S). We cannot do this always because we might find
	// methods that don't exist for P but for S (e.g., m). Thus, if the
	// result is a method we need to discard it.
	//
	// TODO(gri) WTF? There isn't a more direct way? Perhaps we should
	//           outlaw named types to pointer types - they are almost
	//           never what one wants, anyway.
	if t, _ := typ.(*Named); t != nil {
		u := t.underlying
		if _, ok := u.(*Pointer); ok {
			// typ is a named type with an underlying type of the form *T,
			// start the search with the underlying type *T
			if obj2, index2, indirect2 := lookupFieldOrMethod(u, pkg, name); obj2 != nil {
				// only if the result is a field can we keep it
				if _, ok := obj2.(*Var); ok {
					return obj2, index2, indirect2
				}
			}
		}
	}

	return
}

func lookupFieldOrMethod(typ Type, pkg *Package, name string) (obj Object, index []int, indirect bool) {
	// WARNING: The code in this function is extremely subtle - do not modify casually!

	if name == "_" {
		return // blank fields/methods are never found
	}

	// Start with typ as single entry at lowest depth.
	// If typ is not a named type, insert a nil type instead.
	typ, isPtr := deref(typ)
	t, _ := typ.(*Named)
	current := []embeddedType{{t, nil, isPtr, false}}

	// named types that we have seen already, allocated lazily
	var seen map[*Named]bool

	// search current depth
	for len(current) > 0 {
		var next []embeddedType // embedded types found at current depth

		// look for (pkg, name) in all types at current depth
		for _, e := range current {
			// The very first time only, e.typ may be nil.
			// In this case, we don't have a named type and
			// we simply continue with the underlying type.
			if e.typ != nil {
				if seen[e.typ] {
					// We have seen this type before, at a more shallow depth
					// (note that multiples of this type at the current depth
					// were consolidated before). The type at that depth shadows
					// this same type at the current depth, so we can ignore
					// this one.
					continue
				}
				if seen == nil {
					seen = make(map[*Named]bool)
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
					indirect = e.indirect
					continue // we can't have a matching field or interface method
				}

				// continue with underlying type
				typ = e.typ.underlying
			}

			switch t := typ.(type) {
			case *Struct:
				// look for a matching field and collect embedded types
				for i, f := range t.fields {
					if f.SameName(pkg, name) {
						assert(f.typ != nil)
						index = concat(e.index, i)
						if obj != nil || e.multiples {
							obj = nil // collision
							return
						}
						obj = f
						indirect = e.indirect
						continue // we can't have a matching interface method
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
						// named types can have methods or struct fields.
						typ, isPtr := deref(f.typ)
						if t, _ := typ.(*Named); t != nil {
							next = append(next, embeddedType{t, concat(e.index, i), e.indirect || isPtr, e.multiples})
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
					indirect = e.indirect
				}
			}
		}

		if obj != nil {
			return // found a match
		}

		current = consolidateMultiples(next)
	}

	index = nil
	indirect = false
	return // not found
}

// embeddedType represents an embedded named type
type embeddedType struct {
	typ       *Named // nil means use the outer typ variable instead
	index     []int  // embedded field indices, starting with index at depth 0
	indirect  bool   // if set, there was a pointer indirection on the path to this field
	multiples bool   // if set, typ appears multiple times at this depth
}

// consolidateMultiples collects multiple list entries with the same type
// into a single entry marked as containing multiples. The result is the
// consolidated list.
func consolidateMultiples(list []embeddedType) []embeddedType {
	if len(list) <= 1 {
		return list // at most one entry - nothing to do
	}

	n := 0                       // number of entries w/ unique type
	prev := make(map[*Named]int) // index at which type was previously seen
	for _, e := range list {
		if i, found := prev[e.typ]; found {
			list[i].multiples = true
			// ignore this entry
		} else {
			prev[e.typ] = n
			list[n] = e
			n++
		}
	}
	return list[:n]
}

// MissingMethod returns (nil, false) if typ implements T, otherwise
// it returns the first missing method required by T and whether it
// is missing or simply has the wrong type.
//
func MissingMethod(typ Type, T *Interface) (method *Func, wrongType bool) {
	// an interface type implements T if it has no methods with conflicting signatures
	// Note: This is stronger than the current spec. Should the spec require this?

	// fast path for common case
	if T.NumMethods() == 0 {
		return
	}

	// An interface type implements T if it has at least the methods of T.
	if ityp, _ := typ.Underlying().(*Interface); ityp != nil {
		for _, m := range T.methods {
			_, obj := lookupMethod(ityp.methods, m.pkg, m.name)
			if obj == nil {
				return m, false
			}
			if !IsIdentical(obj.Type(), m.typ) {
				return m, true
			}
		}
		return
	}

	// A concrete type implements T if it implements all methods of T.
	for _, m := range T.methods {
		obj, _, indirect := lookupFieldOrMethod(typ, m.pkg, m.name)
		if obj == nil {
			return m, false
		}

		f, _ := obj.(*Func)
		if f == nil {
			return m, false
		}

		// verify that f is in the method set of typ
		// (the receiver is nil if f is an interface method)
		if recv := f.typ.(*Signature).recv; recv != nil {
			if _, isPtr := deref(recv.typ); isPtr && !indirect {
				return m, false
			}
		}

		if !IsIdentical(obj.Type(), m.typ) {
			return m, true
		}
	}

	return
}

// Deref dereferences typ if it is a pointer and returns its base and true.
// Otherwise it returns (typ, false).
func deref(typ Type) (Type, bool) {
	if p, _ := typ.(*Pointer); p != nil {
		return p.base, true
	}
	return typ, false
}

// concat returns the result of concatenating list and i.
// The result does not share its underlying array with list.
func concat(list []int, i int) []int {
	var t []int
	t = append(t, list...)
	return append(t, i)
}

// fieldIndex returns the index for the field with matching package and name, or a value < 0.
func fieldIndex(fields []*Var, pkg *Package, name string) int {
	if name == "_" {
		return -1 // blank identifiers are never found
	}
	for i, f := range fields {
		if f.SameName(pkg, name) {
			return i
		}
	}
	return -1
}

// lookupMethod returns the index of and method with matching package and name, or (-1, nil).
func lookupMethod(methods []*Func, pkg *Package, name string) (int, *Func) {
	assert(name != "_")
	for i, m := range methods {
		if m.SameName(pkg, name) {
			return i, m
		}
	}
	return -1, nil
}
