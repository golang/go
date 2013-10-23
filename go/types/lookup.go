// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various field and method lookup functions.

package types

// TODO(gri) The named type consolidation and seen maps below must be
//           indexed by unique keys for a given type. Verify that named
//           types always have only one representation (even when imported
//           indirectly via different packages.)

// LookupFieldOrMethod looks up a field or method with given package and name
// in T and returns the corresponding *Var or *Func, an index sequence, and a
// bool indicating if there were any pointer indirections on the path to the
// field or method.
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
func LookupFieldOrMethod(T Type, pkg *Package, name string) (obj Object, index []int, indirect bool) {
	obj, index, indirect = lookupFieldOrMethod(T, pkg, name)
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
	if t, _ := T.(*Named); t != nil {
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

func lookupFieldOrMethod(T Type, pkg *Package, name string) (obj Object, index []int, indirect bool) {
	// WARNING: The code in this function is extremely subtle - do not modify casually!
	//          This function and NewMethodSet should be kept in sync.

	if name == "_" {
		return // blank fields/methods are never found
	}

	typ, isPtr := deref(T)
	named, _ := typ.(*Named)

	// *typ where typ is an interface has no methods.
	if isPtr {
		utyp := typ
		if named != nil {
			utyp = named.underlying
		}
		if _, ok := utyp.(*Interface); ok {
			return
		}
	}

	// Start with typ as single entry at shallowest depth.
	// If typ is not a named type, insert a nil type instead.
	current := []embeddedType{{named, nil, isPtr, false}}

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
					if f.sameId(pkg, name) {
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
				// TODO(gri) t.allMethods is sorted - use binary search
				if i, m := lookupMethod(t.allMethods, pkg, name); m != nil {
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

// MissingMethod returns (nil, false) if V implements T, otherwise it
// returns a missing method required by T and whether it is missing or
// just has the wrong type.
//
// For non-interface types V, or if static is set, V implements T if all
// methods of T are present in V. Otherwise (V is an interface and static
// is not set), MissingMethod only checks that methods of T which are also
// present in V have matching types (e.g., for a type assertion x.(T) where
// x is of interface type typ).
//
func MissingMethod(V Type, T *Interface, static bool) (method *Func, wrongType bool) {
	// fast path for common case
	if T.NumMethods() == 0 {
		return
	}

	// TODO(gri) Consider using method sets here. Might be more efficient.

	if ityp, _ := V.Underlying().(*Interface); ityp != nil {
		// TODO(gri) allMethods is sorted - can do this more efficiently
		for _, m := range T.allMethods {
			_, obj := lookupMethod(ityp.allMethods, m.pkg, m.name)
			switch {
			case obj == nil:
				if static {
					return m, false
				}
			case !IsIdentical(obj.Type(), m.typ):
				return m, true
			}
		}
		return
	}

	// A concrete type implements T if it implements all methods of T.
	for _, m := range T.allMethods {
		obj, _, indirect := lookupFieldOrMethod(V, m.pkg, m.name)
		if obj == nil {
			return m, false
		}

		f, _ := obj.(*Func)
		if f == nil {
			return m, false
		}

		// verify that f is in the method set of typ
		if !indirect && ptrRecv(f) {
			return m, false
		}

		if !IsIdentical(obj.Type(), m.typ) {
			return m, true
		}
	}

	return
}

// deref dereferences typ if it is a *Pointer and returns its base and true.
// Otherwise it returns (typ, false).
func deref(typ Type) (Type, bool) {
	if p, _ := typ.(*Pointer); p != nil {
		return p.base, true
	}
	return typ, false
}

// derefStructPtr dereferences typ if it is a (named or unnamed) pointer to a
// (named or unnamed) struct and returns its base. Otherwise it returns typ.
func derefStructPtr(typ Type) Type {
	if p, _ := typ.Underlying().(*Pointer); p != nil {
		if _, ok := p.base.Underlying().(*Struct); ok {
			return p.base
		}
	}
	return typ
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
	if name != "_" {
		for i, f := range fields {
			if f.sameId(pkg, name) {
				return i
			}
		}
	}
	return -1
}

// lookupMethod returns the index of and method with matching package and name, or (-1, nil).
func lookupMethod(methods []*Func, pkg *Package, name string) (int, *Func) {
	if name != "_" {
		for i, m := range methods {
			if m.sameId(pkg, name) {
				return i, m
			}
		}
	}
	return -1, nil
}
