// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various field and method lookup functions.

package types

// LookupFieldOrMethod looks up a field or method with given package and name
// in T and returns the corresponding *Var or *Func, an index sequence, and a
// bool indicating if there were any pointer indirections on the path to the
// field or method. If addressable is set, T is the type of an addressable
// variable (only matters for method lookups).
//
// The last index entry is the field or method index in the (possibly embedded)
// type where the entry was found, either:
//
//	1) the list of declared methods of a named type; or
//	2) the list of all methods (method set) of an interface type; or
//	3) the list of fields of a struct type.
//
// The earlier index entries are the indices of the embedded struct fields
// traversed to get to the found entry, starting at depth 0.
//
// If no entry is found, a nil object is returned. In this case, the returned
// index and indirect values have the following meaning:
//
//	- If index != nil, the index sequence points to an ambiguous entry
//	(the same name appeared more than once at the same embedding level).
//
//	- If indirect is set, a method with a pointer receiver type was found
//      but there was no pointer on the path from the actual receiver type to
//	the method's formal receiver base type, nor was the receiver addressable.
//
func LookupFieldOrMethod(T Type, addressable bool, pkg *Package, name string) (obj Object, index []int, indirect bool) {
	// Methods cannot be associated to a named pointer type
	// (spec: "The type denoted by T is called the receiver base type;
	// it must not be a pointer or interface type and it must be declared
	// in the same package as the method.").
	// Thus, if we have a named pointer type, proceed with the underlying
	// pointer type but discard the result if it is a method since we would
	// not have found it for T (see also issue 8590).
	if t, _ := T.(*Named); t != nil {
		if p, _ := t.underlying.(*Pointer); p != nil {
			obj, index, indirect = lookupFieldOrMethod(p, false, pkg, name)
			if _, ok := obj.(*Func); ok {
				return nil, nil, false
			}
			return
		}
	}

	return lookupFieldOrMethod(T, addressable, pkg, name)
}

// TODO(gri) The named type consolidation and seen maps below must be
//           indexed by unique keys for a given type. Verify that named
//           types always have only one representation (even when imported
//           indirectly via different packages.)

func lookupFieldOrMethod(T Type, addressable bool, pkg *Package, name string) (obj Object, index []int, indirect bool) {
	// WARNING: The code in this function is extremely subtle - do not modify casually!
	//          This function and NewMethodSet should be kept in sync.

	if name == "_" {
		return // blank fields/methods are never found
	}

	typ, isPtr := deref(T)

	// *typ where typ is an interface has no methods.
	if isPtr && IsInterface(typ) {
		return
	}

	// Start with typ as single entry at shallowest depth.
	current := []embeddedType{{typ, nil, isPtr, false}}

	// Named types that we have seen already, allocated lazily.
	// Used to avoid endless searches in case of recursive types.
	// Since only Named types can be used for recursive types, we
	// only need to track those.
	// (If we ever allow type aliases to construct recursive types,
	// we must use type identity rather than pointer equality for
	// the map key comparison, as we do in consolidateMultiples.)
	var seen map[*Named]bool

	// search current depth
	for len(current) > 0 {
		var next []embeddedType // embedded types found at current depth

		// look for (pkg, name) in all types at current depth
		for _, e := range current {
			typ := e.typ

			// If we have a named type, we may have associated methods.
			// Look for those first.
			if named, _ := typ.(*Named); named != nil {
				if seen[named] {
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
				seen[named] = true

				// look for a matching attached method
				if i, m := lookupMethod(named.methods, pkg, name); m != nil {
					// potential match
					assert(m.typ != nil)
					index = concat(e.index, i)
					if obj != nil || e.multiples {
						return nil, index, false // collision
					}
					obj = m
					indirect = e.indirect
					continue // we can't have a matching field or interface method
				}

				// continue with underlying type
				typ = named.underlying
			}

			switch t := typ.(type) {
			case *Struct:
				// look for a matching field and collect embedded types
				for i, f := range t.fields {
					if f.sameId(pkg, name) {
						assert(f.typ != nil)
						index = concat(e.index, i)
						if obj != nil || e.multiples {
							return nil, index, false // collision
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
					// T is a type name. If e.typ appeared multiple times at
					// this depth, f.typ appears multiple times at the next
					// depth.
					if obj == nil && f.embedded {
						typ, isPtr := deref(f.typ)
						// TODO(gri) optimization: ignore types that can't
						// have fields or methods (only Named, Struct, and
						// Interface types need to be considered).
						next = append(next, embeddedType{typ, concat(e.index, i), e.indirect || isPtr, e.multiples})
					}
				}

			case *Interface:
				// look for a matching method
				// TODO(gri) t.allMethods is sorted - use binary search
				if i, m := lookupMethod(t.allMethods, pkg, name); m != nil {
					assert(m.typ != nil)
					index = concat(e.index, i)
					if obj != nil || e.multiples {
						return nil, index, false // collision
					}
					obj = m
					indirect = e.indirect
				}
			}
		}

		if obj != nil {
			// found a potential match
			// spec: "A method call x.m() is valid if the method set of (the type of) x
			//        contains m and the argument list can be assigned to the parameter
			//        list of m. If x is addressable and &x's method set contains m, x.m()
			//        is shorthand for (&x).m()".
			if f, _ := obj.(*Func); f != nil && ptrRecv(f) && !indirect && !addressable {
				return nil, nil, true // pointer/addressable receiver required
			}
			return
		}

		current = consolidateMultiples(next)
	}

	return nil, nil, false // not found
}

// embeddedType represents an embedded type
type embeddedType struct {
	typ       Type
	index     []int // embedded field indices, starting with index at depth 0
	indirect  bool  // if set, there was a pointer indirection on the path to this field
	multiples bool  // if set, typ appears multiple times at this depth
}

// consolidateMultiples collects multiple list entries with the same type
// into a single entry marked as containing multiples. The result is the
// consolidated list.
func consolidateMultiples(list []embeddedType) []embeddedType {
	if len(list) <= 1 {
		return list // at most one entry - nothing to do
	}

	n := 0                     // number of entries w/ unique type
	prev := make(map[Type]int) // index at which type was previously seen
	for _, e := range list {
		if i, found := lookupType(prev, e.typ); found {
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

func lookupType(m map[Type]int, typ Type) (int, bool) {
	// fast path: maybe the types are equal
	if i, found := m[typ]; found {
		return i, true
	}

	for t, i := range m {
		if Identical(t, typ) {
			return i, true
		}
	}

	return 0, false
}

// MissingMethod returns (nil, false) if V implements T, otherwise it
// returns a missing method required by T and whether it is missing or
// just has the wrong type.
//
// For non-interface types V, or if static is set, V implements T if all
// methods of T are present in V. Otherwise (V is an interface and static
// is not set), MissingMethod only checks that methods of T which are also
// present in V have matching types (e.g., for a type assertion x.(T) where
// x is of interface type V).
//
func MissingMethod(V Type, T *Interface, static bool) (method *Func, wrongType bool) {
	// fast path for common case
	if T.Empty() {
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
			case !Identical(obj.Type(), m.typ):
				return m, true
			}
		}
		return
	}

	// A concrete type implements T if it implements all methods of T.
	for _, m := range T.allMethods {
		obj, _, _ := lookupFieldOrMethod(V, false, m.pkg, m.name)

		f, _ := obj.(*Func)
		if f == nil {
			return m, false
		}

		if !Identical(f.typ, m.typ) {
			return m, true
		}
	}

	return
}

// assertableTo reports whether a value of type V can be asserted to have type T.
// It returns (nil, false) as affirmative answer. Otherwise it returns a missing
// method required by V and whether it is missing or just has the wrong type.
func assertableTo(V *Interface, T Type) (method *Func, wrongType bool) {
	// no static check is required if T is an interface
	// spec: "If T is an interface type, x.(T) asserts that the
	//        dynamic type of x implements the interface T."
	if _, ok := T.Underlying().(*Interface); ok && !strict {
		return
	}
	return MissingMethod(T, V, false)
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
