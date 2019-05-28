// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements method sets.

package types

import (
	"fmt"
	"sort"
	"strings"
)

// A MethodSet is an ordered set of concrete or abstract (interface) methods;
// a method is a MethodVal selection, and they are ordered by ascending m.Obj().Id().
// The zero value for a MethodSet is a ready-to-use empty method set.
type MethodSet struct {
	list []*Selection
}

func (s *MethodSet) String() string {
	if s.Len() == 0 {
		return "MethodSet {}"
	}

	var buf strings.Builder
	fmt.Fprintln(&buf, "MethodSet {")
	for _, f := range s.list {
		fmt.Fprintf(&buf, "\t%s\n", f)
	}
	fmt.Fprintln(&buf, "}")
	return buf.String()
}

// Len returns the number of methods in s.
func (s *MethodSet) Len() int { return len(s.list) }

// At returns the i'th method in s for 0 <= i < s.Len().
func (s *MethodSet) At(i int) *Selection { return s.list[i] }

// Lookup returns the method with matching package and name, or nil if not found.
func (s *MethodSet) Lookup(pkg *Package, name string) *Selection {
	if s.Len() == 0 {
		return nil
	}

	key := Id(pkg, name)
	i := sort.Search(len(s.list), func(i int) bool {
		m := s.list[i]
		return m.obj.Id() >= key
	})
	if i < len(s.list) {
		m := s.list[i]
		if m.obj.Id() == key {
			return m
		}
	}
	return nil
}

// Shared empty method set.
var emptyMethodSet MethodSet

// NewMethodSet returns the method set for the given type T.
// It always returns a non-nil method set, even if it is empty.
func NewMethodSet(T Type) *MethodSet {
	// WARNING: The code in this function is extremely subtle - do not modify casually!
	//          This function and lookupFieldOrMethod should be kept in sync.

	// method set up to the current depth, allocated lazily
	var base methodSet

	typ, isPtr := deref(T)

	// *typ where typ is an interface has no methods.
	if isPtr && IsInterface(typ) {
		return &emptyMethodSet
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

	// collect methods at current depth
	for len(current) > 0 {
		var next []embeddedType // embedded types found at current depth

		// field and method sets at current depth, allocated lazily
		var fset fieldSet
		var mset methodSet

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

				mset = mset.add(named.methods, e.index, e.indirect, e.multiples)

				// continue with underlying type
				typ = named.underlying
			}

			switch t := typ.(type) {
			case *Struct:
				for i, f := range t.fields {
					fset = fset.add(f, e.multiples)

					// Embedded fields are always of the form T or *T where
					// T is a type name. If typ appeared multiple times at
					// this depth, f.Type appears multiple times at the next
					// depth.
					if f.embedded {
						typ, isPtr := deref(f.typ)
						// TODO(gri) optimization: ignore types that can't
						// have fields or methods (only Named, Struct, and
						// Interface types need to be considered).
						next = append(next, embeddedType{typ, concat(e.index, i), e.indirect || isPtr, e.multiples})
					}
				}

			case *Interface:
				mset = mset.add(t.allMethods, e.index, true, e.multiples)
			}
		}

		// Add methods and collisions at this depth to base if no entries with matching
		// names exist already.
		for k, m := range mset {
			if _, found := base[k]; !found {
				// Fields collide with methods of the same name at this depth.
				if _, found := fset[k]; found {
					m = nil // collision
				}
				if base == nil {
					base = make(methodSet)
				}
				base[k] = m
			}
		}

		// Multiple fields with matching names collide at this depth and shadow all
		// entries further down; add them as collisions to base if no entries with
		// matching names exist already.
		for k, f := range fset {
			if f == nil {
				if _, found := base[k]; !found {
					if base == nil {
						base = make(methodSet)
					}
					base[k] = nil // collision
				}
			}
		}

		current = consolidateMultiples(next)
	}

	if len(base) == 0 {
		return &emptyMethodSet
	}

	// collect methods
	var list []*Selection
	for _, m := range base {
		if m != nil {
			m.recv = T
			list = append(list, m)
		}
	}
	// sort by unique name
	sort.Slice(list, func(i, j int) bool {
		return list[i].obj.Id() < list[j].obj.Id()
	})
	return &MethodSet{list}
}

// A fieldSet is a set of fields and name collisions.
// A collision indicates that multiple fields with the
// same unique id appeared.
type fieldSet map[string]*Var // a nil entry indicates a name collision

// Add adds field f to the field set s.
// If multiples is set, f appears multiple times
// and is treated as a collision.
func (s fieldSet) add(f *Var, multiples bool) fieldSet {
	if s == nil {
		s = make(fieldSet)
	}
	key := f.Id()
	// if f is not in the set, add it
	if !multiples {
		if _, found := s[key]; !found {
			s[key] = f
			return s
		}
	}
	s[key] = nil // collision
	return s
}

// A methodSet is a set of methods and name collisions.
// A collision indicates that multiple methods with the
// same unique id appeared.
type methodSet map[string]*Selection // a nil entry indicates a name collision

// Add adds all functions in list to the method set s.
// If multiples is set, every function in list appears multiple times
// and is treated as a collision.
func (s methodSet) add(list []*Func, index []int, indirect bool, multiples bool) methodSet {
	if len(list) == 0 {
		return s
	}
	if s == nil {
		s = make(methodSet)
	}
	for i, f := range list {
		key := f.Id()
		// if f is not in the set, add it
		if !multiples {
			// TODO(gri) A found method may not be added because it's not in the method set
			// (!indirect && ptrRecv(f)). A 2nd method on the same level may be in the method
			// set and may not collide with the first one, thus leading to a false positive.
			// Is that possible? Investigate.
			if _, found := s[key]; !found && (indirect || !ptrRecv(f)) {
				s[key] = &Selection{MethodVal, nil, f, concat(index, i), indirect}
				continue
			}
		}
		s[key] = nil // collision
	}
	return s
}

// ptrRecv reports whether the receiver is of the form *T.
func ptrRecv(f *Func) bool {
	// If a method's receiver type is set, use that as the source of truth for the receiver.
	// Caution: Checker.funcDecl (decl.go) marks a function by setting its type to an empty
	// signature. We may reach here before the signature is fully set up: we must explicitly
	// check if the receiver is set (we cannot just look for non-nil f.typ).
	if sig, _ := f.typ.(*Signature); sig != nil && sig.recv != nil {
		_, isPtr := deref(sig.recv.typ)
		return isPtr
	}

	// If a method's type is not set it may be a method/function that is:
	// 1) client-supplied (via NewFunc with no signature), or
	// 2) internally created but not yet type-checked.
	// For case 1) we can't do anything; the client must know what they are doing.
	// For case 2) we can use the information gathered by the resolver.
	return f.hasPtrRecv
}
