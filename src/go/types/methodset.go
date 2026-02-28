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
// a method is a [MethodVal] selection, and they are ordered by ascending m.Obj().Id().
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

// Note: NewMethodSet is intended for external use only as it
//       requires interfaces to be complete. It may be used
//       internally if LookupFieldOrMethod completed the same
//       interfaces beforehand.

// NewMethodSet returns the method set for the given type T.
// It always returns a non-nil method set, even if it is empty.
func NewMethodSet(T Type) *MethodSet {
	// WARNING: The code in this function is extremely subtle - do not modify casually!
	//          This function and lookupFieldOrMethod should be kept in sync.

	// TODO(rfindley) confirm that this code is in sync with lookupFieldOrMethod
	//                with respect to type params.

	// Methods cannot be associated with a named pointer type.
	// (spec: "The type denoted by T is called the receiver base type;
	// it must not be a pointer or interface type and it must be declared
	// in the same package as the method.").
	if t := asNamed(T); t != nil && isPointer(t) {
		return &emptyMethodSet
	}

	// method set up to the current depth, allocated lazily
	var base methodSet

	typ, isPtr := deref(T)

	// *typ where typ is an interface has no methods.
	if isPtr && IsInterface(typ) {
		return &emptyMethodSet
	}

	// Start with typ as single entry at shallowest depth.
	current := []embeddedType{{typ, nil, isPtr, false}}

	// seen tracks named types that we have seen already, allocated lazily.
	// Used to avoid endless searches in case of recursive types.
	//
	// We must use a lookup on identity rather than a simple map[*Named]bool as
	// instantiated types may be identical but not equal.
	var seen instanceLookup

	// collect methods at current depth
	for len(current) > 0 {
		var next []embeddedType // embedded types found at current depth

		// field and method sets at current depth, indexed by names (Id's), and allocated lazily
		var fset map[string]bool // we only care about the field names
		var mset methodSet

		for _, e := range current {
			typ := e.typ

			// If we have a named type, we may have associated methods.
			// Look for those first.
			if named := asNamed(typ); named != nil {
				if alt := seen.lookup(named); alt != nil {
					// We have seen this type before, at a more shallow depth
					// (note that multiples of this type at the current depth
					// were consolidated before). The type at that depth shadows
					// this same type at the current depth, so we can ignore
					// this one.
					continue
				}
				seen.add(named)

				for i := 0; i < named.NumMethods(); i++ {
					mset = mset.addOne(named.Method(i), concat(e.index, i), e.indirect, e.multiples)
				}
			}

			switch t := typ.Underlying().(type) {
			case *Struct:
				for i, f := range t.fields {
					if fset == nil {
						fset = make(map[string]bool)
					}
					fset[f.Id()] = true

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
				mset = mset.add(t.typeSet().methods, e.index, true, e.multiples)
			}
		}

		// Add methods and collisions at this depth to base if no entries with matching
		// names exist already.
		for k, m := range mset {
			if _, found := base[k]; !found {
				// Fields collide with methods of the same name at this depth.
				if fset[k] {
					m = nil // collision
				}
				if base == nil {
					base = make(methodSet)
				}
				base[k] = m
			}
		}

		// Add all (remaining) fields at this depth as collisions (since they will
		// hide any method further down) if no entries with matching names exist already.
		for k := range fset {
			if _, found := base[k]; !found {
				if base == nil {
					base = make(methodSet)
				}
				base[k] = nil // collision
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

// A methodSet is a set of methods and name collisions.
// A collision indicates that multiple methods with the
// same unique id, or a field with that id appeared.
type methodSet map[string]*Selection // a nil entry indicates a name collision

// Add adds all functions in list to the method set s.
// If multiples is set, every function in list appears multiple times
// and is treated as a collision.
func (s methodSet) add(list []*Func, index []int, indirect bool, multiples bool) methodSet {
	if len(list) == 0 {
		return s
	}
	for i, f := range list {
		s = s.addOne(f, concat(index, i), indirect, multiples)
	}
	return s
}

func (s methodSet) addOne(f *Func, index []int, indirect bool, multiples bool) methodSet {
	if s == nil {
		s = make(methodSet)
	}
	key := f.Id()
	// if f is not in the set, add it
	if !multiples {
		// TODO(gri) A found method may not be added because it's not in the method set
		// (!indirect && f.hasPtrRecv()). A 2nd method on the same level may be in the method
		// set and may not collide with the first one, thus leading to a false positive.
		// Is that possible? Investigate.
		if _, found := s[key]; !found && (indirect || !f.hasPtrRecv()) {
			s[key] = &Selection{MethodVal, nil, f, index, indirect}
			return s
		}
	}
	s[key] = nil // collision
	return s
}
