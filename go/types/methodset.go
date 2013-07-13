// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements method sets.

package types

import (
	"bytes"
	"fmt"
	"sort"
)

// A MethodSet is an ordered set of methods.
type MethodSet struct {
	list []*Func
}

func (s *MethodSet) String() string {
	var buf bytes.Buffer
	fmt.Fprint(&buf, "MethodSet{")
	if len(s.list) > 0 {
		fmt.Fprintln(&buf)
	}
	for _, m := range s.list {
		fmt.Fprintf(&buf, "\t%s\n", m.uniqueName())
	}
	fmt.Fprintln(&buf, "}")
	return buf.String()
}

// Len returns the number of methods in s.
func (s *MethodSet) Len() int { return len(s.list) }

// At returns the i'th method in s.
func (s *MethodSet) At(i int) *Func { return s.list[i] }

// Lookup returns the method with matching package and name, or nil if not found.
func (s *MethodSet) Lookup(pkg *Package, name string) *Func {
	key := (&object{pkg: pkg, name: name}).uniqueName()

	i := sort.Search(len(s.list), func(i int) bool {
		m := s.list[i]
		return m.uniqueName() >= key
	})

	if i < len(s.list) {
		m := s.list[i]
		if m.uniqueName() == key {
			return m
		}
	}
	return nil
}

// NewMethodSet computes the method set for the given type.
// BUG(gri): The pointer-ness of the receiver type is still ignored.
func NewMethodSet(typ Type) *MethodSet {
	// method set up to the current depth
	// TODO(gri) allocate lazily, method sets are often empty
	base := make(methodSet)

	// Start with typ as single entry at lowest depth.
	// If typ is not a named type, insert a nil type instead.
	typ, isPtr := deref(typ)
	t, _ := typ.(*Named)
	current := []embeddedType{{t, nil, isPtr, false}}

	// named types that we have seen already
	seen := make(map[*Named]bool)

	// collect methods at current depth
	for len(current) > 0 {
		var next []embeddedType // embedded types found at current depth

		// field and method sets for current depth
		fset := make(fieldSet)
		mset := make(methodSet)

		for _, e := range current {
			// The very first time only, e.typ may be nil.
			// In this case, we don't have a named type and
			// we simply continue with the underlying type.
			if e.typ != nil {
				if seen[e.typ] {
					// We have seen this type before, at a more shallow depth
					// (note that multiples of this type at the current depth
					// were eliminated before). The type at that depth shadows
					// this same type at the current depth, so we can ignore
					// this one.
					continue
				}
				seen[e.typ] = true

				mset.add(e.typ.methods, e.multiples)

				// continue with underlying type
				typ = e.typ.underlying
			}

			switch t := typ.(type) {
			case *Struct:
				for _, f := range t.fields {
					fset.add(f, e.multiples)

					// Embedded fields are always of the form T or *T where
					// T is a named type. If typ appeared multiple times at
					// this depth, f.Type appears multiple times at the next
					// depth.
					if f.anonymous {
						// Ignore embedded basic types - only user-defined
						// named types can have methods or have struct fields.
						typ, isPtr := deref(f.typ)
						if t, _ := typ.(*Named); t != nil {
							next = append(next, embeddedType{t, nil, e.indirect || isPtr, e.multiples})
						}
					}
				}

			case *Interface:
				mset.add(t.methods, e.multiples)
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
				base[k] = m
			}
		}

		// Multiple fields with matching names collide at this depth and shadow all
		// entries further down; add them as collisions to base if no entries with
		// matching names exist already.
		for k, f := range fset {
			if f == nil {
				if _, found := base[k]; !found {
					base[k] = nil
				}
			}
		}

		current = consolidateMultiples(next)
	}

	// collect methods
	var list []*Func
	for _, m := range base {
		if m != nil {
			list = append(list, m)
		}
	}
	sort.Sort(byUniqueName(list))

	return &MethodSet{list}
}

// A fieldSet is a set of fields and name collisions.
// A conflict indicates that multiple fields with the same package and name appeared.
type fieldSet map[string]*Field // a nil entry indicates a name collision

// Add adds field f to the field set s.
// If multiples is set, f appears multiple times
// and is treated as a collision at this level.
func (s fieldSet) add(f *Field, multiples bool) {
	key := f.uniqueName()
	// if f is not in the set, add it
	if !multiples {
		if _, found := s[key]; !found {
			s[key] = f
			return
		}
	}
	s[key] = nil // collision
}

// A methodSet is a set of methods and name collisions.
// A conflict indicates that multiple methods with the same package and name appeared.
type methodSet map[string]*Func // a nil entry indicates a name collision

// Add adds all methods in list to the method set s.
// If multiples is set, every method in list appears multiple times
// and is treated as a collision at this level.
func (s methodSet) add(list []*Func, multiples bool) {
	for _, m := range list {
		key := m.uniqueName()
		// if m is not in the set, add it
		if !multiples {
			if _, found := s[key]; !found {
				s[key] = m
				continue
			}
		}
		s[key] = nil // collision
	}
}

// byUniqueName function lists can be sorted by their unique names.
type byUniqueName []*Func

func (a byUniqueName) Len() int           { return len(a) }
func (a byUniqueName) Less(i, j int) bool { return a[i].uniqueName() < a[j].uniqueName() }
func (a byUniqueName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
