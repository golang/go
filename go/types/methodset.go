// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements method sets.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
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
		fmt.Fprintf(&buf, "\t%s\n", key(m.pkg, m.name))
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
	k := key(pkg, name)

	i := sort.Search(len(s.list), func(i int) bool {
		m := s.list[i]
		return key(m.pkg, m.name) >= k
	})

	if i < len(s.list) {
		m := s.list[i]
		if key(m.pkg, m.name) == k {
			return m
		}
	}
	return nil
}

// NewMethodSet computes the method set for the given type.
// BUG(gri): The pointer-ness of the receiver type is still ignored.
func NewMethodSet(typ Type) *MethodSet {
	isPtr := false
	if p, ok := typ.Underlying().(*Pointer); ok {
		typ = p.base
		isPtr = true
	}

	// TODO(gri) consult isPtr for precise method set computation
	_ = isPtr

	// method set up to the current depth
	// TODO(gri) allocate lazily, method sets are often empty
	base := make(methodSet)

	// named types that we have seen already
	seen := make(map[*Named]bool)

	// We treat the top-most level separately because it's simpler
	// (no incoming multiples) and because it's the common case.

	if t, _ := typ.(*Named); t != nil {
		seen[t] = true
		base.add(t.methods, false)
		typ = t.underlying
	}

	// embedded named types at the current and next lower depth
	type embedded struct {
		typ       *Named
		multiples bool
	}
	var current, next []embedded

	switch t := typ.(type) {
	case *Struct:
		for _, f := range t.fields {
			// Fields and methods must be distinct at the most shallow depth.
			// If they are not, the type checker reported an error before, so
			// we are ignoring potential conflicts here.
			if f.anonymous {
				// Ignore embedded basic types - only user-defined
				// named types can have methods or struct fields.
				if t, _ := f.typ.Deref().(*Named); t != nil {
					next = append(next, embedded{t, false})
				}
			}
		}

	case *Interface:
		base.add(t.methods, false)
	}

	// collect methods at next lower depth
	for len(next) > 0 {
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

		// field and method sets at this depth
		fset := make(fieldSet)
		mset := make(methodSet)

		for _, e := range current {
			if seen[e.typ] {
				continue
			}
			seen[e.typ] = true

			mset.add(e.typ.methods, e.multiples)

			switch t := e.typ.underlying.(type) {
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
						if t, _ := f.typ.Deref().(*Named); t != nil {
							next = append(next, embedded{t, e.multiples})
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
	}

	// collect methods
	var list []*Func
	for _, m := range base {
		if m != nil {
			list = append(list, m)
		}
	}
	sort.Sort(byKey(list))

	return &MethodSet{list}
}

// key computes a unique (lookup and sort) key given a package and name.
func key(pkg *Package, name string) string {
	if ast.IsExported(name) {
		return name
	}
	if pkg == nil {
		panic("unexported object without package information: " + name)
	}
	return pkg.path + "." + name
}

// A fieldSet is a set of fields and name collisions.
// A conflict indicates that multiple fields with the same package and name appeared.
type fieldSet map[string]*Field // a nil entry indicates a name collision

// Add adds field f to the field set s.
// If multiples is set, f appears multiple times
// and is treated as a collision at this level.
func (s fieldSet) add(f *Field, multiples bool) {
	k := key(f.pkg, f.name)
	// if f is not in the set, add it
	if !multiples {
		if _, found := s[k]; !found {
			s[k] = f
			return
		}
	}
	s[k] = nil // collision
}

// A methodSet is a set of methods and name collisions.
// A conflict indicates that multiple methods with the same package and name appeared.
type methodSet map[string]*Func // a nil entry indicates a name collision

// Add adds all methods in list to the method set s.
// If multiples is set, every method in list appears multiple times
// and is treated as a collision at this level.
func (s methodSet) add(list []*Func, multiples bool) {
	for _, m := range list {
		k := key(m.pkg, m.name)
		// if m is not in the set, add it
		if !multiples {
			if _, found := s[k]; !found {
				s[k] = m
				continue
			}
		}
		s[k] = nil // collision
	}
}

// byKey function lists can be sorted by key(pkg, name).
type byKey []*Func

func (a byKey) Len() int { return len(a) }
func (a byKey) Less(i, j int) bool {
	x := a[i]
	y := a[j]
	return key(x.pkg, x.name) < key(y.pkg, y.name)
}
func (a byKey) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
