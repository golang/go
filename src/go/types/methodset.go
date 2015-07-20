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

	var buf bytes.Buffer
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
	named, _ := typ.(*Named)

	// *typ where typ is an interface has no methods.
	if isPtr {
		utyp := typ
		if named != nil {
			utyp = named.underlying
		}
		if _, ok := utyp.(*Interface); ok {
			return &emptyMethodSet
		}
	}

	// Start with typ as single entry at shallowest depth.
	// If typ is not a named type, insert a nil type instead.
	current := []embeddedType{{named, nil, isPtr, false}}

	// named types that we have seen already, allocated lazily
	var seen map[*Named]bool

	// collect methods at current depth
	for len(current) > 0 {
		var next []embeddedType // embedded types found at current depth

		// field and method sets at current depth, allocated lazily
		var fset fieldSet
		var mset methodSet

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

				mset = mset.add(e.typ.methods, e.index, e.indirect, e.multiples)

				// continue with underlying type
				typ = e.typ.underlying
			}

			switch t := typ.(type) {
			case *Struct:
				for i, f := range t.fields {
					fset = fset.add(f, e.multiples)

					// Embedded fields are always of the form T or *T where
					// T is a named type. If typ appeared multiple times at
					// this depth, f.Type appears multiple times at the next
					// depth.
					if f.anonymous {
						// Ignore embedded basic types - only user-defined
						// named types can have methods or struct fields.
						typ, isPtr := deref(f.typ)
						if t, _ := typ.(*Named); t != nil {
							next = append(next, embeddedType{t, concat(e.index, i), e.indirect || isPtr, e.multiples})
						}
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
	sort.Sort(byUniqueName(list))
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
// The receiver must exist.
func ptrRecv(f *Func) bool {
	_, isPtr := deref(f.typ.(*Signature).recv.typ)
	return isPtr
}

// byUniqueName function lists can be sorted by their unique names.
type byUniqueName []*Selection

func (a byUniqueName) Len() int           { return len(a) }
func (a byUniqueName) Less(i, j int) bool { return a[i].obj.Id() < a[j].obj.Id() }
func (a byUniqueName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
