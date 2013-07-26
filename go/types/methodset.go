// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements method sets.

package types

import (
	"bytes"
	"fmt"
	"sort"
	"sync"
)

// SelectorKind describes the kind of a selector expression recv.obj.
type SelectionKind int

const (
	FieldVal   SelectionKind = iota // recv.obj is a struct field selector
	MethodVal                       // recv.obj is a method selector
	MethodExpr                      // recv.obj is a method expression
	PackageObj                      // recv.obj is a qualified identifier
)

// A Selection describes a selector expression recv.obj.
type Selection struct {
	kind         SelectionKind
	obj          Object
	selectorPath // not valid for PackageObj
}

func (s *Selection) Kind() SelectionKind { return s.kind }
func (s *Selection) Obj() Object         { return s.obj }

// Type returns the type of the selector recv.obj which may be different
// from the type of the selected object obj.
//
// Given the example declarations:
//
// 	type T struct{ x int; E }
// 	type E struct{}
// 	func (e E) m() {}
// 	var p *T
//
// the following relationships exist:
//
//	Kind          Selector    Recv type    Selection type
//
//      FieldVal      p.x         T            int (type of x)
//      MethodVal     p.m         *T           func (e *T) m()
//      MethodExpr    T.m         T            func m(_ T)
//      PackageObj    math.Pi     nil          untyped numeric constant (type of Pi)
//
func (s *Selection) Type() Type {
	switch s.kind {
	case MethodVal:
		// The type of recv.obj is a method with its receiver type
		// changed to s.recv.
		// TODO(gri) Similar code is already in Method.Type below.
		//           Method may go away in favor of Selection.
		sig := *s.obj.(*Func).typ.(*Signature)
		recv := *sig.recv
		recv.typ = s.recv
		sig.recv = &recv
		return &sig

	case MethodExpr:
		// The type of recv.obj is a function (without receiver)
		// and an additional first argument of type recv.
		// TODO(gri) Similar code is already in call.go - factor!
		sig := *s.obj.(*Func).typ.(*Signature)
		arg0 := *sig.recv
		arg0.typ = s.recv
		var params []*Var
		if sig.params != nil {
			params = sig.params.vars
		}
		sig.params = NewTuple(append([]*Var{&arg0}, params...)...)
		return &sig
	}

	// In all other cases, the type of recv.obj is the type of obj.
	return s.obj.Type()
}

// TODO(gri) Replace Method in favor of Selection (make Method an interface?).

// A Method represents a concrete or abstract (interface) method x.m
// and corresponding path. The method belongs to the method set of x.
type Method struct {
	*Func
	selectorPath
}

// TODO(gri): expose this or something like it (e.g. lookupResult) to
// make life easier for callers of LookupFieldOrMethod.
func NewMethod(f *Func, recv Type, index []int, indirect bool) *Method {
	return &Method{f, selectorPath{recv, index, indirect}}
}

// Type returns the promoted signature type of m (i.e., the receiver
// type is Recv()).
func (m *Method) Type() Type {
	sig := *m.Func.typ.(*Signature)
	recv := *sig.recv
	recv.typ = m.recv
	sig.recv = &recv
	return &sig
}

func (m *Method) String() string {
	return fmt.Sprintf("method (%s).%s %s", m.Recv(), m.Name(), m.Type())
}

// A selectorPath describes the path from a value x to one of its fields
// or methods f for a selector expression x.f. A path may include implicit
// field selections (e.g., x.f may be x.a.b.c.f).
type selectorPath struct {
	recv     Type
	index    []int
	indirect bool
}

// Recv returns the type of x for the path p for x.f.
func (p *selectorPath) Recv() Type { return p.recv }

// Index describes the path from x to the concrete (possibly embedded) field
// or method f for the path p for x.f.
//
// The last index entry is the field or method index of the type declaring f;
// either:
//
//	1) the list of declared methods of a named type; or
//	2) the list of methods of an interface type; or
//	3) the list of fields of a struct type.
//
// The earlier index entries are the indices of the embedded fields implicitly
// traversed to get from (the type of) x to f, starting at embedding depth 0.
func (p *selectorPath) Index() []int { return p.index }

// Indirect reports whether any pointer indirection was required to get from
// a value x to f for the path p for x.f.
func (p *selectorPath) Indirect() bool { return p.indirect }

// A MethodSet is an ordered set of concrete or abstract (interface) methods.
// The zero value for a MethodSet is a ready-to-use empty method set.
type MethodSet struct {
	list []*Method
}

func (s *MethodSet) String() string {
	if s.Len() == 0 {
		return "MethodSet {}"
	}

	var buf bytes.Buffer
	fmt.Fprintln(&buf, "MethodSet {")
	for _, m := range s.list {
		fmt.Fprintf(&buf, "\t%s -> %s\n", m.Id(), m.Func)
	}
	fmt.Fprintln(&buf, "}")
	return buf.String()
}

// Len returns the number of methods in s.
func (s *MethodSet) Len() int { return len(s.list) }

// At returns the i'th method in s for 0 <= i < s.Len().
func (s *MethodSet) At(i int) *Method { return s.list[i] }

// Lookup returns the method with matching package and name, or nil if not found.
func (s *MethodSet) Lookup(pkg *Package, name string) *Method {
	if s.Len() == 0 {
		return nil
	}

	key := Id(pkg, name)
	i := sort.Search(len(s.list), func(i int) bool {
		m := s.list[i]
		return m.Id() >= key
	})
	if i < len(s.list) {
		m := s.list[i]
		if m.Id() == key {
			return m
		}
	}
	return nil
}

// Shared empty method set.
var emptyMethodSet MethodSet

// A cachedMethodSet provides access to a method set
// for a given type by computing it once on demand,
// and then caching it for future use. Threadsafe.
type cachedMethodSet struct {
	mset *MethodSet
	mu   sync.RWMutex // protects mset
}

// Of returns the (possibly cached) method set for typ.
// Threadsafe.
func (c *cachedMethodSet) of(typ Type) *MethodSet {
	c.mu.RLock()
	mset := c.mset
	c.mu.RUnlock()
	if mset == nil {
		mset = NewMethodSet(typ)
		c.mu.Lock()
		c.mset = mset
		c.mu.Unlock()
	}
	return mset
}

// NewMethodSet computes the method set for the given type T.
// It always returns a non-nil method set, even if it is empty.
func NewMethodSet(T Type) *MethodSet {
	// WARNING: The code in this function is extremely subtle - do not modify casually!

	// method set up to the current depth, allocated lazily
	var base methodSet

	// Start with typ as single entry at shallowest depth.
	// If typ is not a named type, insert a nil type instead.
	typ, isPtr := deref(T)
	t, _ := typ.(*Named)
	current := []embeddedType{{t, nil, isPtr, false}}

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
				mset = mset.add(t.methods, e.index, true, e.multiples)
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
	var list []*Method
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
type methodSet map[string]*Method // a nil entry indicates a name collision

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
			if _, found := s[key]; !found && (indirect || !ptrRecv(f)) {
				s[key] = &Method{f, selectorPath{index: concat(index, i), indirect: indirect}}
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
type byUniqueName []*Method

func (a byUniqueName) Len() int           { return len(a) }
func (a byUniqueName) Less(i, j int) bool { return a[i].Id() < a[j].Id() }
func (a byUniqueName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
