// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements Scopes.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"io"
	"iter"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
)

// A Scope maintains a set of objects and links to its containing
// (parent) and contained (children) scopes. Objects may be inserted
// and looked up by name. The zero value for Scope is a ready-to-use
// empty scope.
type Scope struct {
	parent      *Scope
	children    []*Scope
	number      int                      // parent.children[number-1] is this scope; 0 if there is no parent
	objects     map[string]Object        // lazily allocated
	pos, end    syntax.Pos               // scope extent; may be invalid
	comment     string                   // for debugging only
	isFunc      bool                     // set if this is a function scope (internal use only)
	sortedNames atomic.Pointer[[]string] // lazy cache of Names(), cleared during mutation
}

// NewScope returns a new, empty scope contained in the given parent
// scope, if any. The comment is for debugging only.
func NewScope(parent *Scope, pos, end syntax.Pos, comment string) *Scope {
	s := &Scope{parent: parent, pos: pos, end: end, comment: comment}
	// don't add children to Universe scope!
	if parent != nil && parent != Universe {
		parent.children = append(parent.children, s)
		s.number = len(parent.children)
	}
	return s
}

// Parent returns the scope's containing (parent) scope.
func (s *Scope) Parent() *Scope { return s.parent }

// Len returns the number of scope objects.
func (s *Scope) Len() int { return len(s.objects) }

// Names returns the scope's object names in sorted order.
// The caller must not mutate the array.
func (s *Scope) Names() []string {
	// We cache the result to avoid allocation on
	// each call, as this is a known hotspot.
	ptr := s.sortedNames.Load()
	if ptr == nil {
		// cache miss
		names := make([]string, len(s.objects))
		i := 0
		for name := range s.objects {
			names[i] = name
			i++
		}
		slices.Sort(names)
		ptr = &names

		// Don't overwrite if another goroutine got there first.
		s.sortedNames.CompareAndSwap(nil, &names)
	}
	return *ptr
}

// NumChildren returns the number of scopes nested in s.
func (s *Scope) NumChildren() int { return len(s.children) }

// Child returns the i'th child scope for 0 <= i < NumChildren().
func (s *Scope) Child(i int) *Scope { return s.children[i] }

// Lookup returns the object in scope s with the given name if such an
// object exists; otherwise the result is nil.
func (s *Scope) Lookup(name string) Object { return resolve(name, s.objects[name]) }

// Objects returns the sequence of objects in the scope in name order.
//
// The caller should not mutate the Scope during iteration.
//
// Example:
//
//	for obj := range s.Objects() { ... }
func (s *Scope) Objects() iter.Seq[Object] {
	return func(yield func(obj Object) bool) {
		names := s.Names()
		for _, name := range names {
			if !yield(s.Lookup(name)) {
				break
			}
		}
	}
}

// lookupIgnoringCase returns the objects in scope s whose names match
// the given name ignoring case. If exported is set, only exported names
// are returned.
func (s *Scope) lookupIgnoringCase(name string, exported bool) []Object {
	var matches []Object
	for _, n := range s.Names() {
		if (!exported || isExported(n)) && strings.EqualFold(n, name) {
			matches = append(matches, s.Lookup(n))
		}
	}
	return matches
}

// Insert attempts to insert an object obj into scope s.
// If s already contains an alternative object alt with
// the same name, Insert leaves s unchanged and returns alt.
// Otherwise it inserts obj, sets the object's parent scope
// if not already set, and returns nil.
func (s *Scope) Insert(obj Object) Object {
	name := obj.Name()
	if alt := s.Lookup(name); alt != nil {
		return alt
	}
	s.insert(name, obj)
	// TODO(gri) Can we always set the parent to s (or is there
	// a need to keep the original parent or some race condition)?
	// If we can, than we may not need environment.lookupScope
	// which is only there so that we get the correct scope for
	// marking "used" dot-imported packages.
	if obj.Parent() == nil {
		obj.setParent(s)
	}
	return nil
}

// InsertLazy is like Insert, but allows deferring construction of the
// inserted object until it's accessed with Lookup. The Object
// returned by resolve must have the same name as given to InsertLazy.
// If s already contains an alternative object with the same name,
// InsertLazy leaves s unchanged and returns false. Otherwise it
// records the binding and returns true. The object's parent scope
// will be set to s after resolve is called.
func (s *Scope) InsertLazy(name string, resolve func() Object) bool {
	if s.objects[name] != nil {
		return false
	}
	s.insert(name, &lazyObject{parent: s, resolve: resolve})
	return true
}

func (s *Scope) insert(name string, obj Object) {
	if s.objects == nil {
		s.objects = make(map[string]Object)
	}
	s.sortedNames.Store(nil) // clear cache
	s.objects[name] = obj
}

// WriteTo writes a string representation of the scope to w,
// with the scope elements sorted by name.
// The level of indentation is controlled by n >= 0, with
// n == 0 for no indentation.
// If recurse is set, it also writes nested (children) scopes.
func (s *Scope) WriteTo(w io.Writer, n int, recurse bool) {
	const ind = ".  "
	indn := strings.Repeat(ind, n)

	fmt.Fprintf(w, "%s%s scope %p {\n", indn, s.comment, s)

	indn1 := indn + ind
	for _, name := range s.Names() {
		fmt.Fprintf(w, "%s%s\n", indn1, s.Lookup(name))
	}

	if recurse {
		for _, s := range s.children {
			s.WriteTo(w, n+1, recurse)
		}
	}

	fmt.Fprintf(w, "%s}\n", indn)
}

// String returns a string representation of the scope, for debugging.
func (s *Scope) String() string {
	var buf strings.Builder
	s.WriteTo(&buf, 0, false)
	return buf.String()
}

// A lazyObject represents an imported Object that has not been fully
// resolved yet by its importer.
type lazyObject struct {
	parent  *Scope
	resolve func() Object
	obj     Object
	once    sync.Once
}

// resolve returns the Object represented by obj, resolving lazy
// objects as appropriate.
func resolve(name string, obj Object) Object {
	if lazy, ok := obj.(*lazyObject); ok {
		lazy.once.Do(func() {
			obj := lazy.resolve()

			if _, ok := obj.(*lazyObject); ok {
				panic("recursive lazy object")
			}
			if obj.Name() != name {
				panic("lazy object has unexpected name")
			}

			if obj.Parent() == nil {
				obj.setParent(lazy.parent)
			}
			lazy.obj = obj
		})

		obj = lazy.obj
	}
	return obj
}

// stub implementations so *lazyObject implements Object and we can
// store them directly into Scope.elems.
func (*lazyObject) Parent() *Scope                     { panic("unreachable") }
func (*lazyObject) Pos() syntax.Pos                    { panic("unreachable") }
func (*lazyObject) Pkg() *Package                      { panic("unreachable") }
func (*lazyObject) Name() string                       { panic("unreachable") }
func (*lazyObject) Type() Type                         { panic("unreachable") }
func (*lazyObject) Exported() bool                     { panic("unreachable") }
func (*lazyObject) Id() string                         { panic("unreachable") }
func (*lazyObject) String() string                     { panic("unreachable") }
func (*lazyObject) order() uint32                      { panic("unreachable") }
func (*lazyObject) setType(Type)                       { panic("unreachable") }
func (*lazyObject) setOrder(uint32)                    { panic("unreachable") }
func (*lazyObject) setParent(*Scope)                   { panic("unreachable") }
func (*lazyObject) sameId(*Package, string, bool) bool { panic("unreachable") }
func (*lazyObject) scopePos() syntax.Pos               { panic("unreachable") }
func (*lazyObject) setScopePos(syntax.Pos)             { panic("unreachable") }
