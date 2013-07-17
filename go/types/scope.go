// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
	"io"
	"strings"
)

// TODO(gri) Provide scopes with a name or other mechanism so that
//           objects can use that information for better printing.

// A Scope maintains a set of objects and links to its containing
// (parent) and contained (children) scopes.
// Objects may be inserted and looked up by name, or by package path
// and name. A nil *Scope acts like an empty scope for operations that
// do not modify the scope or access a scope's parent scope.
type Scope struct {
	parent   *Scope
	children []*Scope
	entries  []Object
	node     ast.Node
}

// NewScope returns a new, empty scope contained in the given parent
// scope, if any.
func NewScope(parent *Scope) *Scope {
	scope := &Scope{parent: parent}
	// don't add children to Universe scope!
	if parent != nil && parent != Universe {
		parent.children = append(parent.children, scope)
	}
	return scope
}

// Parent returns the scope's containing (parent) scope.
func (s *Scope) Parent() *Scope { return s.parent }

// Node returns the ast.Node responsible for this scope,
// which may be one of the following:
//
//	ast.File
//	ast.FuncType
//	ast.BlockStmt
//	ast.IfStmt
//	ast.SwitchStmt
//	ast.TypeSwitchStmt
//	ast.CaseClause
//	ast.CommClause
//	ast.ForStmt
//	ast.RangeStmt
//
// The result is nil if there is no corresponding node
// (universe and package scopes).
func (s *Scope) Node() ast.Node { return s.node }

// NumEntries() returns the number of scope entries.
// If s == nil, the result is 0.
func (s *Scope) NumEntries() int {
	if s == nil {
		return 0 // empty scope
	}
	return len(s.entries)
}

// At returns the i'th scope entry for 0 <= i < NumEntries().
func (s *Scope) At(i int) Object { return s.entries[i] }

// NumChildren() returns the number of scopes nested in s.
// If s == nil, the result is 0.
func (s *Scope) NumChildren() int {
	if s == nil {
		return 0
	}
	return len(s.children)
}

// Child returns the i'th child scope for 0 <= i < NumChildren().
func (s *Scope) Child(i int) *Scope { return s.children[i] }

// Lookup returns the object in scope s with the given package
// and name if such an object exists; otherwise the result is nil.
// A nil scope acts like an empty scope, and parent scopes are ignored.
//
// If pkg != nil, both pkg.Path() and name are used to identify an
// entry, per the Go rules for identifier equality. If pkg == nil,
// only the name is used and the package path is ignored.
func (s *Scope) Lookup(pkg *Package, name string) Object {
	if s == nil {
		return nil // empty scope
	}

	// fast path: only the name must match
	if pkg == nil {
		for _, obj := range s.entries {
			if obj.Name() == name {
				return obj
			}
		}
		return nil
	}

	// slow path: both pkg path and name must match
	for _, obj := range s.entries {
		if obj.SameName(pkg, name) {
			return obj
		}
	}

	// not found
	return nil

	// TODO(gri) Optimize Lookup by also maintaining a map representation
	//           for larger scopes.
}

// LookupParent follows the parent chain of scopes starting with s until it finds
// a scope where Lookup(nil, name) returns a non-nil object, and then returns that
// object. If no such scope exists, the result is nil.
func (s *Scope) LookupParent(name string) Object {
	for s != nil {
		if obj := s.Lookup(nil, name); obj != nil {
			return obj
		}
		s = s.parent
	}
	return nil
}

// TODO(gri): Should Insert not be exported?

// Insert attempts to insert an object obj into scope s.
// If s already contains an object with the same package path
// and name, Insert leaves s unchanged and returns that object.
// Otherwise it inserts obj, sets the object's scope to s, and
// returns nil. The object must not have the blank _ name.
//
func (s *Scope) Insert(obj Object) Object {
	name := obj.Name()
	assert(name != "_")
	if alt := s.Lookup(obj.Pkg(), name); alt != nil {
		return alt
	}
	s.entries = append(s.entries, obj)
	obj.setParent(s)
	return nil
}

// WriteTo writes a string representation of the scope to w.
// The level of indentation is controlled by n >= 0, with
// n == 0 for no indentation.
// If recurse is set, it also prints nested (children) scopes.
func (s *Scope) WriteTo(w io.Writer, n int, recurse bool) {
	const ind = ".  "
	indn := strings.Repeat(ind, n)

	if s.NumEntries() == 0 {
		fmt.Fprintf(w, "%sscope %p {}\n", indn, s)
		return
	}

	fmt.Fprintf(w, "%sscope %p {\n", indn, s)
	indn1 := indn + ind
	for _, obj := range s.entries {
		fmt.Fprintf(w, "%s%s\n", indn1, obj)
	}

	if recurse {
		for _, s := range s.children {
			fmt.Fprintln(w)
			s.WriteTo(w, n+1, recurse)
		}
	}

	fmt.Fprintf(w, "%s}", indn)
}

// String returns a string representation of the scope, for debugging.
func (s *Scope) String() string {
	var buf bytes.Buffer
	s.WriteTo(&buf, 0, false)
	return buf.String()
}
