// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements Scopes.

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
// (parent) and contained (children) scopes. Objects may be inserted
// and looked up by name. The zero value for Scope is a ready-to-use
// empty scope.
type Scope struct {
	parent   *Scope
	children []*Scope
	node     ast.Node

	entries []Object
	objmap  map[string]Object // lazily allocated for large scopes
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
func (s *Scope) NumEntries() int { return len(s.entries) }

// At returns the i'th scope entry for 0 <= i < NumEntries().
func (s *Scope) At(i int) Object { return s.entries[i] }

// NumChildren() returns the number of scopes nested in s.
func (s *Scope) NumChildren() int { return len(s.children) }

// Child returns the i'th child scope for 0 <= i < NumChildren().
func (s *Scope) Child(i int) *Scope { return s.children[i] }

// Lookup returns the object in scope s with the given name if such an
// object exists; otherwise the result is nil.
func (s *Scope) Lookup(name string) Object {
	if s.objmap != nil {
		return s.objmap[name]
	}
	for _, obj := range s.entries {
		if obj.Name() == name {
			return obj
		}
	}
	return nil
}

// LookupParent follows the parent chain of scopes starting with s until
// it finds a scope where Lookup(name) returns a non-nil object, and then
// returns that object. If no such scope exists, the result is nil.
func (s *Scope) LookupParent(name string) Object {
	for ; s != nil; s = s.parent {
		if obj := s.Lookup(name); obj != nil {
			return obj
		}
	}
	return nil
}

// TODO(gri): Should Insert not be exported?

// Insert attempts to insert an object obj into scope s.
// If s already contains an alternative object alt with
// the same name, Insert leaves s unchanged and returns alt.
// Otherwise it inserts obj, sets the object's scope to
// s, and returns nil. Objects with blank "_" names are
// not inserted, but have their parent field set to s.
func (s *Scope) Insert(obj Object) Object {
	name := obj.Name()

	// spec: "The blank identifier, represented by the underscore
	// character _, may be used in a declaration like any other
	// identifier but the declaration does not introduce a new
	// binding."
	if name == "_" {
		obj.setParent(s)
		return nil
	}

	if alt := s.Lookup(name); alt != nil {
		return alt
	}

	// populate parallel objmap for larger scopes
	// TODO(gri) what is the right threshold? should we only use a map?
	if len(s.entries) == 32 {
		m := make(map[string]Object)
		for _, obj := range s.entries {
			m[obj.Name()] = obj
		}
		s.objmap = m
	}

	// add object
	s.entries = append(s.entries, obj)
	if s.objmap != nil {
		s.objmap[name] = obj
	}
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

	if len(s.entries) == 0 {
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
