// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

// A Scope maintains the set of identifiers visible
// in the scope and a link to the immediately surrounding
// (outer) scope.
//
//	NOTE: WORK IN PROGRESS
//
type Scope struct {
	Outer *Scope
	Names map[string]*Ident
}


// NewScope creates a new scope nested in the outer scope.
func NewScope(outer *Scope) *Scope { return &Scope{outer, make(map[string]*Ident)} }


// Declare inserts an identifier into the scope s. If the
// declaration succeeds, the result is true, if the identifier
// exists already in the scope, the result is false.
//
func (s *Scope) Declare(ident *Ident) bool {
	if _, found := s.Names[ident.Value]; found {
		return false
	}
	s.Names[ident.Value] = ident
	return true
}


// Lookup looks up an identifier in the current scope chain.
// If the identifier is found, it is returned; otherwise the
// result is nil.
//
func (s *Scope) Lookup(name string) *Ident {
	for ; s != nil; s = s.Outer {
		if ident, found := s.Names[name]; found {
			return ident
		}
	}
	return nil
}


// TODO(gri) Uncomment once this code is needed.
/*
var Universe = Scope {
	Names: map[string]*Ident {
		// basic types
		"bool": nil,
		"byte": nil,
		"int8": nil,
		"int16": nil,
		"int32": nil,
		"int64": nil,
		"uint8": nil,
		"uint16": nil,
		"uint32": nil,
		"uint64": nil,
		"float32": nil,
		"float64": nil,
		"string": nil,

		// convenience types
		"int": nil,
		"uint": nil,
		"uintptr": nil,
		"float": nil,

		// constants
		"false": nil,
		"true": nil,
		"iota": nil,
		"nil": nil,

		// functions
		"cap": nil,
		"len": nil,
		"new": nil,
		"make": nil,
		"panic": nil,
		"panicln": nil,
		"print": nil,
		"println": nil,
	}
}
*/