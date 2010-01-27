// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import "go/token"

type ObjKind int

// The list of possible Object kinds.
const (
	Err ObjKind = iota // object kind unknown (forward reference or error)
	Pkg         // package
	Con         // constant
	Typ         // type
	Var         // variable
	Fun         // function or method
)


// An Object describes a language entity such as a package,
// constant, type, variable, or function (incl. methods).
//
type Object struct {
	Kind ObjKind
	Pos  token.Position // declaration position
	Name string         // declared name
}


func NewObj(kind ObjKind, pos token.Position, name string) *Object {
	return &Object{kind, pos, name}
}


// IsExported returns whether obj is exported.
func (obj *Object) IsExported() bool { return IsExported(obj.Name) }


// A Scope maintains the set of named language entities visible
// in the scope and a link to the immediately surrounding (outer)
// scope.
//
type Scope struct {
	Outer   *Scope
	Objects map[string]*Object
}


// NewScope creates a new scope nested in the outer scope.
func NewScope(outer *Scope) *Scope { return &Scope{outer, make(map[string]*Object)} }


// Declare attempts to insert a named object into the scope s.
// If the scope does not contain an object with that name yet,
// Declare inserts the object, and returns it. Otherwise, the
// scope remains unchanged and Declare returns the object found
// in the scope instead.
func (s *Scope) Declare(obj *Object) *Object {
	decl, found := s.Objects[obj.Name]
	if !found {
		s.Objects[obj.Name] = obj
		decl = obj
	}
	return decl
}


// Lookup looks up an object in the current scope chain.
// The result is nil if the object is not found.
//
func (s *Scope) Lookup(name string) *Object {
	for ; s != nil; s = s.Outer {
		if obj, found := s.Objects[name]; found {
			return obj
		}
	}
	return nil
}
