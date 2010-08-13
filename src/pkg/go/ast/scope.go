// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

type ObjKind int

// The list of possible Object kinds.
const (
	Bad ObjKind = iota // bad object
	Pkg                // package
	Con                // constant
	Typ                // type
	Var                // variable
	Fun                // function or method
)


var objKindStrings = [...]string{
	Bad: "bad",
	Pkg: "package",
	Con: "const",
	Typ: "type",
	Var: "var",
	Fun: "func",
}


func (kind ObjKind) String() string { return objKindStrings[kind] }


// An Object describes a language entity such as a package,
// constant, type, variable, or function (incl. methods).
//
type Object struct {
	Kind ObjKind
	Name string      // declared name
	Decl interface{} // corresponding Field, xxxSpec or FuncDecl
}


func NewObj(kind ObjKind, name string) *Object {
	return &Object{kind, name, nil}
}


// IsExported returns whether obj is exported.
func (obj *Object) IsExported() bool { return IsExported(obj.Name) }


// A Scope maintains the set of named language entities visible
// in the scope and a link to the immediately surrounding (outer)
// scope.
//
type Scope struct {
	Outer   *Scope
	Objects []*Object // in declaration order
	// Implementation note: In some cases (struct fields,
	// function parameters) we need the source order of
	// variables. Thus for now, we store scope entries
	// in a linear list. If scopes become very large
	// (say, for packages), we may need to change this
	// to avoid slow lookups.
}


// NewScope creates a new scope nested in the outer scope.
func NewScope(outer *Scope) *Scope {
	const n = 4 // initial scope capacity, must be > 0
	return &Scope{outer, make([]*Object, 0, n)}
}


func (s *Scope) append(obj *Object) {
	n := len(s.Objects)
	if n >= cap(s.Objects) {
		new := make([]*Object, 2*n)
		copy(new, s.Objects)
		s.Objects = new
	}
	s.Objects = s.Objects[0 : n+1]
	s.Objects[n] = obj
}


func (s *Scope) lookup(name string) *Object {
	for _, obj := range s.Objects {
		if obj.Name == name {
			return obj
		}
	}
	return nil
}


// Declare attempts to insert a named object into the scope s.
// If the scope does not contain an object with that name yet,
// Declare inserts the object, and returns it. Otherwise, the
// scope remains unchanged and Declare returns the object found
// in the scope instead.
func (s *Scope) Declare(obj *Object) *Object {
	alt := s.lookup(obj.Name)
	if alt == nil {
		s.append(obj)
		alt = obj
	}
	return alt
}


// Lookup looks up an object in the current scope chain.
// The result is nil if the object is not found.
//
func (s *Scope) Lookup(name string) *Object {
	for ; s != nil; s = s.Outer {
		if obj := s.lookup(name); obj != nil {
			return obj
		}
	}
	return nil
}
