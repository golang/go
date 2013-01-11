// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// An Object describes a named language entity such as a package,
// constant, type, variable, function (incl. methods), or label.
// All objects implement the Object interface.
//
type Object interface {
	anObject()
	GetName() string
}

// A Package represents the contents (objects) of a Go package.
type Package struct {
	implementsObject
	Name    string
	Path    string              // import path, "" for current (non-imported) package
	Scope   *Scope              // nil for current (non-imported) package for now
	Imports map[string]*Package // map of import paths to packages
}

// A Const represents a declared constant.
type Const struct {
	implementsObject
	Name string
	Type Type
	Val  interface{}
}

// A TypeName represents a declared type.
type TypeName struct {
	implementsObject
	Name string
	Type Type // *NamedType or *Basic
}

// A Variable represents a declared variable (including function parameters and results).
type Var struct {
	implementsObject
	Name string
	Type Type
}

// A Func represents a declared function.
type Func struct {
	implementsObject
	Name string
	Type Type // *Signature or *Builtin
}

func (obj *Package) GetName() string  { return obj.Name }
func (obj *Const) GetName() string    { return obj.Name }
func (obj *TypeName) GetName() string { return obj.Name }
func (obj *Var) GetName() string      { return obj.Name }
func (obj *Func) GetName() string     { return obj.Name }

func (obj *Package) GetType() Type  { return nil }
func (obj *Const) GetType() Type    { return obj.Type }
func (obj *TypeName) GetType() Type { return obj.Type }
func (obj *Var) GetType() Type      { return obj.Type }
func (obj *Func) GetType() Type     { return obj.Type }

// All concrete objects embed implementsObject which
// ensures that they all implement the Object interface.
type implementsObject struct{}

func (*implementsObject) anObject() {}

// A Scope maintains the set of named language entities declared
// in the scope and a link to the immediately surrounding (outer)
// scope.
//
type Scope struct {
	Outer *Scope
	Elems []Object          // scope entries in insertion order
	large map[string]Object // for fast lookup - only used for larger scopes
}

// Lookup returns the object with the given name if it is
// found in scope s, otherwise it returns nil. Outer scopes
// are ignored.
//
func (s *Scope) Lookup(name string) Object {
	if s.large != nil {
		return s.large[name]
	}
	for _, obj := range s.Elems {
		if obj.GetName() == name {
			return obj
		}
	}
	return nil
}

// Insert attempts to insert an object obj into scope s.
// If s already contains an object with the same name,
// Insert leaves s unchanged and returns that object.
// Otherwise it inserts obj and returns nil.
//
func (s *Scope) Insert(obj Object) Object {
	name := obj.GetName()
	if alt := s.Lookup(name); alt != nil {
		return alt
	}
	s.Elems = append(s.Elems, obj)
	if len(s.Elems) > 20 {
		if s.large == nil {
			m := make(map[string]Object, len(s.Elems))
			for _, obj := range s.Elems {
				m[obj.GetName()] = obj
			}
			s.large = m
		}
		s.large[name] = obj
	}
	return nil
}
