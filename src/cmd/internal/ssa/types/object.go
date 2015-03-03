// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package is a drop-in replacement for go/types
// for use until go/types is included in the main repo.

package types

// An Object describes a named language entity such as a package,
// constant, type, variable, function (incl. methods), or label.
// All objects implement the Object interface.
//
type Object interface {
	Name() string // package local object name
	Type() Type   // object type
}

// An object implements the common parts of an Object.
type object struct {
	name string
	typ  Type
}

func (obj *object) Name() string { return obj.name }
func (obj *object) Type() Type   { return obj.typ }

// A Variable represents a declared variable (including function parameters and results, and struct fields).
type Var struct {
	object
	anonymous bool // if set, the variable is an anonymous struct field, and name is the type name
	visited   bool // for initialization cycle detection
	isField   bool // var is struct field
	used      bool // set if the variable was used
}

func NewParam(pos int, pkg *int, name string, typ Type) *Var {
	return &Var{object: object{name, typ}, used: true} // parameters are always 'used'
}
