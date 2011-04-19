// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PACKAGE UNDER CONSTRUCTION. ANY AND ALL PARTS MAY CHANGE.
// Package types declares the types used to represent Go types.
//
package types

import "go/ast"


// All types implement the Type interface.
type Type interface {
	isType()
}


// All concrete types embed ImplementsType which
// ensures that all types implement the Type interface.
type ImplementsType struct{}

func (t *ImplementsType) isType() {}


// A Basic represents a (unnamed) basic type.
type Basic struct {
	ImplementsType
	// TODO(gri) need a field specifying the exact basic type
}


// An Array represents an array type [Len]Elt.
type Array struct {
	ImplementsType
	Len uint64
	Elt Type
}


// A Slice represents a slice type []Elt.
type Slice struct {
	ImplementsType
	Elt Type
}


// A Struct represents a struct type struct{...}.
type Struct struct {
	ImplementsType
	// TODO(gri) need to remember fields.
}


// A Pointer represents a pointer type *Base.
type Pointer struct {
	ImplementsType
	Base Type
}


// A Func represents a function type func(...) (...).
type Func struct {
	ImplementsType
	IsVariadic bool
	// TODO(gri) need to remember parameters.
}


// An Interface represents an interface type interface{...}.
type Interface struct {
	ImplementsType
	// TODO(gri) need to remember methods.
}


// A Map represents a map type map[Key]Elt.
type Map struct {
	ImplementsType
	Key, Elt Type
}


// A Chan represents a channel type chan Elt, <-chan Elt, or chan<-Elt.
type Chan struct {
	ImplementsType
	Dir ast.ChanDir
	Elt Type
}


// A Name represents a named type as declared in a type declaration.
type Name struct {
	ImplementsType
	Underlying Type        // nil if not fully declared
	Obj        *ast.Object // corresponding declared object
	// TODO(gri) need to remember fields and methods.
}


// If typ is a pointer type, Deref returns the pointer's base type;
// otherwise it returns typ.
func Deref(typ Type) Type {
	if typ, ok := typ.(*Pointer); ok {
		return typ.Base
	}
	return typ
}


// Underlying returns the underlying type of a type.
func Underlying(typ Type) Type {
	if typ, ok := typ.(*Name); ok {
		utyp := typ.Underlying
		if _, ok := utyp.(*Basic); ok {
			return typ
		}
		return utyp

	}
	return typ
}
