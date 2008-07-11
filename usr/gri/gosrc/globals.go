// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Globals


// The following types should really be in their respective files
// object.go, type.go, and scope.go but they refer to each other
// and we don't know how to handle forward-declared pointers across
// packages yet.


// ----------------------------------------------------------------------------

export Object
type Object struct {
	mark bool;  // mark => object marked for export
	kind int;
	name string;
	type_ *Type;
	pnolev int;  // >= 0: package no., <= 0: level, 0: global level of compilation
}


// ----------------------------------------------------------------------------

export Type
type Type struct {
	ref int;  // for exporting only: >= 0 means already exported
	form int;
	flags int;  // channels, functions
	size int;  // in bytes
	len_ int;  // array length, no. of parameters (w/o recv)
	obj *Object;  // primary type object or NULL
	key *Object;  // maps
	elt *Object;  // arrays, maps, channels, pointers, references
	scope *Scope;  // incomplete types, structs, interfaces, functions, packages
}


// ----------------------------------------------------------------------------

export Scope
type Scope struct {
	parent *Scope;
	// list ObjList
	
}


/*
func (scope *Scope) Lookup(ident string) *Object {
	panic "UNIMPLEMENTED";
	return nil;
}


func (scope *Scope) Insert(obj *Object) {
	panic "UNIMPLEMENTED";
}


func (scope *Scope) InsertImport(obj *Object) *Object {
	panic "UNIMPLEMENTED";
	return nil;
}
*/
