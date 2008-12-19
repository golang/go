// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Globals


// The following types should really be in their respective files
// (object.go, type.go, scope.go, package.go, compilation.go, etc.) but
// they refer to each other and we don't know how to handle forward
// declared pointers across packages yet.


// ----------------------------------------------------------------------------

type Type struct
type Scope struct
type Elem struct
type OldCompilation struct

// Object represents a language object, such as a constant, variable, type,
// etc. (kind). An objects is (pre-)declared at a particular position in the
// source code (pos), has a name (ident), a type (typ), and a package number
// or nesting level (pnolev).

export type Object struct {
	exported bool;
	pos int;  // source position (< 0 if unknown position)
	kind int;
	ident string;
	typ *Type;  // nil for packages
	pnolev int;  // >= 0: package no., <= 0: function nesting level, 0: global level
}


export type Type struct {
	ref int;  // for exporting only: >= 0 means already exported
	form int;
	size int;  // in bytes
	len int;  // array length, no. of function/method parameters (w/o recv)
	aux int;  // channel info
	obj *Object;  // primary type object or NULL
	key *Type;  // alias base type or map key
	elt *Type;  // aliased type, array, map, channel or pointer element type, function result type, tuple function type
	scope *Scope;  // forwards, structs, interfaces, functions
}


export type Package struct {
	ref int;  // for exporting only: >= 0 means already exported
	file_name string;
	key string;
	obj *Object;
	scope *Scope;  // holds the (global) objects in this package
}


export type Scope struct {
	parent *Scope;
	entries map[string] *Object;
}


export type Environment struct {
	Error *(comp *OldCompilation, pos int, msg string);
	Import *(comp *OldCompilation, pkg_file string) *Package;
	Export *(comp *OldCompilation, pkg_file string);
	Compile *(comp *OldCompilation, src_file string);
}


export type OldCompilation struct {
	// environment
	env *Environment;

	// TODO rethink the need for this here
	src_file string;
	src string;

	// Error handling
	nerrors int;  // number of errors reported
	errpos int;  // last error position

	// TODO use open arrays eventually
	pkg_list [256] *Package;  // pkg_list[0] is the current package
	pkg_ref int;
}


export type Expr interface {
	op() int;  // node operation
	pos() int;  // source position
	typ() *Type;
	// ... more to come here
}


export type Stat interface {
	// ... more to come here
}


// TODO This is hideous! We need to have a decent way to do lists.
// Ideally open arrays that allow '+'.

export type Elem struct {
	next *Elem;
	val int;
	str string;
	obj *Object;
	typ *Type;
	expr Expr
}


// ----------------------------------------------------------------------------
// Creation

export var Universe_void_typ *Type  // initialized by Universe to Universe.void_typ

export func NewObject(pos, kind int, ident string) *Object {
	obj := new(*Object);
	obj.exported = false;
	obj.pos = pos;
	obj.kind = kind;
	obj.ident = ident;
	obj.typ = Universe_void_typ;
	obj.pnolev = 0;
	return obj;
}


export func NewType(form int) *Type {
	typ := new(*Type);
	typ.ref = -1;  // not yet exported
	typ.form = form;
	return typ;
}


export func NewPackage(file_name string, obj *Object, scope *Scope) *Package {
	pkg := new(*Package);
	pkg.ref = -1;  // not yet exported
	pkg.file_name = file_name;
	pkg.key = "<the package key>";  // empty key means package forward declaration
	pkg.obj = obj;
	pkg.scope = scope;
	return pkg;
}


export func NewScope(parent *Scope) *Scope {
	scope := new(*Scope);
	scope.parent = parent;
	scope.entries = new(map[string]*Object, 8);
	return scope;
}


// ----------------------------------------------------------------------------
// Object methods

func (obj *Object) Copy() *Object {
	copy := new(*Object);
	copy.exported = obj.exported;
	copy.pos = obj.pos;
	copy.kind = obj.kind;
	copy.ident = obj.ident;
	copy.typ = obj.typ;
	copy.pnolev = obj.pnolev;
	return copy;
}


// ----------------------------------------------------------------------------
// Scope methods

func (scope *Scope) Lookup(ident string) *Object {
	obj, found := scope.entries[ident];
	if found {
		return obj;
	}
	return nil;
}


func (scope *Scope) Add(obj* Object) {
	scope.entries[obj.ident] = obj;
}


func (scope *Scope) Insert(obj *Object) {
	if scope.Lookup(obj.ident) != nil {
		panic("obj already inserted");
	}
	scope.Add(obj);
}


func (scope *Scope) InsertImport(obj *Object) *Object {
	 p := scope.Lookup(obj.ident);
	 if p == nil {
		scope.Add(obj);
		p = obj;
	 }
	 return p;
}


func (scope *Scope) Print() {
	print("scope {");
	for key := range scope.entries {
		print("\n  ", key);
	}
	print("\n}\n");
}


// ----------------------------------------------------------------------------
// Compilation methods

func (C *OldCompilation) Lookup(file_name string) *Package {
	for i := 0; i < C.pkg_ref; i++ {
		pkg := C.pkg_list[i];
		if pkg.file_name == file_name {
			return pkg;
		}
	}
	return nil;
}


func (C *OldCompilation) Insert(pkg *Package) {
	if C.Lookup(pkg.file_name) != nil {
		panic("package already inserted");
	}
	pkg.obj.pnolev = C.pkg_ref;
	C.pkg_list[C.pkg_ref] = pkg;
	C.pkg_ref++;
}
