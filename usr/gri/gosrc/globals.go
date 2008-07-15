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
	ident string;
	typ *Type;
	pnolev int;  // >= 0: package no., <= 0: level, 0: global level of compilation
}


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


// TODO This is hideous! We need to have a decent way to do lists.
// Ideally open arrays that allow '+'.

type Elem struct {
	next *Elem;
	str string;
	obj *Object;
	typ *Type;
}


export List
type List struct {
	len_ int;
	first, last *Elem;
};


export Scope
type Scope struct {
	parent *Scope;
	entries *List;
	// entries *map[string] *Object;  // doesn't work yet
}


// ----------------------------------------------------------------------------
// Creation

export NewObject
func NewObject(kind int, ident string) *Object {
	obj := new(Object);
	obj.mark = false;
	obj.kind = kind;
	obj.ident = ident;
	obj.typ = nil;  // Universe::undef_t;
	obj.pnolev = 0;
	return obj;
}


export NewType
func NewType(form int) *Type {
	typ := new(Type);
	typ.form = form;
	return typ;
}


export NewList
func NewList() *List {
	return new(List);
}


export NewScope
func NewScope(parent *Scope) *Scope {
	scope := new(Scope);
	scope.parent = parent;
	scope.entries = NewList();
	return scope;
}


// ----------------------------------------------------------------------------
// List methods

func (L* List) len_() int {
	return L.len_;
}


func (L *List) at(i int) *Elem {
	if i < 0 || L.len_ <= i {
		panic "index out of bounds";
	}

	p := L.first;
	for ; i > 0; i-- {
		p = p.next;
	}
	
	return p;
}


func (L *List) Add() *Elem {
	L.len_++;
	e := new(Elem);
	if L.first == nil {
		L.first = e;
	} else {
		L.last.next = e;
	}
	L.last = e;
	return e;
}


func (L *List) StrAt(i int) string {
	return L.at(i).str;
}


func (L *List) ObjAt(i int) *Object {
	return L.at(i).obj;
}


func (L *List) TypAt(i int) *Type {
	return L.at(i).typ;
}


func (L *List) AddStr(str string) {
	L.Add().str = str;
}


func (L *List) AddObj(obj *Object) {
	L.Add().obj = obj;
}


func (L *List) AddTyp(typ *Type) {
	L.Add().typ = typ;
}


// ----------------------------------------------------------------------------
// Scope methods

func (scope *Scope) Lookup(ident string) *Object {
	var p *Elem;
	for p = scope.entries.first; p != nil; p = p.next {
		if p.obj.ident == ident {
			return p.obj;
		}
	}
	return nil;
}


func (scope *Scope) Insert(obj *Object) {
	if scope.Lookup(obj.ident) != nil {
		panic "obj already inserted";
	}
	scope.entries.AddObj(obj);
}


func (scope *Scope) InsertImport(obj *Object) *Object {
	 p := scope.Lookup(obj.ident);
	 if p == nil {
		scope.Insert(obj);
		p = obj;
	 }
	 return p;
}


func (scope *Scope) Print() {
	print "scope {";
	var p* Elem;
	for p = scope.entries.first; p != nil; p = p.next {
		print "\n  ", p.obj.ident;
	}
	print "\n}\n";
}
