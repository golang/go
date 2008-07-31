// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Globals


// The following types should really be in their respective files
// (object.go, type.go, scope.go, package.go, compilation.go, etc.) but
// they refer to each other and we don't know how to handle forward
// declared pointers across packages yet.


// ----------------------------------------------------------------------------

export Object
type Object struct {
	exported bool;
	pos int;  // source position
	kind int;
	ident string;
	typ *Type;
	pnolev int;  // >= 0: package no., <= 0: level, 0: global level of compilation
	scope *Scope;  // which contains the object
}


export Type
type Type struct {
	ref int;  // for exporting only: >= 0 means already exported
	form int;
	flags int;  // channels, functions
	size int;  // in bytes
	len_ int;  // array length, no. of parameters (w/o recv)
	obj *Object;  // primary type object or NULL
	key *Type;  // maps
	elt *Type;  // aliases, arrays, maps, channels, pointers
	scope *Scope;  // structs, interfaces, functions
}


export Package
type Package struct {
	ref int;  // for exporting only: >= 0 means already exported
	file_name string;
	key string;
	obj *Object;
	scope *Scope;
}


// TODO This is hideous! We need to have a decent way to do lists.
// Ideally open arrays that allow '+'.

type Elem struct {
	next *Elem;
	val int;
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
	// entries *map[string] *Object;  // doesn't work properly
}


export Flags;
type Flags struct {
	debug bool;
	print_export bool;
	semantic_checks bool;
	verbose int;
	sixg bool;
}


export Compilation
type Compilation struct {
	flags *Flags;
	// TODO use open arrays eventually
	pkgs [256] *Package;  // pkgs[0] is the current package
	npkgs int;
}


export Expr
type Expr interface {
	typ() *Type;
	// ... more to come here
}


export Stat
type Stat interface {
	// ... more to come here
}


// ----------------------------------------------------------------------------
// Creation

export Universe_undef_t
var Universe_undef_t *Type  // initialized by Universe to Universe.undef_t

export NewObject
func NewObject(pos, kind int, ident string) *Object {
	obj := new(Object);
	obj.exported = false;
	obj.pos = pos;
	obj.kind = kind;
	obj.ident = ident;
	obj.typ = Universe_undef_t;
	obj.pnolev = 0;
	obj.scope = nil;
	return obj;
}


export NewType
func NewType(form int) *Type {
	typ := new(Type);
	typ.ref = -1;  // not yet exported
	typ.form = form;
	return typ;
}


export NewPackage;
func NewPackage(file_name string) *Package {
	pkg := new(Package);
	pkg.ref = -1;  // not yet exported
	pkg.file_name = file_name;
	pkg.key = "<the package key>";  // TODO fix this
	return pkg;
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


export NewCompilation;
func NewCompilation(flags *Flags) *Compilation {
	comp := new(Compilation);
	comp.flags = flags;
	return comp;
}


// ----------------------------------------------------------------------------
// Object methods

func (obj *Object) Copy() *Object {
	copy := new(Object);
	copy.exported = obj.exported;
	copy.pos = obj.pos;
	copy.kind = obj.kind;
	copy.ident = obj.ident;
	copy.typ = obj.typ;
	copy.pnolev = obj.pnolev;
	copy.scope = nil;  // cannot be in the same scope (same ident!)
	return copy;
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


func (L *List) Clear() {
	L.len_, L.first, L.last = 0, nil, nil;
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


func (L *List) IntAt(i int) int {
	return L.at(i).val;
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


func (L *List) AddInt(val int) {
	L.Add().val = val;
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
			if p.obj.scope != scope {
				panic "incorrect scope for object";
			}
			return p.obj;
		}
	}
	return nil;
}


func (scope *Scope) Insert(obj *Object) {
	if scope.Lookup(obj.ident) != nil {
		panic "obj already inserted";
	}
	if obj.scope != nil {
		panic "obj already in a scope";
	}
	scope.entries.AddObj(obj);
	obj.scope = scope;
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


// ----------------------------------------------------------------------------
// Compilation methods

func (C *Compilation) Lookup(file_name string) *Package {
	for i := 0; i < C.npkgs; i++ {
		pkg := C.pkgs[i];
		if pkg.file_name == file_name {
			return pkg;
		}
	}
	return nil;
}


func (C *Compilation) Insert(pkg *Package) {
	if C.Lookup(pkg.file_name) != nil {
		panic "package already inserted";
	}
	pkg.obj.pnolev = C.npkgs;
	C.pkgs[C.npkgs] = pkg;
	C.npkgs++;
}


func (C *Compilation) InsertImport(pkg *Package) *Package {
	p := C.Lookup(pkg.file_name);
	if (p == nil) {
		// no primary package found
		C.Insert(pkg);
		p = pkg;
	}
	return p;
}
