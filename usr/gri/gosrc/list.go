// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package List

import Globals "globals"  // because of 6g warning
import Object "object"
import Type "type"


// TODO This is hideous! We need to have a decent way to do lists.
// Ideally open arrays that allow '+'.

type Elem struct {
	next *Elem;
	str string;
	obj *Object.Object;
	typ *Type.Type;
}


export List
type List struct {
	len_ int;
	first, last *Elem;
};


export NewList
func NewList() *List {
	return new(List);
}


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
	}
	L.last.next = e;
	L.last = e;
}


func (L *List) StrAt(i int) string {
	return L.at(i).str;
}


func (L *List) ObjAt(i int) *Object.Object {
	return L.at(i).obj;
}


func (L *List) TypAt(i int) *Type.Type {
	return L.at(i).typ;
}


func (L *List) AddStr(str string) {
	L.Add().str = str;
}


func (L *List) AddObj(obj *Object.Object) {
	L.Add().obj = obj;
}


func (L *List) AddTyp(typ *Type.Type) {
	L.Add().typ = typ;
}
