// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

export type Element interface {
}

export type Vector struct {
	elem []Element;
}

func (v *Vector) At(i int) Element {
	return v.elem[i];
}

export type TStruct struct {
	name string;
	fields *Vector;
}

func (s *TStruct) field(i int) *TStruct {
	// works if we say
	//	t := s.fields.At(i);
	//	return t;
	return s.fields.At(i);
}

func main() {
	v := new(Vector);
	v.elem = make([]Element, 10);
	t := new(TStruct);
	t.name = "hi";
	v.elem[0] = t;
	s := new(TStruct);
	s.name = "foo";
	s.fields = v;
	if s.field(0).name != "hi" {
		panic("bad name")
	}
}
