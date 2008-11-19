// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "array"


type S struct {
	val int
}


func (p *S) Init(val int) *S {
	p.val = val;
	return p;
}


func test0() {
	v := array.New(0);
	if v.Len() != 0 {
		panic("len = ", v.Len(), "\n");
	}
}


func test1() {
	var a [1000] *S;
	for i := 0; i < len(a); i++ {
		a[i] = new(S).Init(i);
	}

	v := array.New(0);
	for i := 0; i < len(a); i++ {
		v.Insert(0, a[i]);
		if v.Len() != i + 1 {
			panic("len = ", v.Len(), "\n");
		}
	}

	for i := 0; i < v.Len(); i++ {
		x := convert(*S, v.At(i));
		if x.val != v.Len() - i - 1 {
			panic("expected ", i, ", found ", x.val, "\n");
		}
	}
	
	for v.Len() > 10 {
		v.Remove(10);
	}
}


func main() {
	test0();
	test1();
}
