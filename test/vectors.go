// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "container/vector"


type S struct {
	val int
}


func (p *S) Init(val int) *S {
	p.val = val
	return p
}


func test0() {
	v := new(vector.Vector)
	if v.Len() != 0 {
		print("len = ", v.Len(), "\n")
		panic("fail")
	}
}


func test1() {
	var a [1000]*S
	for i := 0; i < len(a); i++ {
		a[i] = new(S).Init(i)
	}

	v := new(vector.Vector)
	for i := 0; i < len(a); i++ {
		v.Insert(0, a[i])
		if v.Len() != i+1 {
			print("len = ", v.Len(), "\n")
			panic("fail")
		}
	}

	for i := 0; i < v.Len(); i++ {
		x := v.At(i).(*S)
		if x.val != v.Len()-i-1 {
			print("expected ", i, ", found ", x.val, "\n")
			panic("fail")
		}
	}

	for v.Len() > 10 {
		v.Delete(10)
	}
}


func main() {
	test0()
	test1()
}
