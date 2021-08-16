// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface{}

type _S[T any] struct {
	x *T
}

// F is a non-generic function, but has a type _S[I] which is instantiated from a
// generic type. Test that _S[I] is successfully exported.
func F() {
	v := _S[I]{}
	if v.x != nil {
		panic(v)
	}
}

// Testing the various combinations of method expressions.
type S1 struct{}

func (*S1) M() {}

type S2 struct{}

func (S2) M() {}

func _F1[T interface{ M() }](t T) {
	_ = T.M
}

func F2() {
	_F1(&S1{})
	_F1(S2{})
	_F1(&S2{})
}

func main() {
	F()
	F2()
}
