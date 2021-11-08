// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f[T any](i interface{foo()}) {
	switch i.(type) {
	case interface{bar() T}:
		println("barT")
	case myint:
		println("myint")
	case myfloat:
		println("myfloat")
	default:
		println("other")
	}
}

type myint int
func (myint) foo() {
}
func (x myint) bar() int {
	return int(x)
}

type myfloat float64
func (myfloat) foo() {
}

func main() {
	f[int](nil)
	f[int](myint(6))
	f[int](myfloat(7))
}
