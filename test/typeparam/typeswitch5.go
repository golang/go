// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type myint int
func (x myint) foo() int {return int(x)}

type myfloat float64
func (x myfloat) foo() float64 {return float64(x) }

func f[T any](i interface{}) {
	switch x := i.(type) {
	case interface { foo() T }:
		println("fooer", x.foo())
	default:
		println("other")
	}
}
func main() {
	f[int](myint(6))
	f[int](myfloat(7))
	f[float64](myint(8))
	f[float64](myfloat(9))
}
