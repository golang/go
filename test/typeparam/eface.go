// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we handle instantiated empty interfaces.

package main

type E[T any] interface {
}

//go:noinline
func f[T any](x E[T]) interface{} {
	return x
}

//go:noinline
func g[T any](x interface{}) E[T] {
	return x
}

type I[T any] interface {
	foo()
}

type myint int

func (x myint) foo() {}

//go:noinline
func h[T any](x I[T]) interface{ foo() } {
	return x
}

//go:noinline
func i[T any](x interface{ foo() }) I[T] {
	return x
}

func main() {
	if f[int](1) != 1 {
		println("test 1 failed")
	}
	if f[int](2) != (interface{})(2) {
		println("test 2 failed")
	}
	if g[int](3) != 3 {
		println("test 3 failed")
	}
	if g[int](4) != (E[int])(4) {
		println("test 4 failed")
	}
	if h[int](myint(5)) != myint(5) {
		println("test 5 failed")
	}
	if h[int](myint(6)) != interface{ foo() }(myint(6)) {
		println("test 6 failed")
	}
	if i[int](myint(7)) != myint(7) {
		println("test 7 failed")
	}
	if i[int](myint(8)) != I[int](myint(8)) {
		println("test 8 failed")
	}
}
