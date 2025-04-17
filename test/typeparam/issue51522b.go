// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f[T comparable](i any) {
	var t T

	switch i {
	case t:
		// ok
	default:
		println("FAIL: switch i")
	}

	switch t {
	case i:
		// ok
	default:
		println("FAIL: switch t")
	}
}

type myint int

func (m myint) foo() {
}

type fooer interface {
	foo()
}

type comparableFoo interface {
	comparable
	foo()
}

func g[T comparableFoo](i fooer) {
	var t T

	switch i {
	case t:
		// ok
	default:
		println("FAIL: switch i")
	}

	switch t {
	case i:
		// ok
	default:
		println("FAIL: switch t")
	}
}

func main() {
	f[int](0)
	g[myint](myint(0))
}
