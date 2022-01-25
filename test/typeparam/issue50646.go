// run -gcflags=-G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func eql[P comparable](x, y P) {
	if x != y {
		panic("not equal")
	}
}

func expectPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("function succeeded unexpectedly")
		}
	}()
	f()
}

func main() {
	eql[int](1, 1)
	eql(1, 1)

	// all interfaces implement comparable
	var x, y any = 2, 2
	eql[any](x, y)
	eql(x, y)

	// but we may get runtime panics
	x, y = 1, 2 // x != y
	expectPanic(func() { eql(x, y) })

	x, y = main, main // functions are not comparable
	expectPanic(func() { eql(x, y) })
}
