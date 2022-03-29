// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For #45062, miscompilation of open defer of method invocation

package main

func main() {
	var x, y, z int = -1, -2, -3
	F(x, y, z)
}

//go:noinline
func F(x, y, z int) {
	defer i.M(x, y, z)
	defer func() { recover() }()
	panic("XXX")
}

type T int

func (t *T) M(x, y, z int) {
	if x == -1 && y == -2 && z == -3 {
		return
	}
	println("FAIL: Expected -1, -2, -3, but x, y, z =", x, y, z)
}

var t T = 42

type I interface{ M(x, y, z int) }

var i I = &t
