// cmpout

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test trivial, bootstrap-level complex numbers, including printing.

package main

const (
	R = 5
	I = 6i

	C1 = R + I // ADD(5,6)
)

func doprint(c complex128) { println(c) }

func main() {

	// constants
	println(C1)
	doprint(C1)

	// variables
	c1 := C1
	println(c1)
	doprint(c1)
}
