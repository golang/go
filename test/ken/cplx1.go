// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test simple arithmetic and assignment for complex numbers.

package main

const (
	R = 5
	I = 6i

	C1 = R + I // ADD(5,6)
)

func main() {
	var b bool

	// constants
	b = (5 + 6i) == C1
	if !b {
		println("const bool 1", b)
		panic("fail")
	}

	b = (5 + 6i) != C1
	if b {
		println("const bool 2", b)
		panic("fail")
	}

	b = C1 == (5 + 6i)
	if !b {
		println("const bool 3", b)
		panic("fail")
	}

	b = C1 != (5 + 6i)
	if b {
		println("const bool 4", b)
		panic("fail")
	}

	// vars passed through parameters
	booltest(5+6i, true)
	booltest(5+7i, false)
	booltest(6+6i, false)
	booltest(6+9i, false)
}

func booltest(a complex64, r bool) {
	var b bool

	b = a == C1
	if b != r {
		println("param bool 1", a, b, r)
		panic("fail")
	}

	b = a != C1
	if b == r {
		println("param bool 2", a, b, r)
		panic("fail")
	}

	b = C1 == a
	if b != r {
		println("param bool 3", a, b, r)
		panic("fail")
	}

	b = C1 != a
	if b == r {
		println("param bool 4", a, b, r)
		panic("fail")
	}

	if r {
		if a != C1 {
			println("param bool 5", a, b, r)
			panic("fail")
		}
		if C1 != a {
			println("param bool 6", a, b, r)
			panic("fail")
		}
	} else {
		if a == C1 {
			println("param bool 6", a, b, r)
			panic("fail")
		}
		if C1 == a {
			println("param bool 7", a, b, r)
			panic("fail")
		}
	}
}
