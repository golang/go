// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
		panicln("const bool 1", b)
	}

	b = (5 + 6i) != C1
	if b {
		panicln("const bool 2", b)
	}

	b = C1 == (5 + 6i)
	if !b {
		panicln("const bool 3", b)
	}

	b = C1 != (5 + 6i)
	if b {
		panicln("const bool 4", b)
	}

	// vars passed through parameters
	booltest(5+6i, true)
	booltest(5+7i, false)
	booltest(6+6i, false)
	booltest(6+9i, false)
}

func booltest(a complex, r bool) {
	var b bool

	b = a == C1
	if b != r {
		panicln("param bool 1", a, b, r)
	}

	b = a != C1
	if b == r {
		panicln("param bool 2", a, b, r)
	}

	b = C1 == a
	if b != r {
		panicln("param bool 3", a, b, r)
	}

	b = C1 != a
	if b == r {
		panicln("param bool 4", a, b, r)
	}

	if r {
		if a != C1 {
			panicln("param bool 5", a, b, r)
		}
		if C1 != a {
			panicln("param bool 6", a, b, r)
		}
	} else {
		if a == C1 {
			panicln("param bool 6", a, b, r)
		}
		if C1 == a {
			panicln("param bool 7", a, b, r)
		}
	}
}
