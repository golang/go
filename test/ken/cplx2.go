// [ $GOARCH != amd64 ] || ($G $D/$F.go && $L $F.$A && ./$A.out)

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	R = 5
	I = 6i

	C1 = R + I    // ADD(5,6)
	C2 = R - I    // SUB(5,-6)
	C3 = -(R + I) // ADD(5,6) NEG(-5,-6)
	C4 = -(R - I) // SUB(5,-6) NEG(-5,6)

	C5 = C1 + R // ADD(10,6)
	C6 = C1 + I // ADD(5,12)

	Ca = C5 + C6 // ADD(15,18)
	Cb = C5 - C6 // SUB(5,-6)

	Cc = C5 * C6 // MUL(-22,-150)
	Cd = C5 / C6 // DIV(0.721893,-0.532544)
	Ce = Cd * C6 // MUL(10,6) sb C5
)

func main() {

	r := 5 + 0i
	if r != R {
		panicln("opcode 1", r, R)
	}

	i := 6i
	if i != I {
		panicln("opcode 2", i, I)
	}

	c1 := r + i
	if c1 != C1 {
		panicln("opcode x", c1, C1)
	}

	c2 := r - i
	if c2 != C2 {
		panicln("opcode x", c2, C2)
	}

	c3 := -(r + i)
	if c3 != C3 {
		panicln("opcode x", c3, C3)
	}

	c4 := -(r - i)
	if c4 != C4 {
		panicln("opcode x", c4, C4)
	}

	c5 := c1 + r
	if c5 != C5 {
		panicln("opcode x", c5, C5)
	}

	c6 := c1 + i
	if c6 != C6 {
		panicln("opcode x", c6, C6)
	}

	ca := c5 + c6
	if ca != Ca {
		panicln("opcode x", ca, Ca)
	}

	cb := c5 - c6
	if cb != Cb {
		panicln("opcode x", cb, Cb)
	}

	cc := c5 * c6
	if cc != Cc {
		panicln("opcode x", cc, Cc)
	}

	cd := c5 / c6
	if cd != Cd {
		panicln("opcode x", cd, Cd)
	}

	ce := cd * c6
	if ce != Ce {
		panicln("opcode x", ce, Ce)
	}
}
