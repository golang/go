// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors. All rights reserved.
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

	var r complex64 = 5 + 0i
	if r != R {
		println("opcode 1", r, R)
		panic("fail")
	}

	var i complex64 = 6i
	if i != I {
		println("opcode 2", i, I)
		panic("fail")
	}

	c1 := r + i
	if c1 != C1 {
		println("opcode x", c1, C1)
		panic("fail")
	}

	c2 := r - i
	if c2 != C2 {
		println("opcode x", c2, C2)
		panic("fail")
	}

	c3 := -(r + i)
	if c3 != C3 {
		println("opcode x", c3, C3)
		panic("fail")
	}

	c4 := -(r - i)
	if c4 != C4 {
		println("opcode x", c4, C4)
		panic("fail")
	}

	c5 := c1 + r
	if c5 != C5 {
		println("opcode x", c5, C5)
		panic("fail")
	}

	c6 := c1 + i
	if c6 != C6 {
		println("opcode x", c6, C6)
		panic("fail")
	}

	ca := c5 + c6
	if ca != Ca {
		println("opcode x", ca, Ca)
		panic("fail")
	}

	cb := c5 - c6
	if cb != Cb {
		println("opcode x", cb, Cb)
		panic("fail")
	}

	cc := c5 * c6
	if cc != Cc {
		println("opcode x", cc, Cc)
		panic("fail")
	}

	cd := c5 / c6
	if cd != Cd {
		println("opcode x", cd, Cd)
		panic("fail")
	}

	ce := cd * c6
	if ce != Ce {
		println("opcode x", ce, Ce)
		panic("fail")
	}
	
	r32 := real(complex64(ce))
	if r32 != float32(real(Ce)) {
		println("real(complex64(ce))", r32, real(Ce))
		panic("fail")
	}
	
	r64 := real(complex128(ce))
	if r64 != real(Ce) {
		println("real(complex128(ce))", r64, real(Ce))
		panic("fail")
	}
}
