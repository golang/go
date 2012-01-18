// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

const (
	R = 5
	I = 6i

	C1 = R + I // ADD(5,6)
)

func want(s, w string) {
	if s != w {
		panic(s + " != " + w)
	}
}

func doprint(c complex128, w string) {
	s := fmt.Sprintf("%f", c)
	want(s, w)
}

func main() {

	// constants
	s := fmt.Sprintf("%f", -C1)
	want(s, "(-5.000000-6.000000i)")
	doprint(C1, "(5.000000+6.000000i)")

	// variables
	c1 := C1
	s = fmt.Sprintf("%f", c1)
	want(s, "(5.000000+6.000000i)")
	doprint(c1, "(5.000000+6.000000i)")

	// 128
	c2 := complex128(C1)
	s = fmt.Sprintf("%G", c2)
	want(s, "(5+6i)")

	// real, imag, complex
	c3 := complex(real(c2)+3, imag(c2)-5) + c2
	s = fmt.Sprintf("%G", c3)
	want(s, "(13+7i)")

	// compiler used to crash on nested divide
	c4 := complex(real(c3/2), imag(c3/2))
	if c4 != c3/2 {
		fmt.Printf("BUG: c3 = %G != c4 = %G\n", c3, c4)
		panic(0)
	}
}
