// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Expects to see error messages on 'p' exponents.

package main

import "fmt"

const (
	x1 = 1.1    // float
	x2 = 1e10   // float
	x3 = 0x1e10 // integer (e is a hex digit)
)

// 'p' exponents are invalid - the 'p' is not considered
// part of a floating-point number, but introduces a new
// (unexpected) name.
//
// Error recovery is not ideal and we use a new declaration
// each time for the parser to recover.

const x4 = 0x1p10 // ERROR "unexpected p10"
const x5 = 1p10   // ERROR "unexpected p10"
const x6 = 0p0    // ERROR "unexpected p0"

func main() {
	fmt.Printf("%g %T\n", x1, x1)
	fmt.Printf("%g %T\n", x2, x2)
	fmt.Printf("%g %T\n", x3, x3)
	fmt.Printf("%g %T\n", x4, x4)
	fmt.Printf("%g %T\n", x5, x5)
	fmt.Printf("%g %T\n", x6, x6)
}
