// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test Go2 literal syntax for basic types.
// Avoid running gofmt on this file to preserve the
// test cases with upper-case prefixes (0B, 0O, 0X).

package main

import "fmt"

func assert(cond bool) {
	if !cond {
		panic("assertion failed")
	}
}

func equal(x, y interface{}) bool {
	if x != y {
		fmt.Printf("%g != %g\n", x, y)
		return false
	}
	return true
}

func main() {
	// 0-octals
	assert(0_1 == 01)
	assert(012 == 012)
	assert(0_1_2 == 012)
	assert(0_1_2i == complex(0, 12)) // decimal digits despite leading 0 for backward-compatibility
	assert(00089i == complex(0, 89)) // decimal digits despite leading 0 for backward-compatibility

	// decimals
	assert(1_000_000 == 1000000)
	assert(1_000i == complex(0, 1000))

	// hexadecimals
	assert(0x_1 == 0x1)
	assert(0x1_2 == 0x12)
	assert(0x_cafe_f00d == 0xcafef00d)
	assert(0x_cafei == complex(0, 0xcafe))

	// octals
	assert(0o_1 == 01)
	assert(0o12 == 012)
	assert(0o_1_2 == 012)
	assert(0o_1_2i == complex(0, 0o12))

	// binaries
	assert(0b_1 == 1)
	assert(0b10 == 2)
	assert(0b_1_0 == 2)
	assert(0b_1_0i == complex(0, 2))

	// decimal floats
	assert(0. == 0.0)
	assert(.0 == 0.0)
	assert(1_0. == 10.0)
	assert(.0_1 == 0.01)
	assert(1_0.0_1 == 10.01)
	assert(1_0.0_1i == complex(0, 10.01))

	assert(0.e1_0 == 0.0e10)
	assert(.0e1_0 == 0.0e10)
	assert(1_0.e1_0 == 10.0e10)
	assert(.0_1e1_0 == 0.01e10)
	assert(1_0.0_1e1_0 == 10.01e10)
	assert(1_0.0_1e1_0i == complex(0, 10.01e10))

	// hexadecimal floats
	assert(equal(0x1p-2, 0.25))
	assert(equal(0x2.p10, 2048.0))
	assert(equal(0x1.Fp+0, 1.9375))
	assert(equal(0x.8p-0, 0.5))
	assert(equal(0x1FFFp-16, 0.1249847412109375))
	assert(equal(0x1.fffffffffffffp1023, 1.7976931348623157e308))
	assert(equal(0x1.fffffffffffffp1023i, complex(0, 1.7976931348623157e308)))

	assert(equal(0x_1p-2, 0.25))
	assert(equal(0x2.p1_0, 2048.0))
	assert(equal(0x1_0.Fp+0, 16.9375))
	assert(equal(0x_0.8p-0, 0.5))
	assert(equal(0x_1FF_Fp-16, 0.1249847412109375))
	assert(equal(0x1.f_ffff_ffff_ffffp1_023, 1.7976931348623157e308))
	assert(equal(0x1.f_ffff_ffff_ffffp1_023i, complex(0, 1.7976931348623157e308)))
}
