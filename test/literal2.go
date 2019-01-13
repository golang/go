// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test Go2 literal syntax for basic types.
// TODO add more tests

package main

import "fmt"

func assert(cond bool) {
	if !cond {
		panic("assertion failed")
	}
}

func equal(x, y float64) bool {
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

	// decimals
	assert(1_000_000 == 1000000)

	// hexadecimals
	assert(0x_1 == 0x1)
	assert(0x1_2 == 0x12)
	assert(0X_cafe_f00d == 0xcafef00d)

	// octals
	assert(0o_1 == 01)
	assert(0o12 == 012)
	assert(0O_1_2 == 012)

	// binaries
	assert(0b_1 == 1)
	assert(0b10 == 2)
	assert(0b_1_0 == 2)

	// decimal floats
	assert(0. == 0.0)
	assert(.0 == 0.0)
	assert(1_0. == 10.0)
	assert(.0_1 == 0.01)
	assert(1_0.0_1 == 10.01)

	assert(0.e1_0 == 0.0e10)
	assert(.0e1_0 == 0.0e10)
	assert(1_0.e1_0 == 10.0e10)
	assert(.0_1e1_0 == 0.01e10)
	assert(1_0.0_1e1_0 == 10.01e10)

	// hexadecimal floats
	assert(equal(0x1p-2, 0.25))
	assert(equal(0x2.p10, 2048.0))
	assert(equal(0x1.Fp+0, 1.9375))
	assert(equal(0X.8p-0, 0.5))
	assert(equal(0X1FFFP-16, 0.1249847412109375))
	assert(equal(0x1.fffffffffffffp1023, 1.7976931348623157e308))

	assert(equal(0x_1p-2, 0.25))
	assert(equal(0x2.p1_0, 2048.0))
	assert(equal(0x1_0.Fp+0, 16.9375))
	assert(equal(0X_0.8p-0, 0.5))
	assert(equal(0X_1FF_FP-16, 0.1249847412109375))
	assert(equal(0x1.f_ffff_ffff_ffffP1_023, 1.7976931348623157e308))

	// imaginaries
	assert(0i == complex(0, 0))
	assert(09i == complex(0, 9)) // "09i" is a decimal int followed by "i"
	assert(1.2e+3i == complex(0, 1.2e+3))

	assert(0_0i == complex(0, 0))
	assert(0_9i == complex(0, 9)) // "0_9i" is a decimal int followed by "i"
	assert(1.2_0e+0_3i == complex(0, 1.2e+3))
}
