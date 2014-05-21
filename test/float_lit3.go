// errorcheck

// Check flagging of invalid conversion of constant to float32/float64 near min/max boundaries.

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// See float_lit2.go for motivation for these values.
const (
	two24   = 1.0 * (1 << 24)
	two53   = 1.0 * (1 << 53)
	two64   = 1.0 * (1 << 64)
	two128  = two64 * two64
	two256  = two128 * two128
	two512  = two256 * two256
	two768  = two512 * two256
	two1024 = two512 * two512

	ulp32 = two128 / two24
	max32 = two128 - ulp32

	ulp64 = two1024 / two53
	max64 = two1024 - ulp64
)

var x = []interface{}{
	float32(max32 + ulp32/2 - 1),             // ok
	float32(max32 + ulp32/2 - two128/two256), // ok
	float32(max32 + ulp32/2),                 // ERROR "constant 3\.40282e\+38 overflows float32"

	float32(-max32 - ulp32/2 + 1),             // ok
	float32(-max32 - ulp32/2 + two128/two256), // ok
	float32(-max32 - ulp32/2),                 // ERROR "constant -3\.40282e\+38 overflows float32"

	// If the compiler's internal floating point representation
	// is shorter than 1024 bits, it cannot distinguish max64+ulp64/2-1 and max64+ulp64/2.
	// gc uses fewer than 1024 bits, so allow it to print the overflow error for the -1 case.
	float64(max64 + ulp64/2 - two1024/two256), // ok
	float64(max64 + ulp64/2 - 1),              // GC_ERROR "constant 1\.79769e\+308 overflows float64"
	float64(max64 + ulp64/2),                  // ERROR "constant 1\.79769e\+308 overflows float64"

	float64(-max64 - ulp64/2 + two1024/two256), // ok
	float64(-max64 - ulp64/2 + 1),              // GC_ERROR "constant -1\.79769e\+308 overflows float64"
	float64(-max64 - ulp64/2),                  // ERROR "constant -1\.79769e\+308 overflows float64"
}
