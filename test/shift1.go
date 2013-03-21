// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test illegal shifts.
// Issue 1708, illegal cases.
// Does not compile.

package p

func f(x int) int         { return 0 }
func g(x interface{}) int { return 0 }
func h(x float64) int     { return 0 }

// from the spec
var (
	s uint    = 33
	u         = 1.0 << s // ERROR "invalid operation|shift of non-integer operand"
	v float32 = 1 << s   // ERROR "invalid" "as type float32"
)

// non-constant shift expressions
var (
	e1       = g(2.0 << s) // ERROR "invalid" "as type interface"
	f1       = h(2 << s)   // ERROR "invalid" "as type float64"
	g1 int64 = 1.1 << s    // ERROR "truncated"
)

// constant shift expressions
const c uint = 65

var (
	a2 int = 1.0 << c    // ERROR "overflow"
	b2     = 1.0 << c    // ERROR "overflow"
	d2     = f(1.0 << c) // ERROR "overflow"
)

var (
	// issues 4882, 4936.
	a3 = 1.0<<s + 0 // ERROR "invalid operation|shift of non-integer operand"
	// issue 4937
	b3 = 1<<s + 1 + 1.0 // ERROR "invalid operation|shift of non-integer operand"
	// issue 5014
	c3     = complex(1<<s, 0) // ERROR "shift of type float64"
	d3 int = complex(1<<s, 3) // ERROR "cannot use.*as type int" "shift of type float64"
	e3     = real(1 << s)     // ERROR "invalid"
	f3     = imag(1 << s)     // ERROR "invalid"
)

// from the spec
func _() {
	var (
		s uint  = 33
		i       = 1 << s         // 1 has type int
		j int32 = 1 << s         // 1 has type int32; j == 0
		k       = uint64(1 << s) // 1 has type uint64; k == 1<<33
		m int   = 1.0 << s       // 1.0 has type int
		n       = 1.0<<s != i    // 1.0 has type int; n == false if ints are 32bits in size
		o       = 1<<s == 2<<s   // 1 and 2 have type int; o == true if ints are 32bits in size
		// next test only fails on 32bit systems
		// p = 1<<s == 1<<33  // illegal if ints are 32bits in size: 1 has type int, but 1<<33 overflows int
		u          = 1.0 << s    // ERROR "float64"
		u1         = 1.0<<s != 0 // ERROR "float64"
		u2         = 1<<s != 1.0 // ERROR "float64"
		v  float32 = 1 << s      // ERROR "float32"
		w  int64   = 1.0 << 33   // 1.0<<33 is a constant shift expression
	)
}

// shifts in comparisons w/ untyped operands
var (
	_ = 1<<s == 1
	_ = 1<<s == 1.  // ERROR "shift of type float64"
	_ = 1.<<s == 1  // ERROR "shift of type float64"
	_ = 1.<<s == 1. // ERROR "shift of type float64"

	_ = 1<<s+1 == 1
	_ = 1<<s+1 == 1.   // ERROR "shift of type float64"
	_ = 1<<s+1. == 1   // ERROR "shift of type float64"
	_ = 1<<s+1. == 1.  // ERROR "shift of type float64"
	_ = 1.<<s+1 == 1   // ERROR "shift of type float64"
	_ = 1.<<s+1 == 1.  // ERROR "shift of type float64"
	_ = 1.<<s+1. == 1  // ERROR "shift of type float64"
	_ = 1.<<s+1. == 1. // ERROR "shift of type float64"

	_ = 1<<s == 1<<s
	_ = 1<<s == 1.<<s  // ERROR "shift of type float64"
	_ = 1.<<s == 1<<s  // ERROR "shift of type float64"
	_ = 1.<<s == 1.<<s // ERROR "shift of type float64"

	_ = 1<<s+1<<s == 1
	_ = 1<<s+1<<s == 1.   // ERROR "shift of type float64"
	_ = 1<<s+1.<<s == 1   // ERROR "shift of type float64"
	_ = 1<<s+1.<<s == 1.  // ERROR "shift of type float64"
	_ = 1.<<s+1<<s == 1   // ERROR "shift of type float64"
	_ = 1.<<s+1<<s == 1.  // ERROR "shift of type float64"
	_ = 1.<<s+1.<<s == 1  // ERROR "shift of type float64"
	_ = 1.<<s+1.<<s == 1. // ERROR "shift of type float64"

	_ = 1<<s+1<<s == 1<<s+1<<s
	_ = 1<<s+1<<s == 1<<s+1.<<s    // ERROR "shift of type float64"
	_ = 1<<s+1<<s == 1.<<s+1<<s    // ERROR "shift of type float64"
	_ = 1<<s+1<<s == 1.<<s+1.<<s   // ERROR "shift of type float64"
	_ = 1<<s+1.<<s == 1<<s+1<<s    // ERROR "shift of type float64"
	_ = 1<<s+1.<<s == 1<<s+1.<<s   // ERROR "shift of type float64"
	_ = 1<<s+1.<<s == 1.<<s+1<<s   // ERROR "shift of type float64"
	_ = 1<<s+1.<<s == 1.<<s+1.<<s  // ERROR "shift of type float64"
	_ = 1.<<s+1<<s == 1<<s+1<<s    // ERROR "shift of type float64"
	_ = 1.<<s+1<<s == 1<<s+1.<<s   // ERROR "shift of type float64"
	_ = 1.<<s+1<<s == 1.<<s+1<<s   // ERROR "shift of type float64"
	_ = 1.<<s+1<<s == 1.<<s+1.<<s  // ERROR "shift of type float64"
	_ = 1.<<s+1.<<s == 1<<s+1<<s   // ERROR "shift of type float64"
	_ = 1.<<s+1.<<s == 1<<s+1.<<s  // ERROR "shift of type float64"
	_ = 1.<<s+1.<<s == 1.<<s+1<<s  // ERROR "shift of type float64"
	_ = 1.<<s+1.<<s == 1.<<s+1.<<s // ERROR "shift of type float64"
)

// shifts in comparisons w/ typed operands
var (
	x int
	_ = 1<<s == x
	_ = 1.<<s == x
	_ = 1.1<<s == x // ERROR "1.1 truncated"

	_ = 1<<s+x == 1
	_ = 1<<s+x == 1.
	_ = 1<<s+x == 1.1 // ERROR "1.1 truncated"
	_ = 1.<<s+x == 1
	_ = 1.<<s+x == 1.
	_ = 1.<<s+x == 1.1  // ERROR "1.1 truncated"
	_ = 1.1<<s+x == 1   // ERROR "1.1 truncated"
	_ = 1.1<<s+x == 1.  // ERROR "1.1 truncated"
	_ = 1.1<<s+x == 1.1 // ERROR "1.1 truncated"

	_ = 1<<s == x<<s
	_ = 1.<<s == x<<s
	_ = 1.1<<s == x<<s // ERROR "1.1 truncated"
)

// shifts as operands in non-arithmetic operations and as arguments
func _() {
	var s uint
	var a []int
	_ = a[1<<s]
	_ = a[1.]
	// For now, the spec disallows these. We may revisit past Go 1.1.
	_ = a[1.<<s]  // ERROR "shift of type float64"
	_ = a[1.1<<s] // ERROR "shift of type float64"

	_ = make([]int, 1)
	_ = make([]int, 1.)
	_ = make([]int, 1.<<s)
	_ = make([]int, 1.1<<s) // ERROR "1.1 truncated"

	_ = float32(1)
	_ = float32(1 << s) // ERROR "shift of type float32"
	_ = float32(1.)
	_ = float32(1. << s)  // ERROR "shift of type float32"
	_ = float32(1.1 << s) // ERROR "shift of type float32"

	_ = append(a, 1<<s)
	_ = append(a, 1.<<s)
	_ = append(a, 1.1<<s) // ERROR "1.1 truncated"

	var b []float32
	_ = append(b, 1<<s)   // ERROR "type float32"
	_ = append(b, 1.<<s)  // ERROR "type float32"
	_ = append(b, 1.1<<s) // ERROR "type float32"

	_ = complex(1.<<s, 0)  // ERROR "shift of type float64"
	_ = complex(1.1<<s, 0) // ERROR "shift of type float64"
	_ = complex(0, 1.<<s)  // ERROR "shift of type float64"
	_ = complex(0, 1.1<<s) // ERROR "shift of type float64"

	var a4 float64
	var b4 int
	_ = complex(1<<s, a4) // ERROR "shift of type float64"
	_ = complex(1<<s, b4) // ERROR "invalid"

	var m1 map[int]string
	delete(m1, 1<<s)
	delete(m1, 1.<<s)
	delete(m1, 1.1<<s) // ERROR "1.1 truncated|shift of type float64"

	var m2 map[float32]string
	delete(m2, 1<<s)   // ERROR "invalid|cannot use 1 << s as type float32"
	delete(m2, 1.<<s)  // ERROR "invalid|cannot use 1 << s as type float32"
	delete(m2, 1.1<<s) // ERROR "invalid|cannot use 1.1 << s as type float32"
}

// shifts of shifts
func _() {
	var s uint
	_ = 1 << (1 << s)
	_ = 1 << (1. << s)
	_ = 1 << (1.1 << s)   // ERROR "1.1 truncated"
	_ = 1. << (1 << s)    // ERROR "shift of type float64"
	_ = 1. << (1. << s)   // ERROR "shift of type float64"
	_ = 1.1 << (1.1 << s) // ERROR "invalid|1.1 truncated"

	_ = (1 << s) << (1 << s)
	_ = (1 << s) << (1. << s)
	_ = (1 << s) << (1.1 << s)   // ERROR "1.1 truncated"
	_ = (1. << s) << (1 << s)    // ERROR "shift of type float64"
	_ = (1. << s) << (1. << s)   // ERROR "shift of type float64"
	_ = (1.1 << s) << (1.1 << s) // ERROR "invalid|1.1 truncated"

	var x int
	x = 1 << (1 << s)
	x = 1 << (1. << s)
	x = 1 << (1.1 << s) // ERROR "1.1 truncated"
	x = 1. << (1 << s)
	x = 1. << (1. << s)
	x = 1.1 << (1.1 << s) // ERROR "1.1 truncated"

	x = (1 << s) << (1 << s)
	x = (1 << s) << (1. << s)
	x = (1 << s) << (1.1 << s) // ERROR "1.1 truncated"
	x = (1. << s) << (1 << s)
	x = (1. << s) << (1. << s)
	x = (1.1 << s) << (1.1 << s) // ERROR "1.1 truncated"

	var y float32
	y = 1 << (1 << s)     // ERROR "type float32"
	y = 1 << (1. << s)    // ERROR "type float32"
	y = 1 << (1.1 << s)   // ERROR "invalid|1.1 truncated|float32"
	y = 1. << (1 << s)    // ERROR "type float32"
	y = 1. << (1. << s)   // ERROR "type float32"
	y = 1.1 << (1.1 << s) // ERROR "invalid|1.1 truncated|float32"

	var z complex128
	z = (1 << s) << (1 << s)     // ERROR "type complex128"
	z = (1 << s) << (1. << s)    // ERROR "type complex128"
	z = (1 << s) << (1.1 << s)   // ERROR "invalid|1.1 truncated|complex128"
	z = (1. << s) << (1 << s)    // ERROR "type complex128"
	z = (1. << s) << (1. << s)   // ERROR "type complex128"
	z = (1.1 << s) << (1.1 << s) // ERROR "invalid|1.1 truncated|complex128"
}
