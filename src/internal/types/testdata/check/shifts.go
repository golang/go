// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package shifts

func shifts0() {
	// basic constant shifts
	const (
		s = 10
		_ = 0<<0
		_ = 1<<s
		_ = 1<<- /* ERROR "negative shift count" */ 1
		// For the test below we may decide to convert to int
		// rather than uint and then report a negative shift
		// count instead, which might be a better error. The
		// (minor) difference is that this would restrict the
		// shift count range by half (from all uint values to
		// the positive int values).
		// This depends on the exact spec wording which is not
		// done yet.
		// TODO(gri) revisit and adjust when spec change is done
		_ = 1<<- /* ERROR "negative shift count" */ 1.0
		_ = 1<<1075 /* ERROR "invalid shift" */
		_ = 2.0<<1
		_ = 1<<1.0
		_ = 1<<(1+0i)

		_ int = 2<<s
		_ float32 = 2<<s
		_ complex64 = 2<<s

		_ int = 2.0<<s
		_ float32 = 2.0<<s
		_ complex64 = 2.0<<s

		_ int = 'a'<<s
		_ float32 = 'a'<<s
		_ complex64 = 'a'<<s
	)
}

func shifts1() {
	// basic non-constant shifts
	var (
		i int
		u uint

		_ = 1<<0
		_ = 1<<i
		_ = 1<<u
		_ = 1<<"foo" /* ERROR "cannot convert" */
		_ = i<<0
		_ = i<<- /* ERROR "negative shift count" */ 1
		_ = i<<1.0
		_ = 1<<(1+0i)
		_ = 1 /* ERROR "overflows" */ <<100

		_ uint = 1 << 0
		_ uint = 1 << u
		_ float32 = 1 /* ERROR "must be integer" */ << u

		// issue #14822
		_ = 1<<( /* ERROR "overflows uint" */ 1<<64)
		_ = 1<<( /* ERROR "invalid shift count" */ 1<<64-1)

		// issue #43697
		_ = u<<( /* ERROR "overflows uint" */ 1<<64)
		_ = u<<(1<<64-1)
	)
}

func shifts2() {
	// from the spec
	var (
		s uint = 33
		i = 1<<s           // 1 has type int
		j int32 = 1<<s     // 1 has type int32; j == 0
		k = uint64(1<<s)   // 1 has type uint64; k == 1<<33
		m int = 1.0<<s     // 1.0 has type int
		n = 1.0<<s != i    // 1.0 has type int; n == false if ints are 32bits in size
		o = 1<<s == 2<<s   // 1 and 2 have type int; o == true if ints are 32bits in size
		p = 1<<s == 1<<33  // illegal if ints are 32bits in size: 1 has type int, but 1<<33 overflows int
		u = 1.0 /* ERROR "must be integer" */ <<s         // illegal: 1.0 has type float64, cannot shift
		u1 = 1.0 /* ERROR "must be integer" */ <<s != 0   // illegal: 1.0 has type float64, cannot shift
		u2 = 1 /* ERROR "must be integer" */ <<s != 1.0   // illegal: 1 has type float64, cannot shift
		v float32 = 1 /* ERROR "must be integer" */ <<s   // illegal: 1 has type float32, cannot shift
		w int64 = 1.0<<33  // 1.0<<33 is a constant shift expression
	)
	_, _, _, _, _, _, _, _, _, _, _, _ = i, j, k, m, n, o, p, u, u1, u2, v, w
}

func shifts3(a int16, b float32) {
	// random tests
	var (
		s uint = 11
		u = 1 /* ERROR "must be integer" */ <<s + 1.0
		v complex128 = 1 /* ERROR "must be integer" */ << s + 1.0 /* ERROR "must be integer" */ << s + 1
	)
	x := 1.0 /* ERROR "must be integer" */ <<s + 1
	shifts3(1.0 << s, 1 /* ERROR "must be integer" */ >> s)
	_, _, _ = u, v, x
}

func shifts4() {
	// shifts in comparisons w/ untyped operands
	var s uint

	_ = 1<<s == 1
	_ = 1 /* ERROR "integer" */ <<s == 1.
	_ = 1. /* ERROR "integer" */ <<s == 1
	_ = 1. /* ERROR "integer" */ <<s == 1.

	_ = 1<<s + 1 == 1
	_ = 1 /* ERROR "integer" */ <<s + 1 == 1.
	_ = 1 /* ERROR "integer" */ <<s + 1. == 1
	_ = 1 /* ERROR "integer" */ <<s + 1. == 1.
	_ = 1. /* ERROR "integer" */ <<s + 1 == 1
	_ = 1. /* ERROR "integer" */ <<s + 1 == 1.
	_ = 1. /* ERROR "integer" */ <<s + 1. == 1
	_ = 1. /* ERROR "integer" */ <<s + 1. == 1.

	_ = 1<<s == 1<<s
	_ = 1 /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s == 1 /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s

	_ = 1<<s + 1<<s == 1
	_ = 1 /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1.
	_ = 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1
	_ = 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1.
	_ = 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1
	_ = 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1.
	_ = 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1
	_ = 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1.

	_ = 1<<s + 1<<s == 1<<s + 1<<s
	_ = 1 /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s
	_ = 1 /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s
	_ = 1 /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s
	_ = 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1 /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s
	_ = 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s
	_ = 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s
	_ = 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1 /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1 /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1 /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s + 1 /* ERROR "integer" */ <<s
	_ = 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s == 1. /* ERROR "integer" */ <<s + 1. /* ERROR "integer" */ <<s
}

func shifts5() {
	// shifts in comparisons w/ typed operands
	var s uint
	var x int

	_ = 1<<s == x
	_ = 1.<<s == x
	_ = 1.1 /* ERROR "int" */ <<s == x

	_ = 1<<s + x == 1
	_ = 1<<s + x == 1.
	_ = 1<<s + x == 1.1 /* ERROR "int" */
	_ = 1.<<s + x == 1
	_ = 1.<<s + x == 1.
	_ = 1.<<s + x == 1.1 /* ERROR "int" */
	_ = 1.1 /* ERROR "int" */ <<s + x == 1
	_ = 1.1 /* ERROR "int" */ <<s + x == 1.
	_ = 1.1 /* ERROR "int" */ <<s + x == 1.1

	_ = 1<<s == x<<s
	_ = 1.<<s == x<<s
	_ = 1.1  /* ERROR "int" */ <<s == x<<s
}

func shifts6() {
	// shifts as operands in non-arithmetic operations and as arguments
	var a [10]int
	var s uint

	_ = a[1<<s]
	_ = a[1.0]
	_ = a[1.0<<s]

	_ = make([]int, 1.0)
	_ = make([]int, 1.0<<s)
	_ = make([]int, 1.1 /* ERROR "must be integer" */ <<s)

	_ = float32(1)
	_ = float32(1 /* ERROR "must be integer" */ <<s)
	_ = float32(1.0)
	_ = float32(1.0 /* ERROR "must be integer" */ <<s)
	_ = float32(1.1 /* ERROR "must be integer" */ <<s)

	// TODO(gri) Re-enable these tests once types2 has the go/types fixes.
	//           Issue #52080.
	// _ = int32(0x80000000 /* ERROR "overflows int32" */ << s)
	// TODO(rfindley) Eliminate the redundant error here.
	// _ = int32(( /* ERROR "truncated to int32" */ 0x80000000 /* ERROR "truncated to int32" */ + 0i) << s)

	_ = int(1+0i<<0)
	// _ = int((1+0i)<<s)
	// _ = int(1.0<<s)
	// _ = int(complex(1, 0)<<s)
	_ = int(float32/* ERROR "must be integer" */(1.0) <<s)
	_ = int(1.1 /* ERROR "must be integer" */ <<s)
	_ = int(( /* ERROR "must be integer" */ 1+1i)  <<s)

	_ = complex(1 /* ERROR "must be integer" */ <<s, 0)

	var b []int
	_ = append(b, 1<<s)
	_ = append(b, 1.0<<s)
	_ = append(b, (1+0i)<<s)
	_ = append(b, 1.1 /* ERROR "must be integer" */ <<s)
	_ = append(b, (1 + 0i) <<s)
	_ = append(b, ( /* ERROR "must be integer" */ 1 + 1i)  <<s)

	_ = complex(1.0 /* ERROR "must be integer" */ <<s, 0)
	_ = complex(1.1 /* ERROR "must be integer" */ <<s, 0)
	_ = complex(0, 1.0 /* ERROR "must be integer" */ <<s)
	_ = complex(0, 1.1 /* ERROR "must be integer" */ <<s)

	// TODO(gri) The delete below is not type-checked correctly yet.
	// var m1 map[int]string
	// delete(m1, 1<<s)
}

func shifts7() {
	// shifts of shifts
	var s uint
	var x int
	_ = x

	_ = 1<<(1<<s)
	_ = 1<<(1.<<s)
	_ = 1. /* ERROR "integer" */ <<(1<<s)
	_ = 1. /* ERROR "integer" */ <<(1.<<s)

	x = 1<<(1<<s)
	x = 1<<(1.<<s)
	x = 1.<<(1<<s)
	x = 1.<<(1.<<s)

	_ = (1<<s)<<(1<<s)
	_ = (1<<s)<<(1.<<s)
	_ = ( /* ERROR "integer" */ 1.<<s)<<(1<<s)
	_ = ( /* ERROR "integer" */ 1.<<s)<<(1.<<s)

	x = (1<<s)<<(1<<s)
	x = (1<<s)<<(1.<<s)
	x = ( /* ERROR "integer" */ 1.<<s)<<(1<<s)
	x = ( /* ERROR "integer" */ 1.<<s)<<(1.<<s)
}

func shifts8() {
	// shift examples from shift discussion: better error messages
	var s uint
	_ = 1.0 /* ERROR "shifted operand 1.0 (type float64) must be integer" */ <<s == 1
	_ = 1.0 /* ERROR "shifted operand 1.0 (type float64) must be integer" */ <<s == 1.0
	_ = 1 /* ERROR "shifted operand 1 (type float64) must be integer" */ <<s == 1.0
	_ = 1 /* ERROR "shifted operand 1 (type float64) must be integer" */ <<s + 1.0 == 1
	_ = 1 /* ERROR "shifted operand 1 (type float64) must be integer" */ <<s + 1.1 == 1
	_ = 1 /* ERROR "shifted operand 1 (type float64) must be integer" */ <<s + 1 == 1.0

	// additional cases
	_ = complex(1.0 /* ERROR "shifted operand 1.0 (type float64) must be integer" */ <<s, 1)
	_ = complex(1.0, 1 /* ERROR "shifted operand 1 (type float64) must be integer" */ <<s)

	_ = int(1.<<s)
	_ = int(1.1 /* ERRORx `shifted operand .* must be integer` */ <<s)
	_ = float32(1 /* ERRORx `shifted operand .* must be integer` */ <<s)
	_ = float32(1. /* ERRORx `shifted operand .* must be integer` */ <<s)
	_ = float32(1.1 /* ERRORx `shifted operand .* must be integer` */ <<s)
	// TODO(gri) the error messages for these two are incorrect - disabled for now
	// _ = complex64(1<<s)
	// _ = complex64(1.<<s)
	_ = complex64(1.1 /* ERRORx `shifted operand .* must be integer` */ <<s)
}

func shifts9() {
	// various originally failing snippets of code from the std library
	// from src/compress/lzw/reader.go:90
	{
		var d struct {
			bits     uint32
			width    uint
		}
		_ = uint16(d.bits & (1<<d.width - 1))
	}

	// from src/debug/dwarf/buf.go:116
	{
		var ux uint64
		var bits uint
		x := int64(ux)
		if x&(1<<(bits-1)) != 0 {}
	}

	// from src/encoding/asn1/asn1.go:160
	{
		var bytes []byte
		if bytes[len(bytes)-1]&((1<<bytes[0])-1) != 0 {}
	}

	// from src/math/big/rat.go:140
	{
		var exp int
		var mantissa uint64
		shift := uint64(-1022 - (exp - 1)) // [1..53)
		_ = mantissa & (1<<shift - 1)
	}

	// from src/net/interface.go:51
	{
		type Flags uint
		var f Flags
		var i int
		if f&(1<<uint(i)) != 0 {}
	}

	// from src/runtime/softfloat64.go:234
	{
		var gm uint64
		var shift uint
		_ = gm & (1<<shift - 1)
	}

	// from src/strconv/atof.go:326
	{
		var mant uint64
		var mantbits uint
		if mant == 2<<mantbits {}
	}

	// from src/route_bsd.go:82
	{
		var Addrs int32
		const rtaRtMask = 1
		var i uint
		if Addrs&rtaRtMask&(1<<i) == 0 {}
	}

	// from src/text/scanner/scanner.go:540
	{
		var s struct { Whitespace uint64 }
		var ch rune
		for s.Whitespace&(1<<uint(ch)) != 0 {}
	}
}

func issue5895() {
	var x = 'a' << 1 // type of x must be rune
	var _ rune = x
}

func issue11325() {
	var _ = 0 >> 1.1 /* ERROR "truncated to uint" */ // example from issue 11325
	_ = 0 >> 1.1 /* ERROR "truncated to uint" */
	_ = 0 << 1.1 /* ERROR "truncated to uint" */
	_ = 0 >> 1.
	_ = 1 >> 1.1 /* ERROR "truncated to uint" */
	_ = 1 >> 1.
	_ = 1. >> 1
	_ = 1. >> 1.
	_ = 1.1 /* ERROR "must be integer" */ >> 1
}

func issue11594() {
	var _ = complex64 /* ERROR "must be integer" */ (1) << 2 // example from issue 11594
	_ = float32 /* ERROR "must be integer" */ (0) << 1
	_ = float64 /* ERROR "must be integer" */ (0) >> 2
	_ = complex64 /* ERROR "must be integer" */ (0) << 3
	_ = complex64 /* ERROR "must be integer" */ (0) >> 4
}

func issue21727() {
	var s uint
	var a = make([]int, 1<<s + 1.2 /* ERROR "truncated to int" */ )
	var _ = a[1<<s - 2.3 /* ERROR "truncated to int" */ ]
	var _ int = 1<<s + 3.4 /* ERROR "truncated to int" */
	var _ = string(1 /* ERRORx `shifted operand 1 .* must be integer` */ << s)
	var _ = string(1.0 /* ERROR "cannot convert" */ << s)
}

func issue22969() {
	var s uint
	var a []byte
	_ = a[0xffffffffffffffff /* ERROR "overflows int" */ <<s] // example from issue 22969
	_ = make([]int, 0xffffffffffffffff /* ERROR "overflows int" */ << s)
	_ = make([]int, 0, 0xffffffffffffffff /* ERROR "overflows int" */ << s)
	var _ byte = 0x100 /* ERROR "overflows byte" */ << s
	var _ int8 = 0xff /* ERROR "overflows int8" */ << s
	var _ int16 = 0xffff /* ERROR "overflows int16" */ << s
	var _ int32 = 0x80000000 /* ERROR "overflows int32" */ << s
}
