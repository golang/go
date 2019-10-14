// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests various representations of literals
// and compares them with literals or constant expressions
// of equal values.

package literals

func _() {
	// 0-octals
	assert(0_123 == 0123)
	assert(0123_456 == 0123456)

	// decimals
	assert(1_234 == 1234)
	assert(1_234_567 == 1234567)

	// hexadecimals
	assert(0X_0 == 0)
	assert(0X_1234 == 0x1234)
	assert(0X_CAFE_f00d == 0xcafef00d)

	// octals
	assert(0o0 == 0)
	assert(0o1234 == 01234)
	assert(0o01234567 == 01234567)

	assert(0O0 == 0)
	assert(0O1234 == 01234)
	assert(0O01234567 == 01234567)

	assert(0o_0 == 0)
	assert(0o_1234 == 01234)
	assert(0o0123_4567 == 01234567)

	assert(0O_0 == 0)
	assert(0O_1234 == 01234)
	assert(0O0123_4567 == 01234567)

	// binaries
	assert(0b0 == 0)
	assert(0b1011 == 0xb)
	assert(0b00101101 == 0x2d)

	assert(0B0 == 0)
	assert(0B1011 == 0xb)
	assert(0B00101101 == 0x2d)

	assert(0b_0 == 0)
	assert(0b10_11 == 0xb)
	assert(0b_0010_1101 == 0x2d)

	// decimal floats
	assert(1_2_3. == 123.)
	assert(0_123. == 123.)

	assert(0_0e0 == 0.)
	assert(1_2_3e0 == 123.)
	assert(0_123e0 == 123.)

	assert(0e-0_0 == 0.)
	assert(1_2_3E+0 == 123.)
	assert(0123E1_2_3 == 123e123)

	assert(0.e+1 == 0.)
	assert(123.E-1_0 == 123e-10)
	assert(01_23.e123 == 123e123)

	assert(.0e-1 == .0)
	assert(.123E+10 == .123e10)
	assert(.0123E123 == .0123e123)

	assert(1_2_3.123 == 123.123)
	assert(0123.01_23 == 123.0123)

	// hexadecimal floats
	assert(0x0.p+0 == 0.)
	assert(0Xdeadcafe.p-10 == 0xdeadcafe/1024.0)
	assert(0x1234.P84 == 0x1234000000000000000000000)

	assert(0x.1p-0 == 1./16)
	assert(0X.deadcafep4 == 1.0*0xdeadcafe/0x10000000)
	assert(0x.1234P+12 == 1.0*0x1234/0x10)

	assert(0x0p0 == 0.)
	assert(0Xdeadcafep+1 == 0x1bd5b95fc)
	assert(0x1234P-10 == 0x1234/1024.0)

	assert(0x0.0p0 == 0.)
	assert(0Xdead.cafep+1 == 1.0*0x1bd5b95fc/0x10000)
	assert(0x12.34P-10 == 1.0*0x1234/0x40000)

	assert(0Xdead_cafep+1 == 0xdeadcafep+1)
	assert(0x_1234P-10 == 0x1234p-10)

	assert(0X_dead_cafe.p-10 == 0xdeadcafe.p-10)
	assert(0x12_34.P1_2_3 == 0x1234.p123)

	assert(1_234i == 1234i)
	assert(1_234_567i == 1234567i)

	assert(0.i == 0i)
	assert(123.i == 123i)
	assert(0123.i == 123i)

	assert(0.e+1i == 0i)
	assert(123.E-1_0i == 123e-10i)
	assert(01_23.e123i == 123e123i)
}
