// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"math"
	"math/cmplx"
	"reflect"
	. "strconv"
	"testing"
)

var (
	infp0 = complex(math.Inf(+1), 0)
	infm0 = complex(math.Inf(-1), 0)
	inf0p = complex(0, math.Inf(+1))
	inf0m = complex(0, math.Inf(-1))

	infpp = complex(math.Inf(+1), math.Inf(+1))
	infpm = complex(math.Inf(+1), math.Inf(-1))
	infmp = complex(math.Inf(-1), math.Inf(+1))
	infmm = complex(math.Inf(-1), math.Inf(-1))
)

type atocTest struct {
	in  string
	out complex128
	err error
}

func TestParseComplex(t *testing.T) {
	tests := []atocTest{
		// Clearly invalid
		{"", 0, ErrSyntax},
		{" ", 0, ErrSyntax},
		{"(", 0, ErrSyntax},
		{")", 0, ErrSyntax},
		{"i", 0, ErrSyntax},
		{"+i", 0, ErrSyntax},
		{"-i", 0, ErrSyntax},
		{"1I", 0, ErrSyntax},
		{"10  + 5i", 0, ErrSyntax},
		{"3+", 0, ErrSyntax},
		{"3+5", 0, ErrSyntax},
		{"3+5+5i", 0, ErrSyntax},

		// Parentheses
		{"()", 0, ErrSyntax},
		{"(i)", 0, ErrSyntax},
		{"(0)", 0, nil},
		{"(1i)", 1i, nil},
		{"(3.0+5.5i)", 3.0 + 5.5i, nil},
		{"(1)+1i", 0, ErrSyntax},
		{"(3.0+5.5i", 0, ErrSyntax},
		{"3.0+5.5i)", 0, ErrSyntax},

		// NaNs
		{"NaN", complex(math.NaN(), 0), nil},
		{"NANi", complex(0, math.NaN()), nil},
		{"nan+nAni", complex(math.NaN(), math.NaN()), nil},
		{"+NaN", 0, ErrSyntax},
		{"-NaN", 0, ErrSyntax},
		{"NaN-NaNi", 0, ErrSyntax},

		// Infs
		{"Inf", infp0, nil},
		{"+inf", infp0, nil},
		{"-inf", infm0, nil},
		{"Infinity", infp0, nil},
		{"+INFINITY", infp0, nil},
		{"-infinity", infm0, nil},
		{"+infi", inf0p, nil},
		{"0-infinityi", inf0m, nil},
		{"Inf+Infi", infpp, nil},
		{"+Inf-Infi", infpm, nil},
		{"-Infinity+Infi", infmp, nil},
		{"inf-inf", 0, ErrSyntax},

		// Zeros
		{"0", 0, nil},
		{"0i", 0, nil},
		{"-0.0i", 0, nil},
		{"0+0.0i", 0, nil},
		{"0e+0i", 0, nil},
		{"0e-0+0i", 0, nil},
		{"-0.0-0.0i", 0, nil},
		{"0e+012345", 0, nil},
		{"0x0p+012345i", 0, nil},
		{"0x0.00p-012345i", 0, nil},
		{"+0e-0+0e-0i", 0, nil},
		{"0e+0+0e+0i", 0, nil},
		{"-0e+0-0e+0i", 0, nil},

		// Regular non-zeroes
		{"0.1", 0.1, nil},
		{"0.1i", 0 + 0.1i, nil},
		{"0.123", 0.123, nil},
		{"0.123i", 0 + 0.123i, nil},
		{"0.123+0.123i", 0.123 + 0.123i, nil},
		{"99", 99, nil},
		{"+99", 99, nil},
		{"-99", -99, nil},
		{"+1i", 1i, nil},
		{"-1i", -1i, nil},
		{"+3+1i", 3 + 1i, nil},
		{"30+3i", 30 + 3i, nil},
		{"+3e+3-3e+3i", 3e+3 - 3e+3i, nil},
		{"+3e+3+3e+3i", 3e+3 + 3e+3i, nil},
		{"+3e+3+3e+3i+", 0, ErrSyntax},

		// Separators
		{"0.1", 0.1, nil},
		{"0.1i", 0 + 0.1i, nil},
		{"0.1_2_3", 0.123, nil},
		{"+0x_3p3i", 0x3p3i, nil},
		{"0_0+0x_0p0i", 0, nil},
		{"0x_10.3p-8+0x3p3i", 0x10.3p-8 + 0x3p3i, nil},
		{"+0x_1_0.3p-8+0x_3_0p3i", 0x10.3p-8 + 0x30p3i, nil},
		{"0x1_0.3p+8-0x_3p3i", 0x10.3p+8 - 0x3p3i, nil},

		// Hexadecimals
		{"0x10.3p-8+0x3p3i", 0x10.3p-8 + 0x3p3i, nil},
		{"+0x10.3p-8+0x3p3i", 0x10.3p-8 + 0x3p3i, nil},
		{"0x10.3p+8-0x3p3i", 0x10.3p+8 - 0x3p3i, nil},
		{"0x1p0", 1, nil},
		{"0x1p1", 2, nil},
		{"0x1p-1", 0.5, nil},
		{"0x1ep-1", 15, nil},
		{"-0x1ep-1", -15, nil},
		{"-0x2p3", -16, nil},
		{"0x1e2", 0, ErrSyntax},
		{"1p2", 0, ErrSyntax},
		{"0x1e2i", 0, ErrSyntax},

		// ErrRange
		// next float64 - too large
		{"+0x1p1024", infp0, ErrRange},
		{"-0x1p1024", infm0, ErrRange},
		{"+0x1p1024i", inf0p, ErrRange},
		{"-0x1p1024i", inf0m, ErrRange},
		{"+0x1p1024+0x1p1024i", infpp, ErrRange},
		{"+0x1p1024-0x1p1024i", infpm, ErrRange},
		{"-0x1p1024+0x1p1024i", infmp, ErrRange},
		{"-0x1p1024-0x1p1024i", infmm, ErrRange},
		// the border is ...158079
		// borderline - okay
		{"+0x1.fffffffffffff7fffp1023+0x1.fffffffffffff7fffp1023i", 1.7976931348623157e+308 + 1.7976931348623157e+308i, nil},
		{"+0x1.fffffffffffff7fffp1023-0x1.fffffffffffff7fffp1023i", 1.7976931348623157e+308 - 1.7976931348623157e+308i, nil},
		{"-0x1.fffffffffffff7fffp1023+0x1.fffffffffffff7fffp1023i", -1.7976931348623157e+308 + 1.7976931348623157e+308i, nil},
		{"-0x1.fffffffffffff7fffp1023-0x1.fffffffffffff7fffp1023i", -1.7976931348623157e+308 - 1.7976931348623157e+308i, nil},
		// borderline - too large
		{"+0x1.fffffffffffff8p1023", infp0, ErrRange},
		{"-0x1fffffffffffff.8p+971", infm0, ErrRange},
		{"+0x1.fffffffffffff8p1023i", inf0p, ErrRange},
		{"-0x1fffffffffffff.8p+971i", inf0m, ErrRange},
		{"+0x1.fffffffffffff8p1023+0x1.fffffffffffff8p1023i", infpp, ErrRange},
		{"+0x1.fffffffffffff8p1023-0x1.fffffffffffff8p1023i", infpm, ErrRange},
		{"-0x1fffffffffffff.8p+971+0x1fffffffffffff.8p+971i", infmp, ErrRange},
		{"-0x1fffffffffffff8p+967-0x1fffffffffffff8p+967i", infmm, ErrRange},
		// a little too large
		{"1e308+1e308i", 1e+308 + 1e+308i, nil},
		{"2e308+2e308i", infpp, ErrRange},
		{"1e309+1e309i", infpp, ErrRange},
		{"0x1p1025+0x1p1025i", infpp, ErrRange},
		{"2e308", infp0, ErrRange},
		{"1e309", infp0, ErrRange},
		{"0x1p1025", infp0, ErrRange},
		{"2e308i", inf0p, ErrRange},
		{"1e309i", inf0p, ErrRange},
		{"0x1p1025i", inf0p, ErrRange},
		// way too large
		{"+1e310+1e310i", infpp, ErrRange},
		{"+1e310-1e310i", infpm, ErrRange},
		{"-1e310+1e310i", infmp, ErrRange},
		{"-1e310-1e310i", infmm, ErrRange},
		// under/overflow exponent
		{"1e-4294967296", 0, nil},
		{"1e-4294967296i", 0, nil},
		{"1e-4294967296+1i", 1i, nil},
		{"1+1e-4294967296i", 1, nil},
		{"1e-4294967296+1e-4294967296i", 0, nil},
		{"1e+4294967296", infp0, ErrRange},
		{"1e+4294967296i", inf0p, ErrRange},
		{"1e+4294967296+1e+4294967296i", infpp, ErrRange},
		{"1e+4294967296-1e+4294967296i", infpm, ErrRange},
	}
	for i := range tests {
		test := &tests[i]
		if test.err != nil {
			test.err = &NumError{Func: "ParseComplex", Num: test.in, Err: test.err}
		}
		got, err := ParseComplex(test.in, 128)
		if !reflect.DeepEqual(err, test.err) {
			t.Fatalf("ParseComplex(%q, 128) = %v, %v; want %v, %v", test.in, got, err, test.out, test.err)
		}
		if !(cmplx.IsNaN(test.out) && cmplx.IsNaN(got)) && got != test.out {
			t.Fatalf("ParseComplex(%q, 128) = %v, %v; want %v, %v", test.in, got, err, test.out, test.err)
		}

		if complex128(complex64(test.out)) == test.out {
			got, err := ParseComplex(test.in, 64)
			if !reflect.DeepEqual(err, test.err) {
				t.Fatalf("ParseComplex(%q, 64) = %v, %v; want %v, %v", test.in, got, err, test.out, test.err)
			}
			got64 := complex64(got)
			if complex128(got64) != test.out {
				t.Fatalf("ParseComplex(%q, 64) = %v, %v; want %v, %v", test.in, got, err, test.out, test.err)
			}
		}
	}
}

// Issue 42297: allow ParseComplex(s, not_32_or_64) for legacy reasons
func TestParseComplexIncorrectBitSize(t *testing.T) {
	const s = "1.5e308+1.0e307i"
	const want = 1.5e308 + 1.0e307i

	for _, bitSize := range []int{0, 10, 100, 256} {
		c, err := ParseComplex(s, bitSize)
		if err != nil {
			t.Fatalf("ParseComplex(%q, %d) gave error %s", s, bitSize, err)
		}
		if c != want {
			t.Fatalf("ParseComplex(%q, %d) = %g (expected %g)", s, bitSize, c, want)
		}
	}
}
