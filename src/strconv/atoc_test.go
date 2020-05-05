// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	. "strconv"
	"math"
	"math/cmplx"
	"reflect"
	"testing"
)

type atocTest struct {
	in  string
	out complex128
	err error
}

func TestParseComplex(t *testing.T) {
	tests := []atocTest{
		// Clear Invalids
		{"", 0, ErrSyntax},
		{" ", 0, ErrSyntax},
		{"(", 0, ErrSyntax},
		{")", 0, ErrSyntax},
		{"foo", 0, ErrSyntax},
		{"10  + 5i", 0, ErrSyntax},
		{"3+3+5.5", 0, ErrSyntax},
		{"3+3+5.5i", 0, ErrSyntax},

		// Parentheses
		{"(3.0+5.5i)", 3.0 + 5.5i, nil},
		{"(3.0+5.5i", 0, ErrSyntax},

		// NaNs
		{"NaN", complex(math.NaN(), 0), nil},
		{"NaNi", complex(0, math.NaN()), nil},
		{"NaN+NaNi", complex(math.NaN(), math.NaN()), nil},
		{"NaN++NaNi", 0, ErrSyntax},
		{"+NaN", 0, ErrSyntax},
		{"++NaN", 0, ErrSyntax},
		{"-NaN", 0, ErrSyntax},
		{"+NaNi", 0, ErrSyntax},
		{"-NaNi", 0, ErrSyntax},
		{"NaN-NaNi", 0, ErrSyntax},

		// Infs
		{"Infi", complex(0, math.Inf(1)), nil},
		{"infi", complex(0, math.Inf(1)), nil},
		{"inf i", 0, ErrSyntax},
		{"-Infi", complex(0, math.Inf(-1)), nil},
		{"-infi", complex(0, math.Inf(-1)), nil},
		{"-inf i", 0, ErrSyntax},
		{"Inf", complex(math.Inf(1), 0), nil},
		{"-Inf", complex(math.Inf(-1), 0), nil},
		{"-Inf-Infi", complex(math.Inf(-1), math.Inf(-1)), nil},
		{"-Inf+Infi", complex(math.Inf(-1), math.Inf(1)), nil},
		{"-Inf-Inf i", 0, ErrSyntax},
		{"-Inf+Inf i", 0, ErrSyntax},
		{"-Inf- Inf i", 0, ErrSyntax},
		{"-Inf+ Inf i", 0, ErrSyntax},
		{"-Inf- Infi", 0, ErrSyntax},
		{"-Inf+ Infi", 0, ErrSyntax},

		// Zeros
		{"0", 0, nil},
		{"0i", 0, nil},
		{"0+0i", 0, nil},
		{"0.0", 0, nil},
		{"0.0i", 0, nil},
		{"0.0+0.0i", 0, nil},
		{"0.0-0.0i", 0, nil},
		{"-0.0-0.0i", 0, nil},
		{"-0.0+0.0i", 0, nil},
		{"0e0", 0, nil},
		{"-0e0", 0, nil},
		{"+0e0", 0, nil},
		{"0e-0", 0, nil},
		{"-0e-0", 0, nil},
		{"+0e-0", 0, nil},
		{"0e+0", 0, nil},
		{"-0e+0", 0, nil},
		{"+0e+0", 0, nil},
		{"0e+01234567890123456789", 0, nil},
		{"0.00e-01234567890123456789", 0, nil},
		{"-0e+01234567890123456789", 0, nil},
		{"-0.00e-01234567890123456789", 0, nil},
		{"0x0p+01234567890123456789", 0, nil},
		{"0x0.00p-01234567890123456789", 0, nil},
		{"-0x0p+01234567890123456789", 0, nil},
		{"-0x0.00p-01234567890123456789", 0, nil},
		{"0e0i", 0, nil},
		{"-0e0i", 0, nil},
		{"+0e0i", 0, nil},
		{"0e-0i", 0, nil},
		{"-0e-0i", 0, nil},
		{"+0e-0i", 0, nil},
		{"0e+0i", 0, nil},
		{"-0e+0i", 0, nil},
		{"+0e+0i", 0, nil},
		{"0e+01234567890123456789i", 0, nil},
		{"0.00e-01234567890123456789i", 0, nil},
		{"-0e+01234567890123456789i", 0, nil},
		{"-0.00e-01234567890123456789i", 0, nil},
		{"0x0p+01234567890123456789i", 0, nil},
		{"0x0.00p-01234567890123456789i", 0, nil},
		{"-0x0p+01234567890123456789i", 0, nil},
		{"-0x0.00p-01234567890123456789i", 0, nil},
		{"0+0i", 0, nil},
		{"0e0+0e0i", 0, nil},
		{"-0e0-0e0i", 0, nil},
		{"+0e0+0e0i", 0, nil},
		{"0e-0+0e-0i", 0, nil},
		{"-0e-0-0e-0i", 0, nil},
		{"+0e-0+0e-0i", 0, nil},
		{"0e+0+0e+0i", 0, nil},
		{"-0e+0-0e+0i", 0, nil},
		{"+0e+0+0e+0i", 0, nil},
		{"0e+01234567890123456789+0e+01234567890123456789i", 0, nil},
		{"0.00e-01234567890123456789+0.00e-01234567890123456789i", 0, nil},
		{"-0e+01234567890123456789-0e+01234567890123456789i", 0, nil},
		{"-0.00e-01234567890123456789-0.00e-01234567890123456789i", 0, nil},
		{"0x0p+01234567890123456789+0x0p+01234567890123456789i", 0, nil},
		{"0x0.00p-01234567890123456789+0x0.00p-01234567890123456789i", 0, nil},
		{"-0x0p+01234567890123456789-0x0p+01234567890123456789i", 0, nil},
		{"-0x0.00p-01234567890123456789-0x0.00p-01234567890123456789i", 0, nil},

		{"0.1", 0.1, nil},
		{"0.1i", 0 + 0.1i, nil},
		{"0.123", 0.123, nil},
		{"0.123i", 0 + 0.123i, nil},
		{"0.123+0.123i", 0.123 + 0.123i, nil},
		{"99", 99, nil},
		{"+99", 99, nil},
		{"-99", -99, nil},
		{"+1i", 1i, nil},
		{"i", 0, ErrSyntax},
		{"+i", 0, ErrSyntax},
		{"-1i", -1i, nil},
		{"-i", 0, ErrSyntax},
		{"+3+1i", 3 + 1i, nil},
		{"30+3i", 30 + 3i, nil},
		{"30+3i)", 0, ErrSyntax},
		{"(30+4i", 0, ErrSyntax},
		{"+3e+3-3e+3i", 3e+3 - 3e+3i, nil},
		{"+3e+3+3e+3i", 3e+3 + 3e+3i, nil},
		{"+3e+3+3e+3i+", 0, ErrSyntax},

		// Hexadecimals
		{"0x10.3p-8+0x3p3i",0x10.3p-8+0x3p3i, nil},
		{"+0x10.3p-8+0x3p3i",0x10.3p-8+0x3p3i, nil},
		{"0x10.3p+8-0x3p3i",0x10.3p+8-0x3p3i, nil},
		{"0x1p0", 1, nil},
		{"0x1p1", 2, nil},
		{"0x1p-1", 0.5, nil},
		{"0x1ep-1", 15, nil},
		{"-0x1ep-1", -15, nil},
		{"-0x1_ep-1", -15, nil},
		{"0x1p-200", 6.223015277861142e-61, nil},
		{"0x1p200", 1.6069380442589903e+60, nil},
		{"0x1fFe2.p0", 131042, nil},
		{"0x1fFe2.P0", 131042, nil},
		{"-0x2p3", -16, nil},
		{"0x0.fp4", 15, nil},
		{"0x0.fp0", 0.9375, nil},
		{"0x1e2", 0, ErrSyntax},
		{"1p2", 0, ErrSyntax},
		{"0x1p0i", 1i, nil},
		{"0x1p1i", 2i, nil},
		{"0x1p-1i", 0.5i, nil},
		{"0x1ep-1i", 15i, nil},
		{"-0x1ep-1i", -15i, nil},
		{"-0x1_ep-1i", -15i, nil},
		{"0x1p-200i", 6.223015277861142e-61i, nil},
		{"0x1p200i", 1.6069380442589903e+60i, nil},
		{"0x1fFe2.p0i", 131042i, nil},
		{"0x1fFe2.P0i", 131042i, nil},
		{"-0x2p3i", -16i, nil},
		{"0x0.fp4i", 15i, nil},
		{"0x0.fp0i", 0.9375i, nil},
		{"0x1e2i", 0, ErrSyntax},
		{"1p2i", 0, ErrSyntax},
		{"0x1p0+0x1p0i", 1 + 1i, nil},
		{"0x1p1+0x1p1i", 2 + 2i, nil},
		{"0x1p-1+0x1p-1i", 0.5 + 0.5i, nil},
		{"0x1ep-1+0x1ep-1i", 15 + 15i, nil},
		{"-0x1ep-1-0x1ep-1i", -15 - 15i, nil},
		{"-0x1_ep-1-0x1_ep-1i", -15 - 15i, nil},
		{"0x1p-200+0x1p-200i", 6.223015277861142e-61 + 6.223015277861142e-61i, nil},
		{"0x1p200+0x1p200i", 1.6069380442589903e+60 + 1.6069380442589903e+60i, nil},
		{"0x1fFe2.p0+0x1fFe2.p0i", 131042 + 131042i, nil},
		{"0x1fFe2.P0+0x1fFe2.P0i", 131042 + 131042i, nil},
		{"-0x2p3-0x2p3i", -16 - 16i, nil},
		{"0x0.fp4+0x0.fp4i", 15 + 15i, nil},
		{"0x0.fp0+0x0.fp0i", 0.9375 + 0.9375i, nil},

		// ErrRange

		// next float64 - too large
		{"1.7976931348623159e308+1.7976931348623159e308i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-1.7976931348623159e308-1.7976931348623159e308i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"0x1p1024+0x1p1024i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-0x1p1024-0x1p1024i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"0x2p1023+0x2p1023i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-0x2p1023-0x2p1023i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"0x.1p1028+0x.1p1028i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-0x.1p1028-0x.1p1028i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"0x.2p1027+0x.2p1027i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-0x.2p1027-0x.2p1027i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},

		// the border is ...158079
		// borderline - okay
		{"1.7976931348623158e308+1.7976931348623158e308i", 1.7976931348623157e+308 + 1.7976931348623157e+308i, nil},
		{"-1.7976931348623158e308-1.7976931348623158e308i", -1.7976931348623157e+308 - 1.7976931348623157e+308i, nil},
		{"0x1.fffffffffffff7fffp1023+0x1.fffffffffffff7fffp1023i", 1.7976931348623157e+308 + 1.7976931348623157e+308i, nil},
		{"-0x1.fffffffffffff7fffp1023-0x1.fffffffffffff7fffp1023i", -1.7976931348623157e+308 - 1.7976931348623157e+308i, nil},
		// borderline - too large
		{"1.797693134862315808e308+1.797693134862315808e308i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-1.797693134862315808e308-1.797693134862315808e308i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"0x1.fffffffffffff8p1023+0x1.fffffffffffff8p1023i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-0x1.fffffffffffff8p1023-0x1.fffffffffffff8p1023i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"0x1fffffffffffff.8p+971+0x1fffffffffffff.8p+971i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-0x1fffffffffffff8p+967-0x1fffffffffffff8p+967i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"0x.1fffffffffffff8p1027+0x.1fffffffffffff8p1027i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-0x.1fffffffffffff9p1027-0x.1fffffffffffff9p1027i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},

		// a little too large
		{"1e308+1e308i", 1e+308 + 1e+308i, nil},
		{"2e308", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"1e309", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"0x1p1025", complex(math.Inf(1), math.Inf(1)), ErrRange},

		// way too large
		{"1e310+1e310i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-1e310-1e310i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"1e400+1e400i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-1e400-1e400i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"1e400000+1e400000i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-1e400000-1e400000i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"0x1p1030+0x1p1030i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"0x1p2000+0x1p2000i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"0x1p2000000000+0x1p2000000000i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"-0x1p1030-0x1p1030i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"-0x1p2000-0x1p2000i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},
		{"-0x1p2000000000-0x1p2000000000i", complex(math.Inf(-1), math.Inf(-1)), ErrRange},

		// try to overflow exponent
		{"1e-4294967296+1e-4294967296i", 0, nil},
		{"1e+4294967296+1e+4294967296i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"1e-18446744073709551616+1e-18446744073709551616i", 0, nil},
		{"1e+18446744073709551616+1e+18446744073709551616i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"0x1p-4294967296+0x1p-4294967296i", 0, nil},
		{"0x1p+4294967296+0x1p+4294967296i", complex(math.Inf(1), math.Inf(1)), ErrRange},
		{"0x1p-18446744073709551616+0x1p-18446744073709551616i", 0, nil},
		{"0x1p+18446744073709551616+0x1p+18446744073709551616i", complex(math.Inf(1), math.Inf(1)), ErrRange},
	}

	for _, tt := range tests {
		tt := tt // for capture in Run closures below
		if tt.err != nil {
			tt.err = &NumError{Func: "ParseComplex", Num: tt.in, Err: tt.err}
		}

		t.Run(tt.in, func(t *testing.T) {
			got, err := ParseComplex(tt.in, 128)
			if g, w := err, tt.err; !reflect.DeepEqual(g, w) {
				t.Fatalf("ParseComplex(%q, 128) = %v, %v want %v %v", tt.in, got, err, tt.out, tt.err)
			}

			if !(cmplx.IsNaN(tt.out) && cmplx.IsNaN(got)) && got != tt.out {
				t.Fatalf("ParseComplex(%q, 128) = %v, %v want %v %v", tt.in, got, err, tt.out, tt.err)
			}
		})
	}
}
