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

func TestParseComplex(t *testing.T) {
	tests := []struct {
		str     string
		want    complex128
		wantErr error
	}{
		// Clear Invalids
		{"", 0, ErrSyntax},
		{" ", 0, ErrSyntax},
		{"(", 0, ErrSyntax},
		{")", 0, ErrSyntax},
		{"foo", 0, ErrSyntax},
		{"10  + 5i", 0, ErrSyntax},
		{"3+3+5.5", 0, ErrSyntax},
		{"3+3+5.5i", 0, ErrSyntax},

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

		{"0.1", complex(0.1, 0), nil},
		{"0.1i", complex(0, 0.1), nil},
		{"0.123", complex(0.123, 0), nil},
		{"0.123i", complex(0, 0.123), nil},
		{"0.123+0.123i", complex(0.123, 0.123), nil},
		{"99", complex(99, 0), nil},
		{"+99", complex(99, 0), nil},
		{"-99", complex(-99, 0), nil},
		{"+1i", complex(0, 1), nil},
		{"i", 0, ErrSyntax},
		{"+i", 0, ErrSyntax},
		{"-1i", complex(0, -1), nil},
		{"-i", 0, ErrSyntax},
		{"+3-1i", complex(3, -1), nil},
		{"+3-i", 0, ErrSyntax},
		{"+3+1i", complex(3, 1), nil},
		{"+3+i", 0, ErrSyntax},
		{"3-i", 0, ErrSyntax},
		{"3+i", 0, ErrSyntax},
		{"30+3i", complex(30, 3), nil},
		{"30+3i)", 0, ErrSyntax},
		{"(30+4i", 0, ErrSyntax},
		{"3e3-1i", complex(3e3, -1), nil},
		{"3e3-i", 0, ErrSyntax},
		{"-3e3-1i", complex(-3e3, -1), nil},
		{"-3e3-i", 0, ErrSyntax},
		{"+3e3-1i", complex(3e3, -1), nil},
		{"+3e3-i", 0, ErrSyntax},
		{"3e+3-1i", complex(3e+3, -1), nil},
		{"3e+3-i", 0, ErrSyntax},
		{"-3e+3-1i", complex(-3e+3, -1), nil},
		{"-3e+3-i", 0, ErrSyntax},
		{"-3e+3-i", 0, ErrSyntax},
		{"+3e+3-3e+3i", complex(3e+3, -3e+3), nil},
		{"+3e+3+3e+3i", complex(3e+3, 3e+3), nil},
		{"+3e+3+3e+3i+", 0, ErrSyntax},

		// Hexadecimals
		{mustFormatComplex(0xBadFace, 0x677a2fcc40c6), complex(0xBadFace, 0x677a2fcc40c6), nil},
		{"0x10.3p-8+0x3p3i",complex(0x10.3p-8, 0x3p3), nil},
		{"+0x10.3p-8+0x3p3i",complex(0x10.3p-8, 0x3p3), nil},
		{"0x10.3p+8-0x3p3i",complex(0x10.3p+8, -0x3p3), nil},
		{"0x1p0", complex(1, 0), nil},
		{"0x1p1", complex(2, 0), nil},
		{"0x1p-1", complex(0.5, 0), nil},
		{"0x1ep-1", complex(15, 0), nil},
		{"-0x1ep-1", complex(-15, 0), nil},
		{"-0x1_ep-1", complex(-15, 0), nil},
		{"0x1p-200", complex(6.223015277861142e-61, 0), nil},
		{"0x1p200", complex(1.6069380442589903e+60, 0), nil},
		{"0x1fFe2.p0", complex(131042, 0), nil},
		{"0x1fFe2.P0", complex(131042, 0), nil},
		{"-0x2p3", complex(-16, 0), nil},
		{"0x0.fp4", complex(15, 0), nil},
		{"0x0.fp0", complex(0.9375, 0), nil},
		{"0x1e2", 0, ErrSyntax},
		{"1p2", 0, ErrSyntax},
		{"0x1p0i", complex(0, 1), nil},
		{"0x1p1i", complex(0, 2), nil},
		{"0x1p-1i", complex(0, 0.5), nil},
		{"0x1ep-1i", complex(0, 15), nil},
		{"-0x1ep-1i", complex(0, -15), nil},
		{"-0x1_ep-1i", complex(0, -15), nil},
		{"0x1p-200i", complex(0, 6.223015277861142e-61), nil},
		{"0x1p200i", complex(0, 1.6069380442589903e+60), nil},
		{"0x1fFe2.p0i", complex(0, 131042), nil},
		{"0x1fFe2.P0i", complex(0, 131042), nil},
		{"-0x2p3i", complex(0, -16), nil},
		{"0x0.fp4i", complex(0, 15), nil},
		{"0x0.fp0i", complex(0, 0.9375), nil},
		{"0x1e2i", 0, ErrSyntax},
		{"1p2i", 0, ErrSyntax},
		{"0x1p0+0x1p0i", complex(1, 1), nil},
		{"0x1p1+0x1p1i", complex(2, 2), nil},
		{"0x1p-1+0x1p-1i", complex(0.5, 0.5), nil},
		{"0x1ep-1+0x1ep-1i", complex(15, 15), nil},
		{"-0x1ep-1-0x1ep-1i", complex(-15, -15), nil},
		{"-0x1_ep-1-0x1_ep-1i", complex(-15, -15), nil},
		{"0x1p-200+0x1p-200i", complex(6.223015277861142e-61, 6.223015277861142e-61), nil},
		{"0x1p200+0x1p200i", complex(1.6069380442589903e+60, 1.6069380442589903e+60), nil},
		{"0x1fFe2.p0+0x1fFe2.p0i", complex(131042, 131042), nil},
		{"0x1fFe2.P0+0x1fFe2.P0i", complex(131042, 131042), nil},
		{"-0x2p3-0x2p3i", complex(-16, -16), nil},
		{"0x0.fp4+0x0.fp4i", complex(15, 15), nil},
		{"0x0.fp0+0x0.fp0i", complex(0.9375, 0.9375), nil},
	}

	for _, tt := range tests {
		tt := tt // for capture in Run closures below
		if tt.wantErr != nil {
			tt.wantErr = &NumError{Func: "ParseComplex", Num: tt.str, Err: tt.wantErr}
		}

		t.Run(tt.str, func(t *testing.T) {
			got, err := ParseComplex(tt.str, 128)
			if g, w := err, tt.wantErr; !reflect.DeepEqual(g, w) {
				t.Fatalf("Error mismatch\nGot:  %v\nWant: %v", g, w)
			}

			if !(cmplx.IsNaN(tt.want) && cmplx.IsNaN(got)) && got != tt.want {
				t.Fatalf("Result mismatch\nGot:  %v\nWant: %v", got, tt.want)
			}
		})

		// Test with parentheses
		if tt.wantErr == nil {
			str := "(" + tt.str + ")"

			t.Run(str, func(t *testing.T) {
				got, err := ParseComplex(str, 128)
				if err != nil {
					t.Fatalf("Error mismatch\nGot:  %v\nWant: %v", err, nil)
				}

				if !(cmplx.IsNaN(tt.want) && cmplx.IsNaN(got)) && got != tt.want {
					t.Fatalf("Result mismatch\nGot:  %v\nWant: %v", got, tt.want)
				}
			})
		}
	}
}

func mustFormatComplex(r, i float64) string {
	s1 := FormatFloat(r, 'x', -1, 64)
	s2 := FormatFloat(i, 'x', -1, 64)

	if i >= 0 {
		return s1 + "+" + s2 + "i"
	}

	return s1 + s2 + "i"
}
