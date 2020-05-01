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

func mustFormatComplex(r, i float64) string {
	s1 := FormatFloat(r, 'x', -1, 64)
	s2 := FormatFloat(i, 'x', -1, 64)

	if i >= 0 {
		return s1 + "+" + s2 + "i"
	}

	return s1 + s2 + "i"
}

func TestParseComplex(t *testing.T) {
	tests := []struct {
		str     string
		want    complex128
		wantErr error
	}{
		{"0", complex(0, 0), nil},
		{"0i", complex(0, 0), nil},
		{"0+0i", complex(0, 0), nil},
		{"0.0", complex(0, 0), nil},
		{"0.1", complex(0.1, 0), nil},
		{"0.0i", complex(0, 0), nil},
		{"0.1i", complex(0, 0.1), nil},
		{"0.123", complex(0.123, 0), nil},
		{"0.123i", complex(0, 0.123), nil},
		{"0.123+0.123i", complex(0.123, 0.123), nil},
		{"99", complex(99, 0), nil},
		{"+99", complex(99, 0), nil},
		{"-99", complex(-99, 0), nil},
		{"+1i", complex(0, 1), nil},
		{"-1i", complex(0, -1), nil},
		{"+3-i", complex(3, -1), nil},
		{"+3+i", complex(3, 1), nil},
		{"3-i", complex(3, -1), nil},
		{"3+i", complex(3, 1), nil},
		{"i", complex(0, 1), nil},
		{"+i", complex(0, 1), nil},
		{"-i", complex(0, -1), nil},
		{"3e3-i", complex(3e3, -1), nil},
		{"-3e3-i", complex(-3e3, -1), nil},
		{"+3e3-i", complex(3e3, -1), nil},
		{"3e+3-i", complex(3e+3, -1), nil},
		{"-3e+3-i", complex(-3e+3, -1), nil},
		{"-3e+3-i", complex(-3e+3, -1), nil},
		{"+3e+3-3e+3i", complex(3e+3, -3e+3), nil},
		{"+3e+3+3e+3i", complex(3e+3, 3e+3), nil},
		{"Infi", complex(0, math.Inf(1)), nil},
		{"-Infi", complex(0, math.Inf(-1)), nil},
		{"Inf", complex(math.Inf(1), 0), nil},
		{"-Inf", complex(math.Inf(-1), 0), nil},
		{"-Inf-Infi", complex(math.Inf(-1), math.Inf(-1)), nil},
		{"-Inf+Infi", complex(math.Inf(-1), math.Inf(1)), nil},
		{"NaN", complex(math.NaN(), 0), nil},
		{"NaNi", complex(0, math.NaN()), nil},
		{"NaN+NaNi", complex(math.NaN(), math.NaN()), nil},
		{mustFormatComplex(0xBadFace, 0x677a2fcc40c6), complex(0xBadFace, 0x677a2fcc40c6), nil},
		{"0x10.3p-8+0x3p3i",complex(0x10.3p-8, 0x3p3), nil},
		{"+0x10.3p-8+0x3p3i",complex(0x10.3p-8, 0x3p3), nil},
		{"0x10.3p+8-0x3p3i",complex(0x10.3p+8, -0x3p3), nil},
		{"", 0, ErrSyntax},
		{" ", 0, ErrSyntax},
		{"30+3i)", 0, ErrSyntax},
		{"(30+4i", 0, ErrSyntax},
		{"(", 0, ErrSyntax},
		{")", 0, ErrSyntax},
		{"foo", 0, ErrSyntax},
		{"10e+10+30i+", 0, ErrSyntax},
		{"10  + 5i", 0, ErrSyntax},
		{"+NaN", 0, ErrSyntax},
		{"-NaN", 0, ErrSyntax},
		{"+NaNi", 0, ErrSyntax},
		{"-NaNi", 0, ErrSyntax},
		{"NaN-NaNi", 0, ErrSyntax},
		{"3+3+5.5", 0, ErrSyntax},
		{"3+3+5.5i", 0, ErrSyntax},
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
