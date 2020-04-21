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

// Test cases required:
// hex form

func TestParseComplex(t *testing.T) {
	tests := []struct {
		str     string
		want    complex128
		wantErr error
	}{
		{
			str:  "99",
			want: complex(99, 0),
		},
		{
			str:  "+99",
			want: complex(99, 0),
		},
		{
			str:  "-99",
			want: complex(-99, 0),
		},
		{
			str:  "+1i",
			want: complex(0, 1),
		},
		{
			str:  "-1i",
			want: complex(0, -1),
		},
		{
			str:  "+3-i",
			want: complex(3, -1),
		},
		{
			str:  "+3+i",
			want: complex(3, 1),
		},
		{
			str:  "3-i",
			want: complex(3, -1),
		},
		{
			str:  "3+i",
			want: complex(3, 1),
		},
		{
			str:  "+i",
			want: complex(0, 1),
		},
		{
			str:  "-i",
			want: complex(0, -1),
		},
		{
			str:  "3e3-i",
			want: complex(3e3, -1),
		},
		{
			str:  "-3e3-i",
			want: complex(-3e3, -1),
		},
		{
			str:  "+3e3-i",
			want: complex(3e3, -1),
		},
		{
			str:  "3e+3-i",
			want: complex(3e+3, -1),
		},
		{
			str:  "-3e+3-i",
			want: complex(-3e+3, -1),
		},
		{
			str:  "-3e+3-i",
			want: complex(-3e+3, -1),
		},
		{
			str:  "+3e+3-3e+3i",
			want: complex(3e+3, -3e+3),
		},
		{
			str:  "+3e+3+3e+3i",
			want: complex(3e+3, 3e+3),
		},
		{
			str:  "Infi",
			want: complex(0, math.Inf(1)),
		},
		{
			str:  "-Infi",
			want: complex(0, math.Inf(-1)),
		},
		{
			str:  "Inf",
			want: complex(math.Inf(1), 0),
		},
		{
			str:  "-Inf",
			want: complex(math.Inf(-1), 0),
		},
		{
			str:  "-Inf-Infi",
			want: complex(math.Inf(-1), math.Inf(-1)),
		},
		{
			str:  "-Inf+Infi",
			want: complex(math.Inf(-1), math.Inf(1)),
		},
		{
			str:  "NaN",
			want: complex(math.NaN(), 0),
		},
		{
			str:  "NaNi",
			want: complex(0, math.NaN()),
		},
		{
			str:  "NaN+NaNi",
			want: complex(math.NaN(), math.NaN()),
		},
		{
			str:  "NaN+NaNi",
			want: complex(math.NaN(), math.NaN()),
		},
		{
			str:  "NaN+NaNi",
			want: complex(math.NaN(), math.NaN()),
		},
		// {
		// 	str:  "0xBadFace+0x677a2fcc40c6i",
		// 	want: complex(0xBadFace, 0x677a2fcc40c6),
		// },
		// {
		// 	str:  "0x10.3p-8+0x3p3i",
		// 	want: complex(0x10.3p-8, 0x3p3),
		// },
		// {
		// 	str:  "+0x10.3p-8+0x3p3i",
		// 	want: complex(+0x10.3p-8, 0x3p3),
		// },
		// {
		// 	str:  "0x10.3p+8-0x3p3i",
		// 	want: complex(0x10.3p+8, -0x3p3),
		// },

		// Malformed cases
		{
			str:     "30+3i)",
			wantErr: ErrSyntax,
		},
		{
			str:     "(30+4i",
			wantErr: ErrSyntax,
		},
		{
			str:     "(",
			wantErr: ErrSyntax,
		},
		{
			str:     ")",
			wantErr: ErrSyntax,
		},
		{
			str:     "foo",
			wantErr: ErrSyntax,
		},
		{
			str:     "10e+10+30i+",
			wantErr: ErrSyntax,
		},
		{
			str:     "10  + 5i",
			wantErr: ErrSyntax,
		},
	}

	for _, tt := range tests {
		tt := tt
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
			tt.str = "(" + tt.str + ")"
			t.Run(tt.str, func(t *testing.T) {
				got, err := ParseComplex(tt.str, 128)
				if g, w := err, tt.wantErr; g != w {
					t.Fatalf("Error mismatch\nGot:  %v\nWant: %v", g, w)
				}

				if !(cmplx.IsNaN(tt.want) && cmplx.IsNaN(got)) && got != tt.want {
					t.Fatalf("Result mismatch\nGot:  %v\nWant: %v", got, tt.want)
				}
			})
		}
	}
}
