// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	. "strconv"
	"testing"
)

// Test cases required:
// hex form
// exp form
// Only real
// Only imag
// Both real and imag
// With and without parentheses
// NaN
// ±Inf

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
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.str, func(t *testing.T) {
			got, err := ParseComplex(tt.str, 128)
			if g, w := err, tt.wantErr; g != w {
				t.Fatalf("Error mismatch\nGot:  %v\nWant: %v", g, w)
			}
			if got != tt.want {
				t.Fatalf("Result mismatch\nGot:  %v\nWant: %v", got, tt.want)
			}
		})
	}
}
