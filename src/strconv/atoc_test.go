// Copyright 2009 The Go Authors. All rights reserved.
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

type tcase struct {
	str       string
	expAnswer complex128
	expErr    error
}

func TestParseComplex(t *testing.T) {

	tests := []tcase{
		{
			str:       "99",
			expAnswer: complex(99, 0),
		},
		{
			str:       "+99",
			expAnswer: complex(99, 0),
		},
		{
			str:       "-99",
			expAnswer: complex(-99, 0),
		},
		{
			str:       "+1i",
			expAnswer: complex(0, 1),
		},
		{
			str:       "-1i",
			expAnswer: complex(0, -1),
		},
		{
			str:       "+3-i",
			expAnswer: complex(3, -1),
		},
		{
			str:       "+3+i",
			expAnswer: complex(3, 1),
		},
		{
			str:       "3-i",
			expAnswer: complex(3, -1),
		},
		{
			str:       "3+i",
			expAnswer: complex(3, 1),
		},
		{
			str:       "+i",
			expAnswer: complex(0, 1),
		},
		{
			str:       "-i",
			expAnswer: complex(0, -1),
		},
		{
			str:       "3e3-i",
			expAnswer: complex(3e3, -1),
		},
		{
			str:       "-3e3-i",
			expAnswer: complex(-3e3, -1),
		},
		{
			str:       "+3e3-i",
			expAnswer: complex(3e3, -1),
		},
		{
			str:       "3e+3-i",
			expAnswer: complex(3e+3, -1),
		},
		{
			str:       "-3e+3-i",
			expAnswer: complex(-3e+3, -1),
		},
		{
			str:       "-3e+3-i",
			expAnswer: complex(-3e+3, -1),
		},
		{
			str:       "+3e+3-3e+3i",
			expAnswer: complex(3e+3, -3e+3),
		},
		{
			str:       "+3e+3+3e+3i",
			expAnswer: complex(3e+3, 3e+3),
		},
	}

	for i, tc := range tests {

		got, gotErr := ParseComplex(tc.str, 128)
		if gotErr != nil {
			if tc.expErr == nil {
				t.Errorf("%d: |got: %v |expected: %v", i, gotErr, tc.expErr)
			}
		} else {
			if tc.expErr != nil {
				t.Errorf("%d: |got: %v |expected: %v", i, got, tc.expErr)
			} else {
				if got != tc.expAnswer {
					t.Errorf("%d: |got: %v |expected: %v", i, got, tc.expAnswer)
				}
			}
		}
	}

}
