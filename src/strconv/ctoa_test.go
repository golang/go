// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	. "strconv"
	"testing"
)

func TestFormatComplex(t *testing.T) {
	tests := []struct {
		c       complex128
		fmt     byte
		prec    int
		bitSize int
		out     string
	}{
		// a variety of signs
		{1 + 2i, 'g', -1, 128, "(1+2i)"},
		{3 - 4i, 'g', -1, 128, "(3-4i)"},
		{-5 + 6i, 'g', -1, 128, "(-5+6i)"},
		{-7 - 8i, 'g', -1, 128, "(-7-8i)"},

		// test that fmt and prec are working
		{3.14159 + 0.00123i, 'e', 3, 128, "(3.142e+00+1.230e-03i)"},
		{3.14159 + 0.00123i, 'f', 3, 128, "(3.142+0.001i)"},
		{3.14159 + 0.00123i, 'g', 3, 128, "(3.14+0.00123i)"},

		// ensure bitSize rounding is working
		{1.2345678901234567 + 9.876543210987654i, 'f', -1, 128, "(1.2345678901234567+9.876543210987654i)"},
		{1.2345678901234567 + 9.876543210987654i, 'f', -1, 64, "(1.2345679+9.876543i)"},

		// other cases are handled by FormatFloat tests
	}
	for _, test := range tests {
		out := FormatComplex(test.c, test.fmt, test.prec, test.bitSize)
		if out != test.out {
			t.Fatalf("FormatComplex(%v, %q, %d, %d) = %q; want %q",
				test.c, test.fmt, test.prec, test.bitSize, out, test.out)
		}
	}
}

func TestFormatComplexInvalidBitSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic due to invalid bitSize")
		}
	}()
	_ = FormatComplex(1+2i, 'g', -1, 100)
}
