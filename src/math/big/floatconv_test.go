// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"testing"
)

var zero_ float64

func TestFloatSetFloat64String(t *testing.T) {
	inf := math.Inf(0)
	nan := math.NaN()

	for _, test := range []struct {
		s string
		x float64 // NaNs represent invalid inputs
	}{
		// basics
		{"0", 0},
		{"-0", -zero_},
		{"+0", 0},
		{"1", 1},
		{"-1", -1},
		{"+1", 1},
		{"1.234", 1.234},
		{"-1.234", -1.234},
		{"+1.234", 1.234},
		{".1", 0.1},
		{"1.", 1},
		{"+1.", 1},

		// various zeros
		{"0e100", 0},
		{"-0e+100", -zero_},
		{"+0e-100", 0},
		{"0E100", 0},
		{"-0E+100", -zero_},
		{"+0E-100", 0},

		// various decimal exponent formats
		{"1.e10", 1e10},
		{"1e+10", 1e10},
		{"+1e-10", 1e-10},
		{"1E10", 1e10},
		{"1.E+10", 1e10},
		{"+1E-10", 1e-10},

		// infinities
		{"Inf", inf},
		{"+Inf", inf},
		{"-Inf", -inf},
		{"inf", inf},
		{"+inf", inf},
		{"-inf", -inf},

		// invalid numbers
		{"", nan},
		{"-", nan},
		{"0x", nan},
		{"0e", nan},
		{"1.2ef", nan},
		{"2..3", nan},
		{"123..", nan},
		{"infinity", nan},
		{"foobar", nan},

		// invalid underscores
		{"_", nan},
		{"0_", nan},
		{"1__0", nan},
		{"123_.", nan},
		{"123._", nan},
		{"123._4", nan},
		{"1_2.3_4_", nan},
		{"_.123", nan},
		{"_123.456", nan},
		{"10._0", nan},
		{"10.0e_0", nan},
		{"10.0e0_", nan},
		{"0P-0__0", nan},

		// misc decimal values
		{"3.14159265", 3.14159265},
		{"-687436.79457e-245", -687436.79457e-245},
		{"-687436.79457E245", -687436.79457e245},
		{".0000000000000000000000000000000000000001", 1e-40},
		{"+10000000000000000000000000000000000000000e-0", 1e40},

		// decimal mantissa, binary exponent
		{"0p0", 0},
		{"-0p0", -zero_},
		{"1p10", 1 << 10},
		{"1p+10", 1 << 10},
		{"+1p-10", 1.0 / (1 << 10)},
		{"1024p-12", 0.25},
		{"-1p10", -1024},
		{"1.5p1", 3},

		// binary mantissa, decimal exponent
		{"0b0", 0},
		{"-0b0", -zero_},
		{"0b0e+10", 0},
		{"-0b0e-10", -zero_},
		{"0b1010", 10},
		{"0B1010E2", 1000},
		{"0b.1", 0.5},
		{"0b.001", 0.125},
		{"0b.001e3", 125},

		// binary mantissa, binary exponent
		{"0b0p+10", 0},
		{"-0b0p-10", -zero_},
		{"0b.1010p4", 10},
		{"0b1p-1", 0.5},
		{"0b001p-3", 0.125},
		{"0b.001p3", 1},
		{"0b0.01p2", 1},
		{"0b0.01P+2", 1},

		// octal mantissa, decimal exponent
		{"0o0", 0},
		{"-0o0", -zero_},
		{"0o0e+10", 0},
		{"-0o0e-10", -zero_},
		{"0o12", 10},
		{"0O12E2", 1000},
		{"0o.4", 0.5},
		{"0o.01", 0.015625},
		{"0o.01e3", 15.625},

		// octal mantissa, binary exponent
		{"0o0p+10", 0},
		{"-0o0p-10", -zero_},
		{"0o.12p6", 10},
		{"0o4p-3", 0.5},
		{"0o0014p-6", 0.1875},
		{"0o.001p9", 1},
		{"0o0.01p7", 2},
		{"0O0.01P+2", 0.0625},

		// hexadecimal mantissa and exponent
		{"0x0", 0},
		{"-0x0", -zero_},
		{"0x0p+10", 0},
		{"-0x0p-10", -zero_},
		{"0xff", 255},
		{"0X.8p1", 1},
		{"-0X0.00008p16", -0.5},
		{"-0X0.00008P+16", -0.5},
		{"0x0.0000000000001p-1022", math.SmallestNonzeroFloat64},
		{"0x1.fffffffffffffp1023", math.MaxFloat64},

		// underscores
		{"0_0", 0},
		{"1_000.", 1000},
		{"1_2_3.4_5_6", 123.456},
		{"1.0e0_0", 1},
		{"1p+1_0", 1024},
		{"0b_1000", 0x8},
		{"0b_1011_1101", 0xbd},
		{"0x_f0_0d_1eP+0_8", 0xf00d1e00},
	} {
		var x Float
		x.SetPrec(53)
		_, ok := x.SetString(test.s)
		if math.IsNaN(test.x) {
			// test.s is invalid
			if ok {
				t.Errorf("%s: want parse error", test.s)
			}
			continue
		}
		// test.s is valid
		if !ok {
			t.Errorf("%s: got parse error", test.s)
			continue
		}
		f, _ := x.Float64()
		want := new(Float).SetFloat64(test.x)
		if x.Cmp(want) != 0 || x.Signbit() != want.Signbit() {
			t.Errorf("%s: got %v (%v); want %v", test.s, &x, f, test.x)
		}
	}
}

func fdiv(a, b float64) float64 { return a / b }

const (
	below1e23 = 99999999999999974834176
	above1e23 = 100000000000000008388608
)

func TestFloat64Text(t *testing.T) {
	for _, test := range []struct {
		x      float64
		format byte
		prec   int
		want   string
	}{
		{0, 'f', 0, "0"},
		{math.Copysign(0, -1), 'f', 0, "-0"},
		{1, 'f', 0, "1"},
		{-1, 'f', 0, "-1"},

		{0.001, 'e', 0, "1e-03"},
		{0.459, 'e', 0, "5e-01"},
		{1.459, 'e', 0, "1e+00"},
		{2.459, 'e', 1, "2.5e+00"},
		{3.459, 'e', 2, "3.46e+00"},
		{4.459, 'e', 3, "4.459e+00"},
		{5.459, 'e', 4, "5.4590e+00"},

		{0.001, 'f', 0, "0"},
		{0.459, 'f', 0, "0"},
		{1.459, 'f', 0, "1"},
		{2.459, 'f', 1, "2.5"},
		{3.459, 'f', 2, "3.46"},
		{4.459, 'f', 3, "4.459"},
		{5.459, 'f', 4, "5.4590"},

		{0, 'b', 0, "0"},
		{math.Copysign(0, -1), 'b', 0, "-0"},
		{1.0, 'b', 0, "4503599627370496p-52"},
		{-1.0, 'b', 0, "-4503599627370496p-52"},
		{4503599627370496, 'b', 0, "4503599627370496p+0"},

		{0, 'p', 0, "0"},
		{math.Copysign(0, -1), 'p', 0, "-0"},
		{1024.0, 'p', 0, "0x.8p+11"},
		{-1024.0, 'p', 0, "-0x.8p+11"},

		// all test cases below from strconv/ftoa_test.go
		{1, 'e', 5, "1.00000e+00"},
		{1, 'f', 5, "1.00000"},
		{1, 'g', 5, "1"},
		{1, 'g', -1, "1"},
		{20, 'g', -1, "20"},
		{1234567.8, 'g', -1, "1.2345678e+06"},
		{200000, 'g', -1, "200000"},
		{2000000, 'g', -1, "2e+06"},

		// g conversion and zero suppression
		{400, 'g', 2, "4e+02"},
		{40, 'g', 2, "40"},
		{4, 'g', 2, "4"},
		{.4, 'g', 2, "0.4"},
		{.04, 'g', 2, "0.04"},
		{.004, 'g', 2, "0.004"},
		{.0004, 'g', 2, "0.0004"},
		{.00004, 'g', 2, "4e-05"},
		{.000004, 'g', 2, "4e-06"},

		{0, 'e', 5, "0.00000e+00"},
		{0, 'f', 5, "0.00000"},
		{0, 'g', 5, "0"},
		{0, 'g', -1, "0"},

		{-1, 'e', 5, "-1.00000e+00"},
		{-1, 'f', 5, "-1.00000"},
		{-1, 'g', 5, "-1"},
		{-1, 'g', -1, "-1"},

		{12, 'e', 5, "1.20000e+01"},
		{12, 'f', 5, "12.00000"},
		{12, 'g', 5, "12"},
		{12, 'g', -1, "12"},

		{123456700, 'e', 5, "1.23457e+08"},
		{123456700, 'f', 5, "123456700.00000"},
		{123456700, 'g', 5, "1.2346e+08"},
		{123456700, 'g', -1, "1.234567e+08"},

		{1.2345e6, 'e', 5, "1.23450e+06"},
		{1.2345e6, 'f', 5, "1234500.00000"},
		{1.2345e6, 'g', 5, "1.2345e+06"},

		{1e23, 'e', 17, "9.99999999999999916e+22"},
		{1e23, 'f', 17, "99999999999999991611392.00000000000000000"},
		{1e23, 'g', 17, "9.9999999999999992e+22"},

		{1e23, 'e', -1, "1e+23"},
		{1e23, 'f', -1, "100000000000000000000000"},
		{1e23, 'g', -1, "1e+23"},

		{below1e23, 'e', 17, "9.99999999999999748e+22"},
		{below1e23, 'f', 17, "99999999999999974834176.00000000000000000"},
		{below1e23, 'g', 17, "9.9999999999999975e+22"},

		{below1e23, 'e', -1, "9.999999999999997e+22"},
		{below1e23, 'f', -1, "99999999999999970000000"},
		{below1e23, 'g', -1, "9.999999999999997e+22"},

		{above1e23, 'e', 17, "1.00000000000000008e+23"},
		{above1e23, 'f', 17, "100000000000000008388608.00000000000000000"},
		{above1e23, 'g', 17, "1.0000000000000001e+23"},

		{above1e23, 'e', -1, "1.0000000000000001e+23"},
		{above1e23, 'f', -1, "100000000000000010000000"},
		{above1e23, 'g', -1, "1.0000000000000001e+23"},

		{5e-304 / 1e20, 'g', -1, "5e-324"},
		{-5e-304 / 1e20, 'g', -1, "-5e-324"},
		{fdiv(5e-304, 1e20), 'g', -1, "5e-324"},   // avoid constant arithmetic
		{fdiv(-5e-304, 1e20), 'g', -1, "-5e-324"}, // avoid constant arithmetic

		{32, 'g', -1, "32"},
		{32, 'g', 0, "3e+01"},

		{100, 'x', -1, "0x1.9p+06"},

		// {math.NaN(), 'g', -1, "NaN"},  // Float doesn't support NaNs
		// {-math.NaN(), 'g', -1, "NaN"}, // Float doesn't support NaNs
		{math.Inf(0), 'g', -1, "+Inf"},
		{math.Inf(-1), 'g', -1, "-Inf"},
		{-math.Inf(0), 'g', -1, "-Inf"},

		{-1, 'b', -1, "-4503599627370496p-52"},

		// fixed bugs
		{0.9, 'f', 1, "0.9"},
		{0.09, 'f', 1, "0.1"},
		{0.0999, 'f', 1, "0.1"},
		{0.05, 'f', 1, "0.1"},
		{0.05, 'f', 0, "0"},
		{0.5, 'f', 1, "0.5"},
		{0.5, 'f', 0, "0"},
		{1.5, 'f', 0, "2"},

		// https://www.exploringbinary.com/java-hangs-when-converting-2-2250738585072012e-308/
		{2.2250738585072012e-308, 'g', -1, "2.2250738585072014e-308"},
		// https://www.exploringbinary.com/php-hangs-on-numeric-value-2-2250738585072011e-308/
		{2.2250738585072011e-308, 'g', -1, "2.225073858507201e-308"},

		// Issue 2625.
		{383260575764816448, 'f', 0, "383260575764816448"},
		{383260575764816448, 'g', -1, "3.8326057576481645e+17"},

		// Issue 15918.
		{1, 'f', -10, "1"},
		{1, 'f', -11, "1"},
		{1, 'f', -12, "1"},
	} {
		// The test cases are from the strconv package which tests float64 values.
		// When formatting values with prec = -1 (shortest representation),
		// the actually available mantissa precision matters.
		// For denormalized values, that precision is < 53 (SetFloat64 default).
		// Compute and set the actual precision explicitly.
		f := new(Float).SetPrec(actualPrec(test.x)).SetFloat64(test.x)
		got := f.Text(test.format, test.prec)
		if got != test.want {
			t.Errorf("%v: got %s; want %s", test, got, test.want)
			continue
		}

		if test.format == 'b' && test.x == 0 {
			continue // 'b' format in strconv.Float requires knowledge of bias for 0.0
		}
		if test.format == 'p' {
			continue // 'p' format not supported in strconv.Format
		}

		// verify that Float format matches strconv format
		want := strconv.FormatFloat(test.x, test.format, test.prec, 64)
		if got != want {
			t.Errorf("%v: got %s; want %s (strconv)", test, got, want)
		}
	}
}

// actualPrec returns the number of actually used mantissa bits.
func actualPrec(x float64) uint {
	if mant := math.Float64bits(x); x != 0 && mant&(0x7ff<<52) == 0 {
		// x is denormalized
		return 64 - uint(bits.LeadingZeros64(mant&(1<<52-1)))
	}
	return 53
}

func TestFloatText(t *testing.T) {
	const defaultRound = ^RoundingMode(0)

	for _, test := range []struct {
		x      string
		round  RoundingMode
		prec   uint
		format byte
		digits int
		want   string
	}{
		{"0", defaultRound, 10, 'f', 0, "0"},
		{"-0", defaultRound, 10, 'f', 0, "-0"},
		{"1", defaultRound, 10, 'f', 0, "1"},
		{"-1", defaultRound, 10, 'f', 0, "-1"},

		{"1.459", defaultRound, 100, 'e', 0, "1e+00"},
		{"2.459", defaultRound, 100, 'e', 1, "2.5e+00"},
		{"3.459", defaultRound, 100, 'e', 2, "3.46e+00"},
		{"4.459", defaultRound, 100, 'e', 3, "4.459e+00"},
		{"5.459", defaultRound, 100, 'e', 4, "5.4590e+00"},

		{"1.459", defaultRound, 100, 'E', 0, "1E+00"},
		{"2.459", defaultRound, 100, 'E', 1, "2.5E+00"},
		{"3.459", defaultRound, 100, 'E', 2, "3.46E+00"},
		{"4.459", defaultRound, 100, 'E', 3, "4.459E+00"},
		{"5.459", defaultRound, 100, 'E', 4, "5.4590E+00"},

		{"1.459", defaultRound, 100, 'f', 0, "1"},
		{"2.459", defaultRound, 100, 'f', 1, "2.5"},
		{"3.459", defaultRound, 100, 'f', 2, "3.46"},
		{"4.459", defaultRound, 100, 'f', 3, "4.459"},
		{"5.459", defaultRound, 100, 'f', 4, "5.4590"},

		{"1.459", defaultRound, 100, 'g', 0, "1"},
		{"2.459", defaultRound, 100, 'g', 1, "2"},
		{"3.459", defaultRound, 100, 'g', 2, "3.5"},
		{"4.459", defaultRound, 100, 'g', 3, "4.46"},
		{"5.459", defaultRound, 100, 'g', 4, "5.459"},

		{"1459", defaultRound, 53, 'g', 0, "1e+03"},
		{"2459", defaultRound, 53, 'g', 1, "2e+03"},
		{"3459", defaultRound, 53, 'g', 2, "3.5e+03"},
		{"4459", defaultRound, 53, 'g', 3, "4.46e+03"},
		{"5459", defaultRound, 53, 'g', 4, "5459"},

		{"1459", defaultRound, 53, 'G', 0, "1E+03"},
		{"2459", defaultRound, 53, 'G', 1, "2E+03"},
		{"3459", defaultRound, 53, 'G', 2, "3.5E+03"},
		{"4459", defaultRound, 53, 'G', 3, "4.46E+03"},
		{"5459", defaultRound, 53, 'G', 4, "5459"},

		{"3", defaultRound, 10, 'e', 40, "3.0000000000000000000000000000000000000000e+00"},
		{"3", defaultRound, 10, 'f', 40, "3.0000000000000000000000000000000000000000"},
		{"3", defaultRound, 10, 'g', 40, "3"},

		{"3e40", defaultRound, 100, 'e', 40, "3.0000000000000000000000000000000000000000e+40"},
		{"3e40", defaultRound, 100, 'f', 4, "30000000000000000000000000000000000000000.0000"},
		{"3e40", defaultRound, 100, 'g', 40, "3e+40"},

		// make sure "stupid" exponents don't stall the machine
		{"1e1000000", defaultRound, 64, 'p', 0, "0x.88b3a28a05eade3ap+3321929"},
		{"1e646456992", defaultRound, 64, 'p', 0, "0x.e883a0c5c8c7c42ap+2147483644"},
		{"1e646456993", defaultRound, 64, 'p', 0, "+Inf"},
		{"1e1000000000", defaultRound, 64, 'p', 0, "+Inf"},
		{"1e-1000000", defaultRound, 64, 'p', 0, "0x.efb4542cc8ca418ap-3321928"},
		{"1e-646456993", defaultRound, 64, 'p', 0, "0x.e17c8956983d9d59p-2147483647"},
		{"1e-646456994", defaultRound, 64, 'p', 0, "0"},
		{"1e-1000000000", defaultRound, 64, 'p', 0, "0"},

		// minimum and maximum values
		{"1p2147483646", defaultRound, 64, 'p', 0, "0x.8p+2147483647"},
		{"0x.8p2147483647", defaultRound, 64, 'p', 0, "0x.8p+2147483647"},
		{"0x.8p-2147483647", defaultRound, 64, 'p', 0, "0x.8p-2147483647"},
		{"1p-2147483649", defaultRound, 64, 'p', 0, "0x.8p-2147483648"},

		// TODO(gri) need tests for actual large Floats

		{"0", defaultRound, 53, 'b', 0, "0"},
		{"-0", defaultRound, 53, 'b', 0, "-0"},
		{"1.0", defaultRound, 53, 'b', 0, "4503599627370496p-52"},
		{"-1.0", defaultRound, 53, 'b', 0, "-4503599627370496p-52"},
		{"4503599627370496", defaultRound, 53, 'b', 0, "4503599627370496p+0"},

		// issue 9939
		{"3", defaultRound, 350, 'b', 0, "1720123961992553633708115671476565205597423741876210842803191629540192157066363606052513914832594264915968p-348"},
		{"03", defaultRound, 350, 'b', 0, "1720123961992553633708115671476565205597423741876210842803191629540192157066363606052513914832594264915968p-348"},
		{"3.", defaultRound, 350, 'b', 0, "1720123961992553633708115671476565205597423741876210842803191629540192157066363606052513914832594264915968p-348"},
		{"3.0", defaultRound, 350, 'b', 0, "1720123961992553633708115671476565205597423741876210842803191629540192157066363606052513914832594264915968p-348"},
		{"3.00", defaultRound, 350, 'b', 0, "1720123961992553633708115671476565205597423741876210842803191629540192157066363606052513914832594264915968p-348"},
		{"3.000", defaultRound, 350, 'b', 0, "1720123961992553633708115671476565205597423741876210842803191629540192157066363606052513914832594264915968p-348"},

		{"3", defaultRound, 350, 'p', 0, "0x.cp+2"},
		{"03", defaultRound, 350, 'p', 0, "0x.cp+2"},
		{"3.", defaultRound, 350, 'p', 0, "0x.cp+2"},
		{"3.0", defaultRound, 350, 'p', 0, "0x.cp+2"},
		{"3.00", defaultRound, 350, 'p', 0, "0x.cp+2"},
		{"3.000", defaultRound, 350, 'p', 0, "0x.cp+2"},

		{"0", defaultRound, 64, 'p', 0, "0"},
		{"-0", defaultRound, 64, 'p', 0, "-0"},
		{"1024.0", defaultRound, 64, 'p', 0, "0x.8p+11"},
		{"-1024.0", defaultRound, 64, 'p', 0, "-0x.8p+11"},

		{"0", defaultRound, 64, 'x', -1, "0x0p+00"},
		{"0", defaultRound, 64, 'x', 0, "0x0p+00"},
		{"0", defaultRound, 64, 'x', 1, "0x0.0p+00"},
		{"0", defaultRound, 64, 'x', 5, "0x0.00000p+00"},
		{"3.25", defaultRound, 64, 'x', 0, "0x1p+02"},
		{"-3.25", defaultRound, 64, 'x', 0, "-0x1p+02"},
		{"3.25", defaultRound, 64, 'x', 1, "0x1.ap+01"},
		{"-3.25", defaultRound, 64, 'x', 1, "-0x1.ap+01"},
		{"3.25", defaultRound, 64, 'x', -1, "0x1.ap+01"},
		{"-3.25", defaultRound, 64, 'x', -1, "-0x1.ap+01"},
		{"1024.0", defaultRound, 64, 'x', 0, "0x1p+10"},
		{"-1024.0", defaultRound, 64, 'x', 0, "-0x1p+10"},
		{"1024.0", defaultRound, 64, 'x', 5, "0x1.00000p+10"},
		{"8191.0", defaultRound, 53, 'x', -1, "0x1.fffp+12"},
		{"8191.5", defaultRound, 53, 'x', -1, "0x1.fff8p+12"},
		{"8191.53125", defaultRound, 53, 'x', -1, "0x1.fff88p+12"},
		{"8191.53125", defaultRound, 53, 'x', 4, "0x1.fff8p+12"},
		{"8191.53125", defaultRound, 53, 'x', 3, "0x1.000p+13"},
		{"8191.53125", defaultRound, 53, 'x', 0, "0x1p+13"},
		{"8191.533203125", defaultRound, 53, 'x', -1, "0x1.fff888p+12"},
		{"8191.533203125", defaultRound, 53, 'x', 5, "0x1.fff88p+12"},
		{"8191.533203125", defaultRound, 53, 'x', 4, "0x1.fff9p+12"},

		{"8191.53125", defaultRound, 53, 'x', -1, "0x1.fff88p+12"},
		{"8191.53125", ToNearestEven, 53, 'x', 5, "0x1.fff88p+12"},
		{"8191.53125", ToNearestAway, 53, 'x', 5, "0x1.fff88p+12"},
		{"8191.53125", ToZero, 53, 'x', 5, "0x1.fff88p+12"},
		{"8191.53125", AwayFromZero, 53, 'x', 5, "0x1.fff88p+12"},
		{"8191.53125", ToNegativeInf, 53, 'x', 5, "0x1.fff88p+12"},
		{"8191.53125", ToPositiveInf, 53, 'x', 5, "0x1.fff88p+12"},

		{"8191.53125", defaultRound, 53, 'x', 4, "0x1.fff8p+12"},
		{"8191.53125", defaultRound, 53, 'x', 3, "0x1.000p+13"},
		{"8191.53125", defaultRound, 53, 'x', 0, "0x1p+13"},
		{"8191.533203125", defaultRound, 53, 'x', -1, "0x1.fff888p+12"},
		{"8191.533203125", defaultRound, 53, 'x', 6, "0x1.fff888p+12"},
		{"8191.533203125", defaultRound, 53, 'x', 5, "0x1.fff88p+12"},
		{"8191.533203125", defaultRound, 53, 'x', 4, "0x1.fff9p+12"},

		{"8191.53125", ToNearestEven, 53, 'x', 4, "0x1.fff8p+12"},
		{"8191.53125", ToNearestAway, 53, 'x', 4, "0x1.fff9p+12"},
		{"8191.53125", ToZero, 53, 'x', 4, "0x1.fff8p+12"},
		{"8191.53125", ToZero, 53, 'x', 2, "0x1.ffp+12"},
		{"8191.53125", AwayFromZero, 53, 'x', 4, "0x1.fff9p+12"},
		{"8191.53125", ToNegativeInf, 53, 'x', 4, "0x1.fff8p+12"},
		{"-8191.53125", ToNegativeInf, 53, 'x', 4, "-0x1.fff9p+12"},
		{"8191.53125", ToPositiveInf, 53, 'x', 4, "0x1.fff9p+12"},
		{"-8191.53125", ToPositiveInf, 53, 'x', 4, "-0x1.fff8p+12"},
	} {
		f, _, err := ParseFloat(test.x, 0, test.prec, ToNearestEven)
		if err != nil {
			t.Errorf("%v: %s", test, err)
			continue
		}
		if test.round != defaultRound {
			f.SetMode(test.round)
		}

		got := f.Text(test.format, test.digits)
		if got != test.want {
			t.Errorf("%v: got %s; want %s", test, got, test.want)
		}

		// compare with strconv.FormatFloat output if possible
		// ('p' format is not supported by strconv.FormatFloat,
		// and its output for 0.0 prints a biased exponent value
		// as in 0p-1074 which makes no sense to emulate here)
		if test.prec == 53 && test.format != 'p' && f.Sign() != 0 && (test.round == ToNearestEven || test.round == defaultRound) {
			f64, acc := f.Float64()
			if acc != Exact {
				t.Errorf("%v: expected exact conversion to float64", test)
				continue
			}
			got := strconv.FormatFloat(f64, test.format, test.digits, 64)
			if got != test.want {
				t.Errorf("%v: got %s; want %s", test, got, test.want)
			}
		}
	}
}

func TestFloatFormat(t *testing.T) {
	for _, test := range []struct {
		format string
		value  interface{} // float32, float64, or string (== 512bit *Float)
		want   string
	}{
		// from fmt/fmt_test.go
		{"%+.3e", 0.0, "+0.000e+00"},
		{"%+.3e", 1.0, "+1.000e+00"},
		{"%+.3f", -1.0, "-1.000"},
		{"%+.3F", -1.0, "-1.000"},
		{"%+.3F", float32(-1.0), "-1.000"},
		{"%+07.2f", 1.0, "+001.00"},
		{"%+07.2f", -1.0, "-001.00"},
		{"%+10.2f", +1.0, "     +1.00"},
		{"%+10.2f", -1.0, "     -1.00"},
		{"% .3E", -1.0, "-1.000E+00"},
		{"% .3e", 1.0, " 1.000e+00"},
		{"%+.3g", 0.0, "+0"},
		{"%+.3g", 1.0, "+1"},
		{"%+.3g", -1.0, "-1"},
		{"% .3g", -1.0, "-1"},
		{"% .3g", 1.0, " 1"},
		{"%b", float32(1.0), "8388608p-23"},
		{"%b", 1.0, "4503599627370496p-52"},

		// from fmt/fmt_test.go: old test/fmt_test.go
		{"%e", 1.0, "1.000000e+00"},
		{"%e", 1234.5678e3, "1.234568e+06"},
		{"%e", 1234.5678e-8, "1.234568e-05"},
		{"%e", -7.0, "-7.000000e+00"},
		{"%e", -1e-9, "-1.000000e-09"},
		{"%f", 1234.5678e3, "1234567.800000"},
		{"%f", 1234.5678e-8, "0.000012"},
		{"%f", -7.0, "-7.000000"},
		{"%f", -1e-9, "-0.000000"},
		{"%g", 1234.5678e3, "1.2345678e+06"},
		{"%g", float32(1234.5678e3), "1.2345678e+06"},
		{"%g", 1234.5678e-8, "1.2345678e-05"},
		{"%g", -7.0, "-7"},
		{"%g", -1e-9, "-1e-09"},
		{"%g", float32(-1e-9), "-1e-09"},
		{"%E", 1.0, "1.000000E+00"},
		{"%E", 1234.5678e3, "1.234568E+06"},
		{"%E", 1234.5678e-8, "1.234568E-05"},
		{"%E", -7.0, "-7.000000E+00"},
		{"%E", -1e-9, "-1.000000E-09"},
		{"%G", 1234.5678e3, "1.2345678E+06"},
		{"%G", float32(1234.5678e3), "1.2345678E+06"},
		{"%G", 1234.5678e-8, "1.2345678E-05"},
		{"%G", -7.0, "-7"},
		{"%G", -1e-9, "-1E-09"},
		{"%G", float32(-1e-9), "-1E-09"},

		{"%20.6e", 1.2345e3, "        1.234500e+03"},
		{"%20.6e", 1.2345e-3, "        1.234500e-03"},
		{"%20e", 1.2345e3, "        1.234500e+03"},
		{"%20e", 1.2345e-3, "        1.234500e-03"},
		{"%20.8e", 1.2345e3, "      1.23450000e+03"},
		{"%20f", 1.23456789e3, "         1234.567890"},
		{"%20f", 1.23456789e-3, "            0.001235"},
		{"%20f", 12345678901.23456789, "  12345678901.234568"},
		{"%-20f", 1.23456789e3, "1234.567890         "},
		{"%20.8f", 1.23456789e3, "       1234.56789000"},
		{"%20.8f", 1.23456789e-3, "          0.00123457"},
		{"%g", 1.23456789e3, "1234.56789"},
		{"%g", 1.23456789e-3, "0.00123456789"},
		{"%g", 1.23456789e20, "1.23456789e+20"},
		{"%20e", math.Inf(1), "                +Inf"},
		{"%-20f", math.Inf(-1), "-Inf                "},

		// from fmt/fmt_test.go: comparison of padding rules with C printf
		{"%.2f", 1.0, "1.00"},
		{"%.2f", -1.0, "-1.00"},
		{"% .2f", 1.0, " 1.00"},
		{"% .2f", -1.0, "-1.00"},
		{"%+.2f", 1.0, "+1.00"},
		{"%+.2f", -1.0, "-1.00"},
		{"%7.2f", 1.0, "   1.00"},
		{"%7.2f", -1.0, "  -1.00"},
		{"% 7.2f", 1.0, "   1.00"},
		{"% 7.2f", -1.0, "  -1.00"},
		{"%+7.2f", 1.0, "  +1.00"},
		{"%+7.2f", -1.0, "  -1.00"},
		{"%07.2f", 1.0, "0001.00"},
		{"%07.2f", -1.0, "-001.00"},
		{"% 07.2f", 1.0, " 001.00"},
		{"% 07.2f", -1.0, "-001.00"},
		{"%+07.2f", 1.0, "+001.00"},
		{"%+07.2f", -1.0, "-001.00"},

		// from fmt/fmt_test.go: zero padding does not apply to infinities
		{"%020f", math.Inf(-1), "                -Inf"},
		{"%020f", math.Inf(+1), "                +Inf"},
		{"% 020f", math.Inf(-1), "                -Inf"},
		{"% 020f", math.Inf(+1), "                 Inf"},
		{"%+020f", math.Inf(-1), "                -Inf"},
		{"%+020f", math.Inf(+1), "                +Inf"},
		{"%20f", -1.0, "           -1.000000"},

		// handle %v like %g
		{"%v", 0.0, "0"},
		{"%v", -7.0, "-7"},
		{"%v", -1e-9, "-1e-09"},
		{"%v", float32(-1e-9), "-1e-09"},
		{"%010v", 0.0, "0000000000"},

		// *Float cases
		{"%.20f", "1e-20", "0.00000000000000000001"},
		{"%.20f", "-1e-20", "-0.00000000000000000001"},
		{"%30.20f", "-1e-20", "       -0.00000000000000000001"},
		{"%030.20f", "-1e-20", "-00000000.00000000000000000001"},
		{"%030.20f", "+1e-20", "000000000.00000000000000000001"},
		{"% 030.20f", "+1e-20", " 00000000.00000000000000000001"},

		// erroneous formats
		{"%s", 1.0, "%!s(*big.Float=1)"},
	} {
		value := new(Float)
		switch v := test.value.(type) {
		case float32:
			value.SetPrec(24).SetFloat64(float64(v))
		case float64:
			value.SetPrec(53).SetFloat64(v)
		case string:
			value.SetPrec(512).Parse(v, 0)
		default:
			t.Fatalf("unsupported test value: %v (%T)", v, v)
		}

		if got := fmt.Sprintf(test.format, value); got != test.want {
			t.Errorf("%v: got %q; want %q", test, got, test.want)
		}
	}
}

func BenchmarkParseFloatSmallExp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for _, s := range []string{
			"1e0",
			"1e-1",
			"1e-2",
			"1e-3",
			"1e-4",
			"1e-5",
			"1e-10",
			"1e-20",
			"1e-50",
			"1e1",
			"1e2",
			"1e3",
			"1e4",
			"1e5",
			"1e10",
			"1e20",
			"1e50",
		} {
			var x Float
			_, _, err := x.Parse(s, 0)
			if err != nil {
				b.Fatalf("%s: %v", s, err)
			}
		}
	}
}

func BenchmarkParseFloatLargeExp(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for _, s := range []string{
			"1e0",
			"1e-10",
			"1e-20",
			"1e-30",
			"1e-40",
			"1e-50",
			"1e-100",
			"1e-500",
			"1e-1000",
			"1e-5000",
			"1e-10000",
			"1e10",
			"1e20",
			"1e30",
			"1e40",
			"1e50",
			"1e100",
			"1e500",
			"1e1000",
			"1e5000",
			"1e10000",
		} {
			var x Float
			_, _, err := x.Parse(s, 0)
			if err != nil {
				b.Fatalf("%s: %v", s, err)
			}
		}
	}
}

func TestFloatScan(t *testing.T) {
	var floatScanTests = []struct {
		input     string
		format    string
		output    string
		remaining int
		wantErr   bool
	}{
		0: {"10.0", "%f", "10", 0, false},
		1: {"23.98+2.0", "%v", "23.98", 4, false},
		2: {"-1+1", "%v", "-1", 2, false},
		3: {" 00000", "%v", "0", 0, false},
		4: {"-123456p-78", "%b", "-4.084816388e-19", 0, false},
		5: {"+123", "%b", "123", 0, false},
		6: {"-1.234e+56", "%e", "-1.234e+56", 0, false},
		7: {"-1.234E-56", "%E", "-1.234e-56", 0, false},
		8: {"-1.234e+567", "%g", "-1.234e+567", 0, false},
		9: {"+1234567891011.234", "%G", "1.234567891e+12", 0, false},

		// Scan doesn't handle ±Inf.
		10: {"Inf", "%v", "", 3, true},
		11: {"-Inf", "%v", "", 3, true},
		12: {"-Inf", "%v", "", 3, true},
	}

	var buf bytes.Buffer
	for i, test := range floatScanTests {
		x := new(Float)
		buf.Reset()
		buf.WriteString(test.input)
		_, err := fmt.Fscanf(&buf, test.format, x)
		if test.wantErr {
			if err == nil {
				t.Errorf("#%d want non-nil err", i)
			}
			continue
		}

		if err != nil {
			t.Errorf("#%d error: %s", i, err)
		}

		if x.String() != test.output {
			t.Errorf("#%d got %s; want %s", i, x.String(), test.output)
		}
		if buf.Len() != test.remaining {
			t.Errorf("#%d got %d bytes remaining; want %d", i, buf.Len(), test.remaining)
		}
	}
}
