// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	. "internal/strconv"
	"math"
	"math/rand"
	"testing"
)

type ftoaTest struct {
	f    float64
	fmt  byte
	prec int
	s    string
}

func fdiv(a, b float64) float64 { return a / b }

const (
	below1e23 = 99999999999999974834176
	above1e23 = 100000000000000008388608
)

var ftoatests = []ftoaTest{
	{1, 'e', 5, "1.00000e+00"},
	{1, 'f', 5, "1.00000"},
	{1, 'g', 5, "1"},
	{1, 'g', -1, "1"},
	{1, 'x', -1, "0x1p+00"},
	{1, 'x', 5, "0x1.00000p+00"},
	{20, 'g', -1, "20"},
	{20, 'x', -1, "0x1.4p+04"},
	{1234567.8, 'g', -1, "1.2345678e+06"},
	{1234567.8, 'x', -1, "0x1.2d687cccccccdp+20"},
	{200000, 'g', -1, "200000"},
	{200000, 'x', -1, "0x1.86ap+17"},
	{200000, 'X', -1, "0X1.86AP+17"},
	{2000000, 'g', -1, "2e+06"},
	{1e10, 'g', -1, "1e+10"},

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
	{0, 'x', 5, "0x0.00000p+00"},

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

	// Round to even
	{1.2345e6, 'e', 3, "1.234e+06"},
	{1.2355e6, 'e', 3, "1.236e+06"},
	{1.2345, 'f', 3, "1.234"},
	{1.2355, 'f', 3, "1.236"},
	{1234567890123456.5, 'e', 15, "1.234567890123456e+15"},
	{1234567890123457.5, 'e', 15, "1.234567890123458e+15"},
	{108678236358137.625, 'g', -1, "1.0867823635813762e+14"},

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

	{fdiv(5e-304, 1e20), 'g', -1, "5e-324"},   // avoid constant arithmetic
	{fdiv(-5e-304, 1e20), 'g', -1, "-5e-324"}, // avoid constant arithmetic

	{32, 'g', -1, "32"},
	{32, 'g', 0, "3e+01"},

	{100, 'x', -1, "0x1.9p+06"},
	{100, 'y', -1, "%y"},

	{math.NaN(), 'g', -1, "NaN"},
	{-math.NaN(), 'g', -1, "NaN"},
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

	// Issue 29491.
	{498484681984085570, 'f', -1, "498484681984085570"},
	{-5.8339553793802237e+23, 'g', -1, "-5.8339553793802237e+23"},

	// Issue 52187
	{123.45, '?', 0, "%?"},
	{123.45, '?', 1, "%?"},
	{123.45, '?', -1, "%?"},

	// rounding
	{2.275555555555555, 'x', -1, "0x1.23456789abcdep+01"},
	{2.275555555555555, 'x', 0, "0x1p+01"},
	{2.275555555555555, 'x', 2, "0x1.23p+01"},
	{2.275555555555555, 'x', 16, "0x1.23456789abcde000p+01"},
	{2.275555555555555, 'x', 21, "0x1.23456789abcde00000000p+01"},
	{2.2755555510520935, 'x', -1, "0x1.2345678p+01"},
	{2.2755555510520935, 'x', 6, "0x1.234568p+01"},
	{2.275555431842804, 'x', -1, "0x1.2345668p+01"},
	{2.275555431842804, 'x', 6, "0x1.234566p+01"},
	{3.999969482421875, 'x', -1, "0x1.ffffp+01"},
	{3.999969482421875, 'x', 4, "0x1.ffffp+01"},
	{3.999969482421875, 'x', 3, "0x1.000p+02"},
	{3.999969482421875, 'x', 2, "0x1.00p+02"},
	{3.999969482421875, 'x', 1, "0x1.0p+02"},
	{3.999969482421875, 'x', 0, "0x1p+02"},

	// Cases that Java once mishandled, from David Chase.
	{1.801439850948199e+16, 'g', -1, "1.801439850948199e+16"},
	{5.960464477539063e-08, 'g', -1, "5.960464477539063e-08"},
	{1.012e-320, 'g', -1, "1.012e-320"},

	// Cases from TestFtoaRandom that caught bugs in fixedFtoa.
	{8177880169308380. * (1 << 1), 'e', 14, "1.63557603386168e+16"},
	{8393378656576888. * (1 << 1), 'e', 15, "1.678675731315378e+16"},
	{8738676561280626. * (1 << 4), 'e', 16, "1.3981882498049002e+17"},
	{8291032395191335. / (1 << 30), 'e', 5, "7.72163e+06"},

	// Exercise divisiblePow5 case in fixedFtoa
	{2384185791015625. * (1 << 12), 'e', 5, "9.76562e+18"},
	{2384185791015625. * (1 << 13), 'e', 5, "1.95312e+19"},
}

func TestFtoa(t *testing.T) {
	for i := 0; i < len(ftoatests); i++ {
		test := &ftoatests[i]
		s := FormatFloat(test.f, test.fmt, test.prec, 64)
		if s != test.s {
			t.Error("testN=64", test.f, string(test.fmt), test.prec, "want", test.s, "got", s)
		}
		x := AppendFloat([]byte("abc"), test.f, test.fmt, test.prec, 64)
		if string(x) != "abc"+test.s {
			t.Error("AppendFloat testN=64", test.f, string(test.fmt), test.prec, "want", "abc"+test.s, "got", string(x))
		}
		if float64(float32(test.f)) == test.f && test.fmt != 'b' {
			test_s := test.s
			if test.f == 5.960464477539063e-08 {
				// This test is an exact float32 but asking for float64 precision in the string.
				// (All our other float64-only tests fail to exactness check above.)
				test_s = "5.9604645e-08"
				continue
			}
			s := FormatFloat(test.f, test.fmt, test.prec, 32)
			if s != test.s {
				t.Error("testN=32", test.f, string(test.fmt), test.prec, "want", test_s, "got", s)
			}
			x := AppendFloat([]byte("abc"), test.f, test.fmt, test.prec, 32)
			if string(x) != "abc"+test_s {
				t.Error("AppendFloat testN=32", test.f, string(test.fmt), test.prec, "want", "abc"+test_s, "got", string(x))
			}
		}
	}
}

func TestFtoaPowersOfTwo(t *testing.T) {
	for exp := -2048; exp <= 2048; exp++ {
		f := math.Ldexp(1, exp)
		if !math.IsInf(f, 0) {
			s := FormatFloat(f, 'e', -1, 64)
			if x, _ := ParseFloat(s, 64); x != f {
				t.Errorf("failed roundtrip %v => %s => %v", f, s, x)
			}
		}
		f32 := float32(f)
		if !math.IsInf(float64(f32), 0) {
			s := FormatFloat(float64(f32), 'e', -1, 32)
			if x, _ := ParseFloat(s, 32); float32(x) != f32 {
				t.Errorf("failed roundtrip %v => %s => %v", f32, s, float32(x))
			}
		}
	}
}

func TestFtoaRandom(t *testing.T) {
	N := int(1e4)
	if testing.Short() {
		N = 100
	}
	t.Logf("testing %d random numbers with fast and slow FormatFloat", N)
	for i := 0; i < N; i++ {
		bits := uint64(rand.Uint32())<<32 | uint64(rand.Uint32())
		x := math.Float64frombits(bits)

		shortFast := FormatFloat(x, 'g', -1, 64)
		SetOptimize(false)
		shortSlow := FormatFloat(x, 'g', -1, 64)
		SetOptimize(true)
		if shortSlow != shortFast {
			t.Errorf("%b printed as %s, want %s", x, shortFast, shortSlow)
		}

		prec := rand.Intn(12) + 5
		shortFast = FormatFloat(x, 'e', prec, 64)
		SetOptimize(false)
		shortSlow = FormatFloat(x, 'e', prec, 64)
		SetOptimize(true)
		if shortSlow != shortFast {
			t.Errorf("%b printed with %%.%de as %s, want %s", x, prec, shortFast, shortSlow)
		}
	}
}

func TestFormatFloatInvalidBitSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic due to invalid bitSize")
		}
	}()
	_ = FormatFloat(3.14, 'g', -1, 100)
}

var ftoaBenches = []struct {
	name    string
	float   float64
	fmt     byte
	prec    int
	bitSize int
}{
	{"Decimal", 33909, 'g', -1, 64},
	{"Float", 339.7784, 'g', -1, 64},
	{"Exp", -5.09e75, 'g', -1, 64},
	{"NegExp", -5.11e-95, 'g', -1, 64},
	{"LongExp", 1.234567890123456e-78, 'g', -1, 64},

	{"Big", 123456789123456789123456789, 'g', -1, 64},
	{"BinaryExp", -1, 'b', -1, 64},

	{"32Integer", 33909, 'g', -1, 32},
	{"32ExactFraction", 3.375, 'g', -1, 32},
	{"32Point", 339.7784, 'g', -1, 32},
	{"32Exp", -5.09e25, 'g', -1, 32},
	{"32NegExp", -5.11e-25, 'g', -1, 32},
	{"32Shortest", 1.234567e-8, 'g', -1, 32},
	{"32Fixed8Hard", math.Ldexp(15961084, -125), 'e', 8, 32},
	{"32Fixed9Hard", math.Ldexp(14855922, -83), 'e', 9, 32},

	{"64Fixed1", 123456, 'e', 3, 64},
	{"64Fixed2", 123.456, 'e', 3, 64},
	{"64Fixed2.5", 1.2345e+06, 'e', 3, 64},
	{"64Fixed3", 1.23456e+78, 'e', 3, 64},
	{"64Fixed4", 1.23456e-78, 'e', 3, 64},
	{"64Fixed5Hard", 4.096e+25, 'e', 5, 64}, // needs divisiblePow5(..., 20)
	{"64Fixed12", 1.23456e-78, 'e', 12, 64},
	{"64Fixed16", 1.23456e-78, 'e', 16, 64},
	// From testdata/testfp.txt
	{"64Fixed12Hard", math.Ldexp(6965949469487146, -249), 'e', 12, 64},
	{"64Fixed17Hard", math.Ldexp(8887055249355788, 665), 'e', 17, 64},
	{"64Fixed18Hard", math.Ldexp(6994187472632449, 690), 'e', 18, 64},

	{"64FixedF1", 123.456, 'f', 6, 64},
	{"64FixedF2", 0.0123, 'f', 6, 64},
	{"64FixedF3", 12.3456, 'f', 2, 64},

	// Trigger slow path (see issue #15672).
	// The shortest is: 8.034137530808823e+43
	{"Slowpath64", 8.03413753080882349e+43, 'e', -1, 64},
	// This denormal is pathological because the lower/upper
	// halfways to neighboring floats are:
	// 622666234635.321003e-320 ~= 622666234635.321e-320
	// 622666234635.321497e-320 ~= 622666234635.3215e-320
	// making it hard to find the 3rd digit
	{"SlowpathDenormal64", 622666234635.3213e-320, 'e', -1, 64},
}

func BenchmarkFormatFloat(b *testing.B) {
	for _, c := range ftoaBenches {
		b.Run(c.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				FormatFloat(c.float, c.fmt, c.prec, c.bitSize)
			}
		})
	}
}

func BenchmarkAppendFloat(b *testing.B) {
	dst := make([]byte, 30)
	for _, c := range ftoaBenches {
		b.Run(c.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				AppendFloat(dst[:0], c.float, c.fmt, c.prec, c.bitSize)
			}
		})
	}
}
