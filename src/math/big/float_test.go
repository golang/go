// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func (x *Float) uint64() uint64 {
	u, acc := x.Uint64()
	if acc != Exact {
		panic(fmt.Sprintf("%s is not a uint64", x.Format('g', 10)))
	}
	return u
}

func (x *Float) int64() int64 {
	i, acc := x.Int64()
	if acc != Exact {
		panic(fmt.Sprintf("%s is not an int64", x.Format('g', 10)))
	}
	return i
}

func TestFloatZeroValue(t *testing.T) {
	// zero (uninitialized) value is a ready-to-use 0.0
	var x Float
	if s := x.Format('f', 1); s != "0.0" {
		t.Errorf("zero value = %s; want 0.0", s)
	}

	// zero value has precision 0
	if prec := x.Prec(); prec != 0 {
		t.Errorf("prec = %d; want 0", prec)
	}

	// zero value can be used in any and all positions of binary operations
	make := func(x int) *Float {
		var f Float
		if x != 0 {
			f.SetInt64(int64(x))
		}
		// x == 0 translates into the zero value
		return &f
	}
	for _, test := range []struct {
		z, x, y, want int
		opname        rune
		op            func(z, x, y *Float) *Float
	}{
		{0, 0, 0, 0, '+', (*Float).Add},
		{0, 1, 2, 3, '+', (*Float).Add},
		{1, 2, 0, 2, '+', (*Float).Add},
		{2, 0, 1, 1, '+', (*Float).Add},

		{0, 0, 0, 0, '-', (*Float).Sub},
		{0, 1, 2, -1, '-', (*Float).Sub},
		{1, 2, 0, 2, '-', (*Float).Sub},
		{2, 0, 1, -1, '-', (*Float).Sub},

		{0, 0, 0, 0, '*', (*Float).Mul},
		{0, 1, 2, 2, '*', (*Float).Mul},
		{1, 2, 0, 0, '*', (*Float).Mul},
		{2, 0, 1, 0, '*', (*Float).Mul},

		{0, 0, 0, 0, '/', (*Float).Quo}, // = +Inf
		{0, 2, 1, 2, '/', (*Float).Quo},
		{1, 2, 0, 0, '/', (*Float).Quo}, // = +Inf
		{2, 0, 1, 0, '/', (*Float).Quo},
	} {
		z := make(test.z)
		test.op(z, make(test.x), make(test.y))
		got := 0
		if !z.IsInf(0) {
			got = int(z.int64())
		}
		if got != test.want {
			t.Errorf("%d %c %d = %d; want %d", test.x, test.opname, test.y, got, test.want)
		}
	}

	// TODO(gri) test how precision is set for zero value results
}

func makeFloat(s string) *Float {
	if s == "Inf" || s == "+Inf" {
		return NewInf(+1)
	}
	if s == "-Inf" {
		return NewInf(-1)
	}
	var x Float
	x.SetPrec(1000)
	if _, ok := x.SetString(s); !ok {
		panic(fmt.Sprintf("%q is not a valid float", s))
	}
	return &x
}

func TestFloatSign(t *testing.T) {
	for _, test := range []struct {
		x string
		s int
	}{
		{"-Inf", -1},
		{"-1", -1},
		{"-0", 0},
		{"+0", 0},
		{"+1", +1},
		{"+Inf", +1},
	} {
		x := makeFloat(test.x)
		s := x.Sign()
		if s != test.s {
			t.Errorf("%s.Sign() = %d; want %d", test.x, s, test.s)
		}
	}
}

// feq(x, y) is like x.Cmp(y) == 0 but it also considers the sign of 0 (0 != -0).
func feq(x, y *Float) bool {
	return x.Cmp(y) == 0 && x.neg == y.neg
}

func TestFloatMantExp(t *testing.T) {
	for _, test := range []struct {
		x    string
		frac string
		exp  int
	}{
		{"0", "0", 0},
		{"+0", "0", 0},
		{"-0", "-0", 0},
		{"Inf", "+Inf", 0},
		{"+Inf", "+Inf", 0},
		{"-Inf", "-Inf", 0},
		{"1.5", "0.75", 1},
		{"1.024e3", "0.5", 11},
		{"-0.125", "-0.5", -2},
	} {
		x := makeFloat(test.x)
		frac := makeFloat(test.frac)
		f, e := x.MantExp()
		if !feq(f, frac) || e != test.exp {
			t.Errorf("%s.MantExp() = %s, %d; want %s, %d", test.x, f.Format('g', 10), e, test.frac, test.exp)
		}
	}
}

func TestFloatSetMantExp(t *testing.T) {
	for _, test := range []struct {
		frac string
		exp  int
		z    string
	}{
		{"0", 0, "0"},
		{"+0", 0, "0"},
		{"-0", 0, "-0"},
		{"Inf", 1234, "+Inf"},
		{"+Inf", -1234, "+Inf"},
		{"-Inf", -1234, "-Inf"},
		{"0", -MaxExp - 1, "0"},
		{"1", -MaxExp - 1, "+Inf"},  // exponent magnitude too large
		{"-1", -MaxExp - 1, "-Inf"}, // exponent magnitude too large
		{"0.75", 1, "1.5"},
		{"0.5", 11, "1024"},
		{"-0.5", -2, "-0.125"},
	} {
		frac := makeFloat(test.frac)
		want := makeFloat(test.z)
		var z Float
		z.SetMantExp(frac, test.exp)
		if !feq(&z, want) {
			t.Errorf("SetMantExp(%s, %d) = %s; want %s", test.frac, test.exp, z.Format('g', 10), test.z)
		}
	}
}

func TestFloatIsInt(t *testing.T) {
	for _, test := range []string{
		"0 int",
		"-0 int",
		"1 int",
		"-1 int",
		"0.5",
		"1.23",
		"1.23e1",
		"1.23e2 int",
		"0.000000001e+8",
		"0.000000001e+9 int",
		"1.2345e200 int",
		"Inf",
		"+Inf",
		"-Inf",
	} {
		s := strings.TrimSuffix(test, " int")
		want := s != test
		if got := makeFloat(s).IsInt(); got != want {
			t.Errorf("%s.IsInt() == %t", s, got)
		}
	}
}

func TestFloatIsInf(t *testing.T) {
	// TODO(gri) implement this
}

func fromBinary(s string) int64 {
	x, err := strconv.ParseInt(s, 2, 64)
	if err != nil {
		panic(err)
	}
	return x
}

func toBinary(x int64) string {
	return strconv.FormatInt(x, 2)
}

func testFloatRound(t *testing.T, x, r int64, prec uint, mode RoundingMode) {
	// verify test data
	var ok bool
	switch mode {
	case ToNearestEven, ToNearestAway:
		ok = true // nothing to do for now
	case ToZero:
		if x < 0 {
			ok = r >= x
		} else {
			ok = r <= x
		}
	case AwayFromZero:
		if x < 0 {
			ok = r <= x
		} else {
			ok = r >= x
		}
	case ToNegativeInf:
		ok = r <= x
	case ToPositiveInf:
		ok = r >= x
	default:
		panic("unreachable")
	}
	if !ok {
		t.Fatalf("incorrect test data for prec = %d, %s: x = %s, r = %s", prec, mode, toBinary(x), toBinary(r))
	}

	// compute expected accuracy
	a := Exact
	switch {
	case r < x:
		a = Below
	case r > x:
		a = Above
	}

	// round
	f := new(Float).SetMode(mode).SetInt64(x).SetPrec(prec)

	// check result
	r1 := f.int64()
	p1 := f.Prec()
	a1 := f.Acc()
	if r1 != r || p1 != prec || a1 != a {
		t.Errorf("round %s (%d bits, %s) incorrect: got %s (%d bits, %s); want %s (%d bits, %s)",
			toBinary(x), prec, mode,
			toBinary(r1), p1, a1,
			toBinary(r), prec, a)
		return
	}

	// g and f should be the same
	// (rounding by SetPrec after SetInt64 using default precision
	// should be the same as rounding by SetInt64 after setting the
	// precision)
	g := new(Float).SetMode(mode).SetPrec(prec).SetInt64(x)
	if !feq(g, f) {
		t.Errorf("round %s (%d bits, %s) not symmetric: got %s and %s; want %s",
			toBinary(x), prec, mode,
			toBinary(g.int64()),
			toBinary(r1),
			toBinary(r),
		)
		return
	}

	// h and f should be the same
	// (repeated rounding should be idempotent)
	h := new(Float).SetMode(mode).SetPrec(prec).Set(f)
	if !feq(h, f) {
		t.Errorf("round %s (%d bits, %s) not idempotent: got %s and %s; want %s",
			toBinary(x), prec, mode,
			toBinary(h.int64()),
			toBinary(r1),
			toBinary(r),
		)
		return
	}
}

// TestFloatRound tests basic rounding.
func TestFloatRound(t *testing.T) {
	for _, test := range []struct {
		prec                        uint
		x, zero, neven, naway, away string // input, results rounded to prec bits
	}{
		{5, "1000", "1000", "1000", "1000", "1000"},
		{5, "1001", "1001", "1001", "1001", "1001"},
		{5, "1010", "1010", "1010", "1010", "1010"},
		{5, "1011", "1011", "1011", "1011", "1011"},
		{5, "1100", "1100", "1100", "1100", "1100"},
		{5, "1101", "1101", "1101", "1101", "1101"},
		{5, "1110", "1110", "1110", "1110", "1110"},
		{5, "1111", "1111", "1111", "1111", "1111"},

		{4, "1000", "1000", "1000", "1000", "1000"},
		{4, "1001", "1001", "1001", "1001", "1001"},
		{4, "1010", "1010", "1010", "1010", "1010"},
		{4, "1011", "1011", "1011", "1011", "1011"},
		{4, "1100", "1100", "1100", "1100", "1100"},
		{4, "1101", "1101", "1101", "1101", "1101"},
		{4, "1110", "1110", "1110", "1110", "1110"},
		{4, "1111", "1111", "1111", "1111", "1111"},

		{3, "1000", "1000", "1000", "1000", "1000"},
		{3, "1001", "1000", "1000", "1010", "1010"},
		{3, "1010", "1010", "1010", "1010", "1010"},
		{3, "1011", "1010", "1100", "1100", "1100"},
		{3, "1100", "1100", "1100", "1100", "1100"},
		{3, "1101", "1100", "1100", "1110", "1110"},
		{3, "1110", "1110", "1110", "1110", "1110"},
		{3, "1111", "1110", "10000", "10000", "10000"},

		{3, "1000001", "1000000", "1000000", "1000000", "1010000"},
		{3, "1001001", "1000000", "1010000", "1010000", "1010000"},
		{3, "1010001", "1010000", "1010000", "1010000", "1100000"},
		{3, "1011001", "1010000", "1100000", "1100000", "1100000"},
		{3, "1100001", "1100000", "1100000", "1100000", "1110000"},
		{3, "1101001", "1100000", "1110000", "1110000", "1110000"},
		{3, "1110001", "1110000", "1110000", "1110000", "10000000"},
		{3, "1111001", "1110000", "10000000", "10000000", "10000000"},

		{2, "1000", "1000", "1000", "1000", "1000"},
		{2, "1001", "1000", "1000", "1000", "1100"},
		{2, "1010", "1000", "1000", "1100", "1100"},
		{2, "1011", "1000", "1100", "1100", "1100"},
		{2, "1100", "1100", "1100", "1100", "1100"},
		{2, "1101", "1100", "1100", "1100", "10000"},
		{2, "1110", "1100", "10000", "10000", "10000"},
		{2, "1111", "1100", "10000", "10000", "10000"},

		{2, "1000001", "1000000", "1000000", "1000000", "1100000"},
		{2, "1001001", "1000000", "1000000", "1000000", "1100000"},
		{2, "1010001", "1000000", "1100000", "1100000", "1100000"},
		{2, "1011001", "1000000", "1100000", "1100000", "1100000"},
		{2, "1100001", "1100000", "1100000", "1100000", "10000000"},
		{2, "1101001", "1100000", "1100000", "1100000", "10000000"},
		{2, "1110001", "1100000", "10000000", "10000000", "10000000"},
		{2, "1111001", "1100000", "10000000", "10000000", "10000000"},

		{1, "1000", "1000", "1000", "1000", "1000"},
		{1, "1001", "1000", "1000", "1000", "10000"},
		{1, "1010", "1000", "1000", "1000", "10000"},
		{1, "1011", "1000", "1000", "1000", "10000"},
		{1, "1100", "1000", "10000", "10000", "10000"},
		{1, "1101", "1000", "10000", "10000", "10000"},
		{1, "1110", "1000", "10000", "10000", "10000"},
		{1, "1111", "1000", "10000", "10000", "10000"},

		{1, "1000001", "1000000", "1000000", "1000000", "10000000"},
		{1, "1001001", "1000000", "1000000", "1000000", "10000000"},
		{1, "1010001", "1000000", "1000000", "1000000", "10000000"},
		{1, "1011001", "1000000", "1000000", "1000000", "10000000"},
		{1, "1100001", "1000000", "10000000", "10000000", "10000000"},
		{1, "1101001", "1000000", "10000000", "10000000", "10000000"},
		{1, "1110001", "1000000", "10000000", "10000000", "10000000"},
		{1, "1111001", "1000000", "10000000", "10000000", "10000000"},
	} {
		x := fromBinary(test.x)
		z := fromBinary(test.zero)
		e := fromBinary(test.neven)
		n := fromBinary(test.naway)
		a := fromBinary(test.away)
		prec := test.prec

		testFloatRound(t, x, z, prec, ToZero)
		testFloatRound(t, x, e, prec, ToNearestEven)
		testFloatRound(t, x, n, prec, ToNearestAway)
		testFloatRound(t, x, a, prec, AwayFromZero)

		testFloatRound(t, x, z, prec, ToNegativeInf)
		testFloatRound(t, x, a, prec, ToPositiveInf)

		testFloatRound(t, -x, -a, prec, ToNegativeInf)
		testFloatRound(t, -x, -z, prec, ToPositiveInf)
	}
}

// TestFloatRound24 tests that rounding a float64 to 24 bits
// matches IEEE-754 rounding to nearest when converting a
// float64 to a float32.
func TestFloatRound24(t *testing.T) {
	const x0 = 1<<26 - 0x10 // 11...110000 (26 bits)
	for d := 0; d <= 0x10; d++ {
		x := float64(x0 + d)
		f := new(Float).SetPrec(24).SetFloat64(x)
		got, _ := f.Float64()
		want := float64(float32(x))
		if got != want {
			t.Errorf("Round(%g, 24) = %g; want %g", x, got, want)
		}
	}
}

func TestFloatSetUint64(t *testing.T) {
	for _, want := range []uint64{
		0,
		1,
		2,
		10,
		100,
		1<<32 - 1,
		1 << 32,
		1<<64 - 1,
	} {
		var f Float
		f.SetUint64(want)
		if got := f.uint64(); got != want {
			t.Errorf("got %#x (%s); want %#x", got, f.Format('p', 0), want)
		}
	}

	// test basic rounding behavior (exhaustive rounding testing is done elsewhere)
	const x uint64 = 0x8765432187654321 // 64 bits needed
	for prec := uint(1); prec <= 64; prec++ {
		f := new(Float).SetPrec(prec).SetMode(ToZero).SetUint64(x)
		got := f.uint64()
		want := x &^ (1<<(64-prec) - 1) // cut off (round to zero) low 64-prec bits
		if got != want {
			t.Errorf("got %#x (%s); want %#x", got, f.Format('p', 0), want)
		}
	}
}

func TestFloatSetInt64(t *testing.T) {
	for _, want := range []int64{
		0,
		1,
		2,
		10,
		100,
		1<<32 - 1,
		1 << 32,
		1<<63 - 1,
	} {
		for i := range [2]int{} {
			if i&1 != 0 {
				want = -want
			}
			var f Float
			f.SetInt64(want)
			if got := f.int64(); got != want {
				t.Errorf("got %#x (%s); want %#x", got, f.Format('p', 0), want)
			}
		}
	}

	// test basic rounding behavior (exhaustive rounding testing is done elsewhere)
	const x int64 = 0x7654321076543210 // 63 bits needed
	for prec := uint(1); prec <= 63; prec++ {
		f := new(Float).SetPrec(prec).SetMode(ToZero).SetInt64(x)
		got := f.int64()
		want := x &^ (1<<(63-prec) - 1) // cut off (round to zero) low 63-prec bits
		if got != want {
			t.Errorf("got %#x (%s); want %#x", got, f.Format('p', 0), want)
		}
	}
}

func TestFloatSetFloat64(t *testing.T) {
	for _, want := range []float64{
		0,
		1,
		2,
		12345,
		1e10,
		1e100,
		3.14159265e10,
		2.718281828e-123,
		1.0 / 3,
		math.Inf(-1),
		math.Inf(0),
		-math.Inf(1),
	} {
		for i := range [2]int{} {
			if i&1 != 0 {
				want = -want
			}
			var f Float
			f.SetFloat64(want)
			if got, _ := f.Float64(); got != want {
				t.Errorf("got %g (%s); want %g", got, f.Format('p', 0), want)
			}
		}
	}

	// test basic rounding behavior (exhaustive rounding testing is done elsewhere)
	const x uint64 = 0x8765432143218 // 53 bits needed
	for prec := uint(1); prec <= 52; prec++ {
		f := new(Float).SetPrec(prec).SetMode(ToZero).SetFloat64(float64(x))
		got, _ := f.Float64()
		want := float64(x &^ (1<<(52-prec) - 1)) // cut off (round to zero) low 53-prec bits
		if got != want {
			t.Errorf("got %g (%s); want %g", got, f.Format('p', 0), want)
		}
	}
}

func TestFloatSetInt(t *testing.T) {
	for _, want := range []string{
		"0",
		"1",
		"-1",
		"1234567890",
		"123456789012345678901234567890",
		"123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
	} {
		var x Int
		_, ok := x.SetString(want, 0)
		if !ok {
			t.Errorf("invalid integer %s", want)
			continue
		}
		n := x.BitLen()

		var f Float
		f.SetInt(&x)

		// check precision
		if n < 64 {
			n = 64
		}
		if prec := f.Prec(); prec != uint(n) {
			t.Errorf("got prec = %d; want %d", prec, n)
		}

		// check value
		got := f.Format('g', 100)
		if got != want {
			t.Errorf("got %s (%s); want %s", got, f.Format('p', 0), want)
		}
	}

	// TODO(gri) test basic rounding behavior
}

func TestFloatSetRat(t *testing.T) {
	for _, want := range []string{
		"0",
		"1",
		"-1",
		"1234567890",
		"123456789012345678901234567890",
		"123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
		"1.2",
		"3.14159265",
		// TODO(gri) expand
	} {
		var x Rat
		_, ok := x.SetString(want)
		if !ok {
			t.Errorf("invalid fraction %s", want)
			continue
		}
		n := max(x.Num().BitLen(), x.Denom().BitLen())

		var f1, f2 Float
		f2.SetPrec(1000)
		f1.SetRat(&x)
		f2.SetRat(&x)

		// check precision when set automatically
		if n < 64 {
			n = 64
		}
		if prec := f1.Prec(); prec != uint(n) {
			t.Errorf("got prec = %d; want %d", prec, n)
		}

		got := f2.Format('g', 100)
		if got != want {
			t.Errorf("got %s (%s); want %s", got, f2.Format('p', 0), want)
		}
	}
}

func TestFloatUint64(t *testing.T) {
	for _, test := range []struct {
		x   string
		out uint64
		acc Accuracy
	}{
		{"-Inf", 0, Above},
		{"-1", 0, Above},
		{"-1e-1000", 0, Above},
		{"-0", 0, Exact},
		{"0", 0, Exact},
		{"1e-1000", 0, Below},
		{"1", 1, Exact},
		{"1.000000000000000000001", 1, Below},
		{"12345.0", 12345, Exact},
		{"12345.000000000000000000001", 12345, Below},
		{"18446744073709551615", 18446744073709551615, Exact},
		{"18446744073709551615.000000000000000000001", math.MaxUint64, Below},
		{"18446744073709551616", math.MaxUint64, Below},
		{"1e10000", math.MaxUint64, Below},
		{"+Inf", math.MaxUint64, Below},
	} {
		x := makeFloat(test.x)
		out, acc := x.Uint64()
		if out != test.out || acc != test.acc {
			t.Errorf("%s: got %d (%s); want %d (%s)", test.x, out, acc, test.out, test.acc)
		}
	}
}

func TestFloatInt64(t *testing.T) {
	for _, test := range []struct {
		x   string
		out int64
		acc Accuracy
	}{
		{"-Inf", math.MinInt64, Above},
		{"-1e10000", math.MinInt64, Above},
		{"-9223372036854775809", math.MinInt64, Above},
		{"-9223372036854775808.000000000000000000001", math.MinInt64, Above},
		{"-9223372036854775808", -9223372036854775808, Exact},
		{"-9223372036854775807.000000000000000000001", -9223372036854775807, Above},
		{"-9223372036854775807", -9223372036854775807, Exact},
		{"-12345.000000000000000000001", -12345, Above},
		{"-12345.0", -12345, Exact},
		{"-1.000000000000000000001", -1, Above},
		{"-1", -1, Exact},
		{"-1e-1000", 0, Above},
		{"0", 0, Exact},
		{"1e-1000", 0, Below},
		{"1", 1, Exact},
		{"1.000000000000000000001", 1, Below},
		{"12345.0", 12345, Exact},
		{"12345.000000000000000000001", 12345, Below},
		{"9223372036854775807", 9223372036854775807, Exact},
		{"9223372036854775807.000000000000000000001", math.MaxInt64, Below},
		{"9223372036854775808", math.MaxInt64, Below},
		{"1e10000", math.MaxInt64, Below},
		{"+Inf", math.MaxInt64, Below},
	} {
		x := makeFloat(test.x)
		out, acc := x.Int64()
		if out != test.out || acc != test.acc {
			t.Errorf("%s: got %d (%s); want %d (%s)", test.x, out, acc, test.out, test.acc)
		}
	}
}

func TestFloatInt(t *testing.T) {
	for _, test := range []struct {
		x   string
		out string
		acc Accuracy
	}{
		{"0", "0", Exact},
		{"+0", "0", Exact},
		{"-0", "0", Exact},
		{"Inf", "nil", Below},
		{"+Inf", "nil", Below},
		{"-Inf", "nil", Above},
		{"1", "1", Exact},
		{"-1", "-1", Exact},
		{"1.23", "1", Below},
		{"-1.23", "-1", Above},
		{"123e-2", "1", Below},
		{"123e-3", "0", Below},
		{"123e-4", "0", Below},
		{"1e-1000", "0", Below},
		{"-1e-1000", "0", Above},
		{"1e+10", "10000000000", Exact},
		{"1e+100", "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", Exact},
	} {
		x := makeFloat(test.x)
		out, acc := x.Int()
		got := "nil"
		if out != nil {
			got = out.String()
		}
		if got != test.out || acc != test.acc {
			t.Errorf("%s: got %s (%s); want %s (%s)", test.x, got, acc, test.out, test.acc)
		}
	}
}

func TestFloatRat(t *testing.T) {
	// TODO(gri) implement this
}

func TestFloatAbs(t *testing.T) {
	for _, test := range []string{
		"0",
		"1",
		"1234",
		"1.23e-2",
		"1e-1000",
		"1e1000",
		"Inf",
	} {
		p := makeFloat(test)
		a := new(Float).Abs(p)
		if !feq(a, p) {
			t.Errorf("%s: got %s; want %s", test, a.Format('g', 10), test)
		}

		n := makeFloat("-" + test)
		a.Abs(n)
		if !feq(a, p) {
			t.Errorf("-%s: got %s; want %s", test, a.Format('g', 10), test)
		}
	}
}

func TestFloatNeg(t *testing.T) {
	for _, test := range []string{
		"0",
		"1",
		"1234",
		"1.23e-2",
		"1e-1000",
		"1e1000",
		"Inf",
	} {
		p1 := makeFloat(test)
		n1 := makeFloat("-" + test)
		n2 := new(Float).Neg(p1)
		p2 := new(Float).Neg(n2)
		if !feq(n2, n1) {
			t.Errorf("%s: got %s; want %s", test, n2.Format('g', 10), n1.Format('g', 10))
		}
		if !feq(p2, p1) {
			t.Errorf("%s: got %s; want %s", test, p2.Format('g', 10), p1.Format('g', 10))
		}
	}
}

func TestFloatInc(t *testing.T) {
	const n = 10
	for _, prec := range precList {
		if 1<<prec < n {
			continue // prec must be large enough to hold all numbers from 0 to n
		}
		var x, one Float
		x.SetPrec(prec)
		one.SetInt64(1)
		for i := 0; i < n; i++ {
			x.Add(&x, &one)
		}
		if x.Cmp(new(Float).SetInt64(n)) != 0 {
			t.Errorf("prec = %d: got %s; want %d", prec, &x, n)
		}
	}
}

// Selected precisions with which to run various tests.
var precList = [...]uint{1, 2, 5, 8, 10, 16, 23, 24, 32, 50, 53, 64, 100, 128, 500, 511, 512, 513, 1000, 10000}

// Selected bits with which to run various tests.
// Each entry is a list of bits representing a floating-point number (see fromBits).
var bitsList = [...][]int{
	{},           // = 0
	{0},          // = 1
	{1},          // = 2
	{-1},         // = 1/2
	{10},         // = 2**10 == 1024
	{-10},        // = 2**-10 == 1/1024
	{100, 10, 1}, // = 2**100 + 2**10 + 2**1
	{0, -1, -2, -10},
	// TODO(gri) add more test cases
}

// TestFloatAdd tests Float.Add/Sub by comparing the result of a "manual"
// addition/subtraction of arguments represented by bits lists with the
// respective floating-point addition/subtraction for a variety of precisions
// and rounding modes.
func TestFloatAdd(t *testing.T) {
	for _, xbits := range bitsList {
		for _, ybits := range bitsList {
			// exact values
			x := fromBits(xbits...)
			y := fromBits(ybits...)
			zbits := append(xbits, ybits...)
			z := fromBits(zbits...)

			for i, mode := range [...]RoundingMode{ToZero, ToNearestEven, AwayFromZero} {
				for _, prec := range precList {
					got := new(Float).SetPrec(prec).SetMode(mode)
					got.Add(x, y)
					want := roundBits(zbits, prec, mode)
					if got.Cmp(want) != 0 {
						t.Errorf("i = %d, prec = %d, %s:\n\t     %s %v\n\t+    %s %v\n\t=    %s\n\twant %s",
							i, prec, mode, x, xbits, y, ybits, got, want)
						return
					}

					got.Sub(z, x)
					want = roundBits(ybits, prec, mode)
					if got.Cmp(want) != 0 {
						t.Errorf("i = %d, prec = %d, %s:\n\t     %s %v\n\t-    %s %v\n\t=    %s\n\twant %s",
							i, prec, mode, z, zbits, x, xbits, got, want)
					}
				}
			}
		}
	}
}

// TestFloatAdd32 tests that Float.Add/Sub of numbers with
// 24bit mantissa behaves like float32 addition/subtraction.
func TestFloatAdd32(t *testing.T) {
	// TODO(gri) fix test for 32bit platforms
	if _W == 32 {
		return
	}

	// chose base such that we cross the mantissa precision limit
	const base = 1<<26 - 0x10 // 11...110000 (26 bits)
	for d := 0; d <= 0x10; d++ {
		for i := range [2]int{} {
			x0, y0 := float64(base), float64(d)
			if i&1 != 0 {
				x0, y0 = y0, x0
			}

			x := new(Float).SetFloat64(x0)
			y := new(Float).SetFloat64(y0)
			z := new(Float).SetPrec(24)

			z.Add(x, y)
			got, acc := z.Float64()
			want := float64(float32(y0) + float32(x0))
			if got != want || acc != Exact {
				t.Errorf("d = %d: %g + %g = %g (%s); want %g exactly", d, x0, y0, got, acc, want)
			}

			z.Sub(z, y)
			got, acc = z.Float64()
			want = float64(float32(want) - float32(y0))
			if got != want || acc != Exact {
				t.Errorf("d = %d: %g - %g = %g (%s); want %g exactly", d, x0+y0, y0, got, acc, want)
			}
		}
	}
}

// TestFloatAdd64 tests that Float.Add/Sub of numbers with
// 53bit mantissa behaves like float64 addition/subtraction.
func TestFloatAdd64(t *testing.T) {
	// chose base such that we cross the mantissa precision limit
	const base = 1<<55 - 0x10 // 11...110000 (55 bits)
	for d := 0; d <= 0x10; d++ {
		for i := range [2]int{} {
			x0, y0 := float64(base), float64(d)
			if i&1 != 0 {
				x0, y0 = y0, x0
			}

			x := new(Float).SetFloat64(x0)
			y := new(Float).SetFloat64(y0)
			z := new(Float).SetPrec(53)

			z.Add(x, y)
			got, acc := z.Float64()
			want := x0 + y0
			if got != want || acc != Exact {
				t.Errorf("d = %d: %g + %g = %g (%s); want %g exactly", d, x0, y0, got, acc, want)
			}

			z.Sub(z, y)
			got, acc = z.Float64()
			want -= y0
			if got != want || acc != Exact {
				t.Errorf("d = %d: %g - %g = %g (%s); want %g exactly", d, x0+y0, y0, got, acc, want)
			}
		}
	}
}

func TestFloatMul(t *testing.T) {
	// TODO(gri) implement this
}

// TestFloatMul64 tests that Float.Mul/Quo of numbers with
// 53bit mantissa behaves like float64 multiplication/division.
func TestFloatMul64(t *testing.T) {
	for _, test := range []struct {
		x, y float64
	}{
		{0, 0},
		{0, 1},
		{1, 1},
		{1, 1.5},
		{1.234, 0.5678},
		{2.718281828, 3.14159265358979},
		{2.718281828e10, 3.14159265358979e-32},
		{1.0 / 3, 1e200},
	} {
		for i := range [8]int{} {
			x0, y0 := test.x, test.y
			if i&1 != 0 {
				x0 = -x0
			}
			if i&2 != 0 {
				y0 = -y0
			}
			if i&4 != 0 {
				x0, y0 = y0, x0
			}

			x := new(Float).SetFloat64(x0)
			y := new(Float).SetFloat64(y0)
			z := new(Float).SetPrec(53)

			z.Mul(x, y)
			got, _ := z.Float64()
			want := x0 * y0
			if got != want {
				t.Errorf("%g * %g = %g; want %g", x0, y0, got, want)
			}

			if y0 == 0 {
				continue // avoid division-by-zero
			}
			z.Quo(z, y)
			got, _ = z.Float64()
			want /= y0
			if got != want {
				t.Errorf("%g / %g = %g; want %g", x0*y0, y0, got, want)
			}
		}
	}
}

func TestIssue6866(t *testing.T) {
	for _, prec := range precList {
		two := new(Float).SetPrec(prec).SetInt64(2)
		one := new(Float).SetPrec(prec).SetInt64(1)
		three := new(Float).SetPrec(prec).SetInt64(3)
		msix := new(Float).SetPrec(prec).SetInt64(-6)
		psix := new(Float).SetPrec(prec).SetInt64(+6)

		p := new(Float).SetPrec(prec)
		z1 := new(Float).SetPrec(prec)
		z2 := new(Float).SetPrec(prec)

		// z1 = 2 + 1.0/3*-6
		p.Quo(one, three)
		p.Mul(p, msix)
		z1.Add(two, p)

		// z2 = 2 - 1.0/3*+6
		p.Quo(one, three)
		p.Mul(p, psix)
		z2.Sub(two, p)

		if z1.Cmp(z2) != 0 {
			t.Fatalf("prec %d: got z1 = %s != z2 = %s; want z1 == z2\n", prec, z1, z2)
		}
		if z1.Sign() != 0 {
			t.Errorf("prec %d: got z1 = %s; want 0", prec, z1)
		}
		if z2.Sign() != 0 {
			t.Errorf("prec %d: got z2 = %s; want 0", prec, z2)
		}
	}
}

func TestFloatQuo(t *testing.T) {
	// TODO(gri) make the test vary these precisions
	preci := 200 // precision of integer part
	precf := 20  // precision of fractional part

	for i := 0; i < 8; i++ {
		// compute accurate (not rounded) result z
		bits := []int{preci - 1}
		if i&3 != 0 {
			bits = append(bits, 0)
		}
		if i&2 != 0 {
			bits = append(bits, -1)
		}
		if i&1 != 0 {
			bits = append(bits, -precf)
		}
		z := fromBits(bits...)

		// compute accurate x as z*y
		y := new(Float).SetFloat64(3.14159265358979323e123)

		x := new(Float).SetPrec(z.Prec() + y.Prec()).SetMode(ToZero)
		x.Mul(z, y)

		// leave for debugging
		// fmt.Printf("x = %s\ny = %s\nz = %s\n", x, y, z)

		if got := x.Acc(); got != Exact {
			t.Errorf("got acc = %s; want exact", got)
		}

		// round accurate z for a variety of precisions and
		// modes and compare against result of x / y.
		for _, mode := range [...]RoundingMode{ToZero, ToNearestEven, AwayFromZero} {
			for d := -5; d < 5; d++ {
				prec := uint(preci + d)
				got := new(Float).SetPrec(prec).SetMode(mode).Quo(x, y)
				want := roundBits(bits, prec, mode)
				if got.Cmp(want) != 0 {
					t.Errorf("i = %d, prec = %d, %s:\n\t     %s\n\t/    %s\n\t=    %s\n\twant %s",
						i, prec, mode, x, y, got, want)
				}
			}
		}
	}
}

// TestFloatQuoSmoke tests all divisions x/y for values x, y in the range [-n, +n];
// it serves as a smoke test for basic correctness of division.
func TestFloatQuoSmoke(t *testing.T) {
	n := 1000
	if testing.Short() {
		n = 10
	}

	const dprec = 3         // max. precision variation
	const prec = 10 + dprec // enough bits to hold n precisely
	for x := -n; x <= n; x++ {
		for y := -n; y < n; y++ {
			if y == 0 {
				continue
			}

			a := float64(x)
			b := float64(y)
			c := a / b

			// vary operand precision (only ok as long as a, b can be represented correctly)
			for ad := -dprec; ad <= dprec; ad++ {
				for bd := -dprec; bd <= dprec; bd++ {
					A := new(Float).SetPrec(uint(prec + ad)).SetFloat64(a)
					B := new(Float).SetPrec(uint(prec + bd)).SetFloat64(b)
					C := new(Float).SetPrec(53).Quo(A, B) // C has float64 mantissa width

					cc, acc := C.Float64()
					if cc != c {
						t.Errorf("%g/%g = %s; want %.5g\n", a, b, C.Format('g', 5), c)
						continue
					}
					if acc != Exact {
						t.Errorf("%g/%g got %s result; want exact result", a, b, acc)
					}
				}
			}
		}
	}
}

func TestFloatCmp(t *testing.T) {
	// TODO(gri) implement this
}

// normBits returns the normalized bits for x: It
// removes multiple equal entries by treating them
// as an addition (e.g., []int{5, 5} => []int{6}),
// and it sorts the result list for reproducible
// results.
func normBits(x []int) []int {
	m := make(map[int]bool)
	for _, b := range x {
		for m[b] {
			m[b] = false
			b++
		}
		m[b] = true
	}
	var z []int
	for b, set := range m {
		if set {
			z = append(z, b)
		}
	}
	sort.Ints(z)
	return z
}

func TestNormBits(t *testing.T) {
	for _, test := range []struct {
		x, want []int
	}{
		{nil, nil},
		{[]int{}, []int{}},
		{[]int{0}, []int{0}},
		{[]int{0, 0}, []int{1}},
		{[]int{3, 1, 1}, []int{2, 3}},
		{[]int{10, 9, 8, 7, 6, 6}, []int{11}},
	} {
		got := fmt.Sprintf("%v", normBits(test.x))
		want := fmt.Sprintf("%v", test.want)
		if got != want {
			t.Errorf("normBits(%v) = %s; want %s", test.x, got, want)
		}

	}
}

// roundBits returns the Float value rounded to prec bits
// according to mode from the bit set x.
func roundBits(x []int, prec uint, mode RoundingMode) *Float {
	x = normBits(x)

	// determine range
	var min, max int
	for i, b := range x {
		if i == 0 || b < min {
			min = b
		}
		if i == 0 || b > max {
			max = b
		}
	}
	prec0 := uint(max + 1 - min)
	if prec >= prec0 {
		return fromBits(x...)
	}
	// prec < prec0

	// determine bit 0, rounding, and sticky bit, and result bits z
	var bit0, rbit, sbit uint
	var z []int
	r := max - int(prec)
	for _, b := range x {
		switch {
		case b == r:
			rbit = 1
		case b < r:
			sbit = 1
		default:
			// b > r
			if b == r+1 {
				bit0 = 1
			}
			z = append(z, b)
		}
	}

	// round
	f := fromBits(z...) // rounded to zero
	if mode == ToNearestAway {
		panic("not yet implemented")
	}
	if mode == ToNearestEven && rbit == 1 && (sbit == 1 || sbit == 0 && bit0 != 0) || mode == AwayFromZero {
		// round away from zero
		f.SetMode(ToZero).SetPrec(prec)
		f.Add(f, fromBits(int(r)+1))
	}
	return f
}

// fromBits returns the *Float z of the smallest possible precision
// such that z = sum(2**bits[i]), with i = range bits.
// If multiple bits[i] are equal, they are added: fromBits(0, 1, 0)
// == 2**1 + 2**0 + 2**0 = 4.
func fromBits(bits ...int) *Float {
	// handle 0
	if len(bits) == 0 {
		return new(Float)
	}
	// len(bits) > 0

	// determine lsb exponent
	var min int
	for i, b := range bits {
		if i == 0 || b < min {
			min = b
		}
	}

	// create bit pattern
	x := NewInt(0)
	for _, b := range bits {
		badj := b - min
		// propagate carry if necessary
		for x.Bit(badj) != 0 {
			x.SetBit(x, badj, 0)
			badj++
		}
		x.SetBit(x, badj, 1)
	}

	// create corresponding float
	z := new(Float).SetInt(x) // normalized
	z.setExp(int64(z.exp) + int64(min))
	return z
}

func TestFromBits(t *testing.T) {
	for _, test := range []struct {
		bits []int
		want string
	}{
		// all different bit numbers
		{nil, "0"},
		{[]int{0}, "0x.8p1"},
		{[]int{1}, "0x.8p2"},
		{[]int{-1}, "0x.8p0"},
		{[]int{63}, "0x.8p64"},
		{[]int{33, -30}, "0x.8000000000000001p34"},
		{[]int{255, 0}, "0x.8000000000000000000000000000000000000000000000000000000000000001p256"},

		// multiple equal bit numbers
		{[]int{0, 0}, "0x.8p2"},
		{[]int{0, 0, 0, 0}, "0x.8p3"},
		{[]int{0, 1, 0}, "0x.8p3"},
		{append([]int{2, 1, 0} /* 7 */, []int{3, 1} /* 10 */ ...), "0x.88p5" /* 17 */},
	} {
		f := fromBits(test.bits...)
		if got := f.Format('p', 0); got != test.want {
			t.Errorf("setBits(%v) = %s; want %s", test.bits, got, test.want)
		}
	}
}
