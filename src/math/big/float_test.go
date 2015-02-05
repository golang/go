// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"fmt"
	"sort"
	"strconv"
	"testing"
)

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
	f := new(Float).SetInt64(x)
	f.Round(f, prec, mode)

	// check result
	r1 := f.Int64()
	p1 := f.Precision()
	a1 := f.Accuracy()
	if r1 != r || p1 != prec || a1 != a {
		t.Errorf("Round(%s, %d, %s): got %s (%d bits, %s); want %s (%d bits, %s)",
			toBinary(x), prec, mode,
			toBinary(r1), p1, a1,
			toBinary(r), prec, a)
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
		f := new(Float).SetFloat64(x)
		f.Round(f, 24, ToNearestEven)
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
		f := new(Float).SetUint64(want)
		if got := f.Uint64(); got != want {
			t.Errorf("got %d (%s); want %d", got, f.Format('p', 0), want)
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
			f := new(Float).SetInt64(want)
			if got := f.Int64(); got != want {
				t.Errorf("got %d (%s); want %d", got, f.Format('p', 0), want)
			}
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
	} {
		for i := range [2]int{} {
			if i&1 != 0 {
				want = -want
			}
			f := new(Float).SetFloat64(want)
			if got, _ := f.Float64(); got != want {
				t.Errorf("got %g (%s); want %g", got, f.Format('p', 0), want)
			}
		}
	}
}

func TestFloatSetInt(t *testing.T) {
	// TODO(gri) implement
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
					got := NewFloat(0, prec, mode)
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
			z := NewFloat(0, 24, ToNearestEven)

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
			z := NewFloat(0, 53, ToNearestEven)

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
			z := NewFloat(0, 53, ToNearestEven)

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
		two := NewFloat(2, prec, ToNearestEven)
		one := NewFloat(1, prec, ToNearestEven)
		three := NewFloat(3, prec, ToNearestEven)
		msix := NewFloat(-6, prec, ToNearestEven)
		psix := NewFloat(+6, prec, ToNearestEven)

		p := NewFloat(0, prec, ToNearestEven)
		z1 := NewFloat(0, prec, ToNearestEven)
		z2 := NewFloat(0, prec, ToNearestEven)

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

		x := NewFloat(0, z.Precision()+y.Precision(), ToZero)
		x.Mul(z, y)

		// leave for debugging
		// fmt.Printf("x = %s\ny = %s\nz = %s\n", x, y, z)

		if got := x.Accuracy(); got != Exact {
			t.Errorf("got acc = %s; want exact", got)
		}

		// round accurate z for a variety of precisions and
		// modes and compare against result of x / y.
		for _, mode := range [...]RoundingMode{ToZero, ToNearestEven, AwayFromZero} {
			for d := -5; d < 5; d++ {
				prec := uint(preci + d)
				got := NewFloat(0, prec, mode).Quo(x, y)
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
					A := NewFloat(a, uint(prec+ad), 0)
					B := NewFloat(b, uint(prec+bd), 0)
					C := NewFloat(0, 53, 0).Quo(A, B) // C has float64 mantissa width

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
		f.Round(f, prec, ToZero) // extend precision // TODO(gri) better approach?
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
		// z.prec = ?
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
