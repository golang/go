// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"fmt"
	"sort"
	"testing"
)

func addBits(x, y []int) []int {
	return append(x, y...)
}

func mulBits(x, y []int) []int {
	var p []int
	for _, x := range x {
		for _, y := range y {
			p = append(p, x+y)
		}
	}
	return p
}

func TestMulBits(t *testing.T) {
	for _, test := range []struct {
		x, y, want []int
	}{
		{nil, nil, nil},
		{[]int{}, []int{}, nil},
		{[]int{0}, []int{0}, []int{0}},
		{[]int{0}, []int{1}, []int{1}},
		{[]int{1}, []int{1, 2, 3}, []int{2, 3, 4}},
		{[]int{-1}, []int{1}, []int{0}},
		{[]int{-10, -1, 0, 1, 10}, []int{1, 2, 3}, []int{-9, -8, -7, 0, 1, 2, 1, 2, 3, 2, 3, 4, 11, 12, 13}},
	} {
		got := fmt.Sprintf("%v", mulBits(test.x, test.y))
		want := fmt.Sprintf("%v", test.want)
		if got != want {
			t.Errorf("%v * %v = %s; want %s", test.x, test.y, got, want)
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
		return fromBits(x)
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
	f := fromBits(z) // rounded to zero
	if mode == ToNearestAway {
		panic("not yet implemented")
	}
	if mode == ToNearestEven && rbit == 1 && (sbit == 1 || sbit == 0 && bit0 != 0) || mode == AwayFromZero {
		// round away from zero
		f.SetMode(ToZero).SetPrec(prec)
		f.Add(f, fromBits([]int{int(r) + 1}))
	}
	return f
}

// fromBits returns the *Float z of the smallest possible precision
// such that z = sum(2**bits[i]), with i = range bits.
// If multiple bits[i] are equal, they are added: fromBits(0, 1, 0)
// == 2**1 + 2**0 + 2**0 = 4.
func fromBits(bits []int) *Float {
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
		f := fromBits(test.bits)
		if got := f.Format('p', 0); got != test.want {
			t.Errorf("setBits(%v) = %s; want %s", test.bits, got, test.want)
		}
	}
}
