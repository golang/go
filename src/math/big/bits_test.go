// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Bits type used for testing Float operations
// via an independent (albeit slower) representations for floating-point
// numbers.

package big

import (
	"fmt"
	"sort"
	"testing"
)

// A Bits value b represents a finite floating-point number x of the form
//
//	x = 2**b[0] + 2**b[1] + ... 2**b[len(b)-1]
//
// The order of slice elements is not significant. Negative elements may be
// used to form fractions. A Bits value is normalized if each b[i] occurs at
// most once. For instance Bits{0, 0, 1} is not normalized but represents the
// same floating-point number as Bits{2}, which is normalized. The zero (nil)
// value of Bits is a ready to use Bits value and represents the value 0.
type Bits []int

func (x Bits) add(y Bits) Bits {
	return append(x, y...)
}

func (x Bits) mul(y Bits) Bits {
	var p Bits
	for _, x := range x {
		for _, y := range y {
			p = append(p, x+y)
		}
	}
	return p
}

func TestMulBits(t *testing.T) {
	for _, test := range []struct {
		x, y, want Bits
	}{
		{nil, nil, nil},
		{Bits{}, Bits{}, nil},
		{Bits{0}, Bits{0}, Bits{0}},
		{Bits{0}, Bits{1}, Bits{1}},
		{Bits{1}, Bits{1, 2, 3}, Bits{2, 3, 4}},
		{Bits{-1}, Bits{1}, Bits{0}},
		{Bits{-10, -1, 0, 1, 10}, Bits{1, 2, 3}, Bits{-9, -8, -7, 0, 1, 2, 1, 2, 3, 2, 3, 4, 11, 12, 13}},
	} {
		got := fmt.Sprintf("%v", test.x.mul(test.y))
		want := fmt.Sprintf("%v", test.want)
		if got != want {
			t.Errorf("%v * %v = %s; want %s", test.x, test.y, got, want)
		}

	}
}

// norm returns the normalized bits for x: It removes multiple equal entries
// by treating them as an addition (e.g., Bits{5, 5} => Bits{6}), and it sorts
// the result list for reproducible results.
func (x Bits) norm() Bits {
	m := make(map[int]bool)
	for _, b := range x {
		for m[b] {
			m[b] = false
			b++
		}
		m[b] = true
	}
	var z Bits
	for b, set := range m {
		if set {
			z = append(z, b)
		}
	}
	sort.Ints([]int(z))
	return z
}

func TestNormBits(t *testing.T) {
	for _, test := range []struct {
		x, want Bits
	}{
		{nil, nil},
		{Bits{}, Bits{}},
		{Bits{0}, Bits{0}},
		{Bits{0, 0}, Bits{1}},
		{Bits{3, 1, 1}, Bits{2, 3}},
		{Bits{10, 9, 8, 7, 6, 6}, Bits{11}},
	} {
		got := fmt.Sprintf("%v", test.x.norm())
		want := fmt.Sprintf("%v", test.want)
		if got != want {
			t.Errorf("normBits(%v) = %s; want %s", test.x, got, want)
		}

	}
}

// round returns the Float value corresponding to x after rounding x
// to prec bits according to mode.
func (x Bits) round(prec uint, mode RoundingMode) *Float {
	x = x.norm()

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
		return x.Float()
	}
	// prec < prec0

	// determine bit 0, rounding, and sticky bit, and result bits z
	var bit0, rbit, sbit uint
	var z Bits
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
	f := z.Float() // rounded to zero
	if mode == ToNearestAway {
		panic("not yet implemented")
	}
	if mode == ToNearestEven && rbit == 1 && (sbit == 1 || sbit == 0 && bit0 != 0) || mode == AwayFromZero {
		// round away from zero
		f.SetMode(ToZero).SetPrec(prec)
		f.Add(f, Bits{int(r) + 1}.Float())
	}
	return f
}

// Float returns the *Float z of the smallest possible precision such that
// z = sum(2**bits[i]), with i = range bits. If multiple bits[i] are equal,
// they are added: Bits{0, 1, 0}.Float() == 2**0 + 2**1 + 2**0 = 4.
func (bits Bits) Float() *Float {
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
	if e := int64(z.exp) + int64(min); MinExp <= e && e <= MaxExp {
		z.exp = int32(e)
	} else {
		// this should never happen for our test cases
		panic("exponent out of range")
	}
	return z
}

func TestFromBits(t *testing.T) {
	for _, test := range []struct {
		bits Bits
		want string
	}{
		// all different bit numbers
		{nil, "0"},
		{Bits{0}, "0x.8p+1"},
		{Bits{1}, "0x.8p+2"},
		{Bits{-1}, "0x.8p+0"},
		{Bits{63}, "0x.8p+64"},
		{Bits{33, -30}, "0x.8000000000000001p+34"},
		{Bits{255, 0}, "0x.8000000000000000000000000000000000000000000000000000000000000001p+256"},

		// multiple equal bit numbers
		{Bits{0, 0}, "0x.8p+2"},
		{Bits{0, 0, 0, 0}, "0x.8p+3"},
		{Bits{0, 1, 0}, "0x.8p+3"},
		{append(Bits{2, 1, 0} /* 7 */, Bits{3, 1} /* 10 */ ...), "0x.88p+5" /* 17 */},
	} {
		f := test.bits.Float()
		if got := f.Text('p', 0); got != test.want {
			t.Errorf("setBits(%v) = %s; want %s", test.bits, got, test.want)
		}
	}
}
