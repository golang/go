// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Floating point binary↔decimal conversion by fast unrounded scaling.
// See “Floating-Point Printing and Parsing Can Be Simple And Fast”,
// https://research.swtch.com/fp

package strconv

import (
	"math/bits"
	"unsafe"
)

// bool2 converts b to an integer: 1 for true, 0 for false.
func bool2[T ~int | ~uint32 | ~uint64](b bool) T {
	if b {
		return 1
	}
	return 0
}

// pack64 takes m, e and returns f = m * 2**e.
// It assumes the caller has provided a 53-bit mantissa m
// and an exponent that is in range for the mantissa.
func pack64(m uint64, e int) (float64, error) {
	if m&(1<<52) == 0 {
		return float64frombits(m), nil
	}
	if e >= 0x7FF-1075 {
		return float64frombits(m&(1<<63) | 0x7ff<<52), ErrRange
	}
	return float64frombits(m&^(1<<52) | uint64(1075+e)<<52), nil
}

// pack32 takes m, e and returns f = m * 2**e.
// It assumes the caller has provided a 24-bit mantissa m
// and an exponent that is in range for the mantissa.
func pack32(m uint32, e int) (float32, error) {
	if m&(1<<23) == 0 {
		return float32frombits(m), nil
	}
	if e >= 0xFF-150 {
		return float32frombits(m&(1<<31) | 0xff<<23), ErrRange
	}
	return float32frombits(m&^(1<<23) | uint32(150+e)<<23), nil
}

// An unrounded represents an unrounded value.
type unrounded uint64

func (u unrounded) floor() uint64         { return uint64((u + 0) >> 2) }
func (u unrounded) roundHalfDown() uint64 { return uint64((u + 1) >> 2) }
func (u unrounded) round() uint64         { return uint64((u + 1 + (u>>2)&1) >> 2) }
func (u unrounded) roundHalfUp() uint64   { return uint64((u + 2) >> 2) }
func (u unrounded) ceil() uint64          { return uint64((u + 3) >> 2) }
func (u unrounded) nudge(δ int) unrounded { return u + unrounded(δ) }

func (u unrounded) div(d uint64) unrounded {
	x := uint64(u)
	return unrounded(x/d) | u&1 | bool2[unrounded](x%d != 0)
}

func (u unrounded) rsh(s int) unrounded {
	return u>>s | u&1 | bool2[unrounded](u&((1<<s)-1) != 0)
}

// log10Pow2(x) returns ⌊log₁₀ 2**x⌋ = ⌊x * log₁₀ 2⌋.
func log10Pow2(x int) int {
	// log₁₀ 2 ≈ 0.30102999566 ≈ 78913 / 2^18
	return (x * 78913) >> 18
}

// log2Pow10(x) returns ⌊log₂ 10**x⌋ = ⌊x * log₂ 10⌋.
func log2Pow10(x int) int {
	// log₂ 10 ≈ 3.32192809489 ≈ 108853 / 2^15
	return (x * 108853) >> 15
}

// uint64pow10[x] is 10**x.
var uint64pow10 = [...]uint64{
	1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
}

// fixedWidthFloat returns the n-digit decimal form of f = m * 2**e as d * 10**p.
// n can be at most 18.
// If fmt == 'f' then n is a conservative estimate of the number of digits,
// and digits are discarded to match prec.
func fixedWidthFloat(m uint64, e, n, prec int, fmt byte) (d uint64, p int) {
	p = n - 1 - log10Pow2(e+63)
	var pre scaler
	prescale(&pre, e, p, log2Pow10(p))
	u := uscale(m, &pre)
	if u >= unmin(uint64pow10[n]) {
		u = u.div(10)
		p--
	}
	if fmt == 'f' {
		for p > prec {
			u = u.div(10)
			p--
		}
	}
	return u.round(), -p
}

// parseFloat64 rounds d * 10**p to the nearest float64 f.
// d can have at most 19 digits.
// It returns ErrRange if the result rounds to infinity.
func parseFloat64(d uint64, p int, sign uint64) (float64, error) {
	b := bits.Len64(d)
	lp := log2Pow10(p)
	e := min(1074, 53-b-lp)
	var pre scaler
	prescale(&pre, e-(64-b), p, lp)
	if pre.s >= 64 {
		return float64frombits(sign | 0), nil
	}
	u := uscale(d<<(64-b), &pre)

	// This block is branch-free code for:
	//	if u.round() >= 1<<53 {
	//		u = u.rsh(1)
	//		e = e - 1
	//	}
	s := bool2[int](u >= unmin(1<<53))
	u = u>>s | u&1
	e = e - s

	return pack64(sign|u.round(), -e)
}

// parseFloat32 rounds d * 10**p to the nearest float32 f.
// d can have at most 19 digits.
// It returns ErrRange if the result rounds to infinity.
func parseFloat32(d uint64, p int, sign uint32) (float32, error) {
	b := bits.Len64(d)
	lp := log2Pow10(p)
	e := min(149, 24-b-lp)
	var pre scaler
	prescale(&pre, e-(64-b), p, lp)
	if pre.s >= 64 {
		return float32frombits(sign | 0), nil
	}
	u := uscale(d<<(64-b), &pre)

	// This block is branch-free code for:
	//	if u.round() >= 1<<24 {
	//		u = u.rsh(1)
	//		e = e - 1
	//	}
	s := bool2[int](u >= unmin(1<<24))
	u = u>>s | u&1
	e = e - s

	return pack32(sign|uint32(u.round()), -e)
}

// unmin returns the minimum unrounded that rounds to x.
func unmin(x uint64) unrounded {
	return unrounded(x<<2 - 2)
}

// shortFloat computes the shortest formatting of f,
// using as few digits as possible that will still round trip
// back to the original float.
func shortFloat[F float32 | float64](m uint64, e int) (d uint64, p int) {
	var mantBits, minExp int // parameterized constants
	switch 8 * unsafe.Sizeof(F(0)) {
	case 32:
		mantBits = float32MantBits
		minExp = float32MinExp
	case 64:
		mantBits = float64MantBits
		minExp = float64MinExp
	}

	// Note: these cases could be factored a little more,
	// but in the first two branches, z is a constant,
	// allowing the compiler to greatly simplify the code.
	var min, max uint64
	var odd int
	z := 63 - mantBits
	if m == 1<<63 && e > minExp {
		p = -skewed(e + z)
		min = m - 1<<(z-2) // min = m - 1/4 * 2**(e+z)
		max = m + 1<<(z-1) // max = m + 1/2 * 2**(e+z)
		odd = int(m>>z) & 1
	} else if e >= minExp {
		p = -log10Pow2(e + z)
		min = m - 1<<(z-1) // min = m - 1/2 * 2**(e+z)
		max = m + 1<<(z-1) // max = m + 1/2 * 2**(e+z)
		odd = int(m>>z) & 1
	} else {
		z = z + (minExp - e)
		p = -log10Pow2(e + z)
		min = m - 1<<(z-1) // min = m - 1/2 * 2**(e+z)
		max = m + 1<<(z-1) // max = m + 1/2 * 2**(e+z)
		odd = int(m>>z) & 1
	}

	var pre scaler
	prescale(&pre, e, p, log2Pow10(p))
	dmin := uscale(min, &pre).nudge(+odd).ceil()
	dmax := uscale(max, &pre).nudge(-odd).floor()

	if d = dmax / 10; d*10 >= dmin {
		return d, -(p - 1)
	}
	if d = dmin; d < dmax {
		d = uscale(m, &pre).round()
	}
	return d, -p
}

// skewed computes the skewed footprint of m * 2**e,
// which is ⌊log₁₀ 3/4 * 2**e⌋ = ⌊e*(log₁₀ 2)-(log₁₀ 4/3)⌋.
func skewed(e int) int {
	return (e*631305 - 261663) >> 21
}

// A pmHiLo represents hi<<64 - lo.
type pmHiLo struct {
	hi uint64
	lo uint64
}

// A scaler holds derived scaling constants for a given e, p pair.
type scaler struct {
	// Note: using pm pmHiLo here nudges uscale just over the inlining boundary. Don't.
	pmHi uint64
	pmLo uint64
	s    int
}

// prescale returns the scaling constants for e, p.
// lp must be log2Pow10(p).
// The caller is responsible for either avoiding e, p pairs
// that cause pre.s < 0 or pre.s >= 64, or else handling
// those cases before passing the result to uscale.
// In practice, pre.s < 0 would indicate a buggy caller
// and pre.s >= 64 can only happen for parsing and is
// picked off at those call sites.
func prescale(pre *scaler, e, p, lp int) {
	pre.pmHi = pow10Tab[p-pow10Min].hi
	pre.pmLo = pow10Tab[p-pow10Min].lo
	pre.s = -(e + lp + 3)
}

// uscale returns unround(x * 2**e * 10**p).
// The caller should pass &pre for prescale(&pre, e, p, log2Pow10(p))
// and should have left-justified x so its high bit is set.
// The caller is also responsible for checking that c.s < 64.
// For formatting, that's always true.
// For parsing, the caller needs to pick it off early and return a signed 0.
func uscale(x uint64, c *scaler) unrounded {
	hi, mid := bits.Mul64(x, c.pmHi)
	s := c.s & 63 // make shifts cheaper
	if hi>>s<<s != hi {
		return unrounded(hi>>s | 1)
	}
	mid2, _ := bits.Mul64(x, c.pmLo)
	hi -= bool2[uint64](mid < mid2)
	return unrounded(hi>>s | bool2[uint64](mid-mid2 > 1))
}

// setDigits sets digs to the nd digits described by d, p.
func setDigits(s []byte, d uint64, p, nd int) (dp, nzd int) {
	// Note: nd <= len(s) is guaranteed by caller,
	// but writing it explicitly here lets the compiler know,
	// so that it can remove the bounds check in the loop.
	// (The slice s[:nd] not panicking only establishes nd <= cap(s).)
	if nd <= len(s) {
		formatBase10(s[:nd], d)
		dp = nd + p
		for nd > 0 && s[nd-1] == '0' {
			nd--
		}
	}
	return dp, nd
}

// numDigits returns the number of decimal digits in d.
// It requires d ≥ 1.
func numDigits(d uint64) int {
	nd := log10Pow2(bits.Len64(d))
	return nd + bool2[int](d >= uint64pow10[nd])
}
