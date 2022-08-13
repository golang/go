// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests floating point arithmetic expressions

package main

import (
	"fmt"
	"testing"
)

// manysub_ssa is designed to tickle bugs that depend on register
// pressure or unfriendly operand ordering in registers (and at
// least once it succeeded in this).
//
//go:noinline
func manysub_ssa(a, b, c, d float64) (aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd float64) {
	aa = a + 11.0 - a
	ab = a - b
	ac = a - c
	ad = a - d
	ba = b - a
	bb = b + 22.0 - b
	bc = b - c
	bd = b - d
	ca = c - a
	cb = c - b
	cc = c + 33.0 - c
	cd = c - d
	da = d - a
	db = d - b
	dc = d - c
	dd = d + 44.0 - d
	return
}

// fpspill_ssa attempts to trigger a bug where phis with floating point values
// were stored in non-fp registers causing an error in doasm.
//
//go:noinline
func fpspill_ssa(a int) float64 {

	ret := -1.0
	switch a {
	case 0:
		ret = 1.0
	case 1:
		ret = 1.1
	case 2:
		ret = 1.2
	case 3:
		ret = 1.3
	case 4:
		ret = 1.4
	case 5:
		ret = 1.5
	case 6:
		ret = 1.6
	case 7:
		ret = 1.7
	case 8:
		ret = 1.8
	case 9:
		ret = 1.9
	case 10:
		ret = 1.10
	case 11:
		ret = 1.11
	case 12:
		ret = 1.12
	case 13:
		ret = 1.13
	case 14:
		ret = 1.14
	case 15:
		ret = 1.15
	case 16:
		ret = 1.16
	}
	return ret
}

//go:noinline
func add64_ssa(a, b float64) float64 {
	return a + b
}

//go:noinline
func mul64_ssa(a, b float64) float64 {
	return a * b
}

//go:noinline
func sub64_ssa(a, b float64) float64 {
	return a - b
}

//go:noinline
func div64_ssa(a, b float64) float64 {
	return a / b
}

//go:noinline
func neg64_ssa(a, b float64) float64 {
	return -a + -1*b
}

//go:noinline
func add32_ssa(a, b float32) float32 {
	return a + b
}

//go:noinline
func mul32_ssa(a, b float32) float32 {
	return a * b
}

//go:noinline
func sub32_ssa(a, b float32) float32 {
	return a - b
}

//go:noinline
func div32_ssa(a, b float32) float32 {
	return a / b
}

//go:noinline
func neg32_ssa(a, b float32) float32 {
	return -a + -1*b
}

//go:noinline
func conv2Float64_ssa(a int8, b uint8, c int16, d uint16,
	e int32, f uint32, g int64, h uint64, i float32) (aa, bb, cc, dd, ee, ff, gg, hh, ii float64) {
	aa = float64(a)
	bb = float64(b)
	cc = float64(c)
	hh = float64(h)
	dd = float64(d)
	ee = float64(e)
	ff = float64(f)
	gg = float64(g)
	ii = float64(i)
	return
}

//go:noinline
func conv2Float32_ssa(a int8, b uint8, c int16, d uint16,
	e int32, f uint32, g int64, h uint64, i float64) (aa, bb, cc, dd, ee, ff, gg, hh, ii float32) {
	aa = float32(a)
	bb = float32(b)
	cc = float32(c)
	dd = float32(d)
	ee = float32(e)
	ff = float32(f)
	gg = float32(g)
	hh = float32(h)
	ii = float32(i)
	return
}

func integer2floatConversions(t *testing.T) {
	{
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(0, 0, 0, 0, 0, 0, 0, 0, 0)
		expectAll64(t, "zero64", 0, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(1, 1, 1, 1, 1, 1, 1, 1, 1)
		expectAll64(t, "one64", 1, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(0, 0, 0, 0, 0, 0, 0, 0, 0)
		expectAll32(t, "zero32", 0, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(1, 1, 1, 1, 1, 1, 1, 1, 1)
		expectAll32(t, "one32", 1, a, b, c, d, e, f, g, h, i)
	}
	{
		// Check maximum values
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(127, 255, 32767, 65535, 0x7fffffff, 0xffffffff, 0x7fffFFFFffffFFFF, 0xffffFFFFffffFFFF, 3.402823e38)
		expect64(t, "a", a, 127)
		expect64(t, "b", b, 255)
		expect64(t, "c", c, 32767)
		expect64(t, "d", d, 65535)
		expect64(t, "e", e, float64(int32(0x7fffffff)))
		expect64(t, "f", f, float64(uint32(0xffffffff)))
		expect64(t, "g", g, float64(int64(0x7fffffffffffffff)))
		expect64(t, "h", h, float64(uint64(0xffffffffffffffff)))
		expect64(t, "i", i, float64(float32(3.402823e38)))
	}
	{
		// Check minimum values (and tweaks for unsigned)
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(-128, 254, -32768, 65534, ^0x7fffffff, 0xfffffffe, ^0x7fffFFFFffffFFFF, 0xffffFFFFffffF401, 1.5e-45)
		expect64(t, "a", a, -128)
		expect64(t, "b", b, 254)
		expect64(t, "c", c, -32768)
		expect64(t, "d", d, 65534)
		expect64(t, "e", e, float64(^int32(0x7fffffff)))
		expect64(t, "f", f, float64(uint32(0xfffffffe)))
		expect64(t, "g", g, float64(^int64(0x7fffffffffffffff)))
		expect64(t, "h", h, float64(uint64(0xfffffffffffff401)))
		expect64(t, "i", i, float64(float32(1.5e-45)))
	}
	{
		// Check maximum values
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(127, 255, 32767, 65535, 0x7fffffff, 0xffffffff, 0x7fffFFFFffffFFFF, 0xffffFFFFffffFFFF, 3.402823e38)
		expect32(t, "a", a, 127)
		expect32(t, "b", b, 255)
		expect32(t, "c", c, 32767)
		expect32(t, "d", d, 65535)
		expect32(t, "e", e, float32(int32(0x7fffffff)))
		expect32(t, "f", f, float32(uint32(0xffffffff)))
		expect32(t, "g", g, float32(int64(0x7fffffffffffffff)))
		expect32(t, "h", h, float32(uint64(0xffffffffffffffff)))
		expect32(t, "i", i, float32(float64(3.402823e38)))
	}
	{
		// Check minimum values (and tweaks for unsigned)
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(-128, 254, -32768, 65534, ^0x7fffffff, 0xfffffffe, ^0x7fffFFFFffffFFFF, 0xffffFFFFffffF401, 1.5e-45)
		expect32(t, "a", a, -128)
		expect32(t, "b", b, 254)
		expect32(t, "c", c, -32768)
		expect32(t, "d", d, 65534)
		expect32(t, "e", e, float32(^int32(0x7fffffff)))
		expect32(t, "f", f, float32(uint32(0xfffffffe)))
		expect32(t, "g", g, float32(^int64(0x7fffffffffffffff)))
		expect32(t, "h", h, float32(uint64(0xfffffffffffff401)))
		expect32(t, "i", i, float32(float64(1.5e-45)))
	}
}

func multiplyAdd(t *testing.T) {
	{
		// Test that a multiply-accumulate operation with intermediate
		// rounding forced by a float32() cast produces the expected
		// result.
		// Test cases generated experimentally on a system (s390x) that
		// supports fused multiply-add instructions.
		var tests = [...]struct{ x, y, z, res float32 }{
			{0.6046603, 0.9405091, 0.6645601, 1.2332485},      // fused multiply-add result: 1.2332486
			{0.67908466, 0.21855305, 0.20318687, 0.3516029},   // fused multiply-add result: 0.35160288
			{0.29311424, 0.29708257, 0.752573, 0.8396522},     // fused multiply-add result: 0.8396521
			{0.5305857, 0.2535405, 0.282081, 0.41660595},      // fused multiply-add result: 0.41660598
			{0.29711226, 0.89436173, 0.097454615, 0.36318043}, // fused multiply-add result: 0.36318046
			{0.6810783, 0.24151509, 0.31152245, 0.47601312},   // fused multiply-add result: 0.47601315
			{0.73023146, 0.18292491, 0.4283571, 0.5619346},    // fused multiply-add result: 0.56193465
			{0.89634174, 0.32208398, 0.7211478, 1.009845},     // fused multiply-add result: 1.0098451
			{0.6280982, 0.12675293, 0.2813303, 0.36094356},    // fused multiply-add result: 0.3609436
			{0.29400632, 0.75316125, 0.15096405, 0.3723982},   // fused multiply-add result: 0.37239823
		}
		check := func(s string, got, expected float32) {
			if got != expected {
				fmt.Printf("multiplyAdd: %s, expected %g, got %g\n", s, expected, got)
			}
		}
		for _, t := range tests {
			check(
				fmt.Sprintf("float32(%v * %v) + %v", t.x, t.y, t.z),
				func(x, y, z float32) float32 {
					return float32(x*y) + z
				}(t.x, t.y, t.z),
				t.res)

			check(
				fmt.Sprintf("%v += float32(%v * %v)", t.z, t.x, t.y),
				func(x, y, z float32) float32 {
					z += float32(x * y)
					return z
				}(t.x, t.y, t.z),
				t.res)
		}
	}
	{
		// Test that a multiply-accumulate operation with intermediate
		// rounding forced by a float64() cast produces the expected
		// result.
		// Test cases generated experimentally on a system (s390x) that
		// supports fused multiply-add instructions.
		var tests = [...]struct{ x, y, z, res float64 }{
			{0.4688898449024232, 0.28303415118044517, 0.29310185733681576, 0.42581369658590373}, // fused multiply-add result: 0.4258136965859037
			{0.7886049150193449, 0.3618054804803169, 0.8805431227416171, 1.1658647029293308},    // fused multiply-add result: 1.1658647029293305
			{0.7302314772948083, 0.18292491645390843, 0.4283570818068078, 0.5619346137829748},   // fused multiply-add result: 0.5619346137829747
			{0.6908388315056789, 0.7109071952999951, 0.5637795958152644, 1.0549018919252924},    // fused multiply-add result: 1.0549018919252926
			{0.4584424785756506, 0.6001655953233308, 0.02626515060968944, 0.3014065536855481},   // fused multiply-add result: 0.30140655368554814
			{0.539210105890946, 0.9756748149873165, 0.7507630564795985, 1.2768567767840384},     // fused multiply-add result: 1.2768567767840386
			{0.7830349733960021, 0.3932509992288867, 0.1304138461737918, 0.4383431318929343},    // fused multiply-add result: 0.43834313189293433
			{0.6841751300974551, 0.6530402051353608, 0.524499759549865, 0.9712936268572192},     // fused multiply-add result: 0.9712936268572193
			{0.3691117091643448, 0.826454125634742, 0.34768170859156955, 0.6527356034505334},    // fused multiply-add result: 0.6527356034505333
			{0.16867966833433606, 0.33136826030698385, 0.8279280961505588, 0.8838231843956668},  // fused multiply-add result: 0.8838231843956669
		}
		check := func(s string, got, expected float64) {
			if got != expected {
				fmt.Printf("multiplyAdd: %s, expected %g, got %g\n", s, expected, got)
			}
		}
		for _, t := range tests {
			check(
				fmt.Sprintf("float64(%v * %v) + %v", t.x, t.y, t.z),
				func(x, y, z float64) float64 {
					return float64(x*y) + z
				}(t.x, t.y, t.z),
				t.res)

			check(
				fmt.Sprintf("%v += float64(%v * %v)", t.z, t.x, t.y),
				func(x, y, z float64) float64 {
					z += float64(x * y)
					return z
				}(t.x, t.y, t.z),
				t.res)
		}
	}
	{
		// Test that a multiply-accumulate operation with intermediate
		// rounding forced by a complex128() cast produces the expected
		// result.
		// Test cases generated experimentally on a system (s390x) that
		// supports fused multiply-add instructions.
		var tests = [...]struct {
			x, y float64
			res  complex128
		}{
			{0.6046602879796196, 0.9405090880450124, (2.754489951983871 + 3i)},    // fused multiply-add result: (2.7544899519838713 + 3i)
			{0.09696951891448456, 0.30091186058528707, (0.5918204173287407 + 3i)}, // fused multiply-add result: (0.5918204173287408 + 3i)
			{0.544155573000885, 0.27850762181610883, (1.910974340818764 + 3i)},    // fused multiply-add result: (1.9109743408187638 + 3i)
			{0.9769168685862624, 0.07429099894984302, (3.0050416047086297 + 3i)},  // fused multiply-add result: (3.00504160470863 + 3i)
			{0.9269868035744142, 0.9549454404167818, (3.735905851140024 + 3i)},    // fused multiply-add result: (3.7359058511400245 + 3i)
			{0.7109071952999951, 0.5637795958152644, (2.69650118171525 + 3i)},     // fused multiply-add result: (2.6965011817152496 + 3i)
			{0.7558235074915978, 0.40380328579570035, (2.671273808270494 + 3i)},   // fused multiply-add result: (2.6712738082704934 + 3i)
			{0.13065111702897217, 0.9859647293402467, (1.3779180804271633 + 3i)},  // fused multiply-add result: (1.3779180804271631 + 3i)
			{0.8963417453962161, 0.3220839705208817, (3.0111092067095298 + 3i)},   // fused multiply-add result: (3.01110920670953 + 3i)
			{0.39998376285699544, 0.497868113342702, (1.697819401913688 + 3i)},    // fused multiply-add result: (1.6978194019136883 + 3i)
		}
		check := func(s string, got, expected complex128) {
			if got != expected {
				fmt.Printf("multiplyAdd: %s, expected %v, got %v\n", s, expected, got)
			}
		}
		for _, t := range tests {
			check(
				fmt.Sprintf("complex128(complex(%v, 1)*3) + complex(%v, 0)", t.x, t.y),
				func(x, y float64) complex128 {
					return complex128(complex(x, 1)*3) + complex(y, 0)
				}(t.x, t.y),
				t.res)

			check(
				fmt.Sprintf("z := complex(%v, 1); z += complex128(complex(%v, 1) * 3)", t.y, t.x),
				func(x, y float64) complex128 {
					z := complex(y, 0)
					z += complex128(complex(x, 1) * 3)
					return z
				}(t.x, t.y),
				t.res)
		}
	}
}

const (
	aa = 0x1000000000000000
	ab = 0x100000000000000
	ac = 0x10000000000000
	ad = 0x1000000000000
	ba = 0x100000000000
	bb = 0x10000000000
	bc = 0x1000000000
	bd = 0x100000000
	ca = 0x10000000
	cb = 0x1000000
	cc = 0x100000
	cd = 0x10000
	da = 0x1000
	db = 0x100
	dc = 0x10
	dd = 0x1
)

//go:noinline
func compares64_ssa(a, b, c, d float64) (lt, le, eq, ne, ge, gt uint64) {
	if a < a {
		lt += aa
	}
	if a < b {
		lt += ab
	}
	if a < c {
		lt += ac
	}
	if a < d {
		lt += ad
	}

	if b < a {
		lt += ba
	}
	if b < b {
		lt += bb
	}
	if b < c {
		lt += bc
	}
	if b < d {
		lt += bd
	}

	if c < a {
		lt += ca
	}
	if c < b {
		lt += cb
	}
	if c < c {
		lt += cc
	}
	if c < d {
		lt += cd
	}

	if d < a {
		lt += da
	}
	if d < b {
		lt += db
	}
	if d < c {
		lt += dc
	}
	if d < d {
		lt += dd
	}

	if a <= a {
		le += aa
	}
	if a <= b {
		le += ab
	}
	if a <= c {
		le += ac
	}
	if a <= d {
		le += ad
	}

	if b <= a {
		le += ba
	}
	if b <= b {
		le += bb
	}
	if b <= c {
		le += bc
	}
	if b <= d {
		le += bd
	}

	if c <= a {
		le += ca
	}
	if c <= b {
		le += cb
	}
	if c <= c {
		le += cc
	}
	if c <= d {
		le += cd
	}

	if d <= a {
		le += da
	}
	if d <= b {
		le += db
	}
	if d <= c {
		le += dc
	}
	if d <= d {
		le += dd
	}

	if a == a {
		eq += aa
	}
	if a == b {
		eq += ab
	}
	if a == c {
		eq += ac
	}
	if a == d {
		eq += ad
	}

	if b == a {
		eq += ba
	}
	if b == b {
		eq += bb
	}
	if b == c {
		eq += bc
	}
	if b == d {
		eq += bd
	}

	if c == a {
		eq += ca
	}
	if c == b {
		eq += cb
	}
	if c == c {
		eq += cc
	}
	if c == d {
		eq += cd
	}

	if d == a {
		eq += da
	}
	if d == b {
		eq += db
	}
	if d == c {
		eq += dc
	}
	if d == d {
		eq += dd
	}

	if a != a {
		ne += aa
	}
	if a != b {
		ne += ab
	}
	if a != c {
		ne += ac
	}
	if a != d {
		ne += ad
	}

	if b != a {
		ne += ba
	}
	if b != b {
		ne += bb
	}
	if b != c {
		ne += bc
	}
	if b != d {
		ne += bd
	}

	if c != a {
		ne += ca
	}
	if c != b {
		ne += cb
	}
	if c != c {
		ne += cc
	}
	if c != d {
		ne += cd
	}

	if d != a {
		ne += da
	}
	if d != b {
		ne += db
	}
	if d != c {
		ne += dc
	}
	if d != d {
		ne += dd
	}

	if a >= a {
		ge += aa
	}
	if a >= b {
		ge += ab
	}
	if a >= c {
		ge += ac
	}
	if a >= d {
		ge += ad
	}

	if b >= a {
		ge += ba
	}
	if b >= b {
		ge += bb
	}
	if b >= c {
		ge += bc
	}
	if b >= d {
		ge += bd
	}

	if c >= a {
		ge += ca
	}
	if c >= b {
		ge += cb
	}
	if c >= c {
		ge += cc
	}
	if c >= d {
		ge += cd
	}

	if d >= a {
		ge += da
	}
	if d >= b {
		ge += db
	}
	if d >= c {
		ge += dc
	}
	if d >= d {
		ge += dd
	}

	if a > a {
		gt += aa
	}
	if a > b {
		gt += ab
	}
	if a > c {
		gt += ac
	}
	if a > d {
		gt += ad
	}

	if b > a {
		gt += ba
	}
	if b > b {
		gt += bb
	}
	if b > c {
		gt += bc
	}
	if b > d {
		gt += bd
	}

	if c > a {
		gt += ca
	}
	if c > b {
		gt += cb
	}
	if c > c {
		gt += cc
	}
	if c > d {
		gt += cd
	}

	if d > a {
		gt += da
	}
	if d > b {
		gt += db
	}
	if d > c {
		gt += dc
	}
	if d > d {
		gt += dd
	}

	return
}

//go:noinline
func compares32_ssa(a, b, c, d float32) (lt, le, eq, ne, ge, gt uint64) {
	if a < a {
		lt += aa
	}
	if a < b {
		lt += ab
	}
	if a < c {
		lt += ac
	}
	if a < d {
		lt += ad
	}

	if b < a {
		lt += ba
	}
	if b < b {
		lt += bb
	}
	if b < c {
		lt += bc
	}
	if b < d {
		lt += bd
	}

	if c < a {
		lt += ca
	}
	if c < b {
		lt += cb
	}
	if c < c {
		lt += cc
	}
	if c < d {
		lt += cd
	}

	if d < a {
		lt += da
	}
	if d < b {
		lt += db
	}
	if d < c {
		lt += dc
	}
	if d < d {
		lt += dd
	}

	if a <= a {
		le += aa
	}
	if a <= b {
		le += ab
	}
	if a <= c {
		le += ac
	}
	if a <= d {
		le += ad
	}

	if b <= a {
		le += ba
	}
	if b <= b {
		le += bb
	}
	if b <= c {
		le += bc
	}
	if b <= d {
		le += bd
	}

	if c <= a {
		le += ca
	}
	if c <= b {
		le += cb
	}
	if c <= c {
		le += cc
	}
	if c <= d {
		le += cd
	}

	if d <= a {
		le += da
	}
	if d <= b {
		le += db
	}
	if d <= c {
		le += dc
	}
	if d <= d {
		le += dd
	}

	if a == a {
		eq += aa
	}
	if a == b {
		eq += ab
	}
	if a == c {
		eq += ac
	}
	if a == d {
		eq += ad
	}

	if b == a {
		eq += ba
	}
	if b == b {
		eq += bb
	}
	if b == c {
		eq += bc
	}
	if b == d {
		eq += bd
	}

	if c == a {
		eq += ca
	}
	if c == b {
		eq += cb
	}
	if c == c {
		eq += cc
	}
	if c == d {
		eq += cd
	}

	if d == a {
		eq += da
	}
	if d == b {
		eq += db
	}
	if d == c {
		eq += dc
	}
	if d == d {
		eq += dd
	}

	if a != a {
		ne += aa
	}
	if a != b {
		ne += ab
	}
	if a != c {
		ne += ac
	}
	if a != d {
		ne += ad
	}

	if b != a {
		ne += ba
	}
	if b != b {
		ne += bb
	}
	if b != c {
		ne += bc
	}
	if b != d {
		ne += bd
	}

	if c != a {
		ne += ca
	}
	if c != b {
		ne += cb
	}
	if c != c {
		ne += cc
	}
	if c != d {
		ne += cd
	}

	if d != a {
		ne += da
	}
	if d != b {
		ne += db
	}
	if d != c {
		ne += dc
	}
	if d != d {
		ne += dd
	}

	if a >= a {
		ge += aa
	}
	if a >= b {
		ge += ab
	}
	if a >= c {
		ge += ac
	}
	if a >= d {
		ge += ad
	}

	if b >= a {
		ge += ba
	}
	if b >= b {
		ge += bb
	}
	if b >= c {
		ge += bc
	}
	if b >= d {
		ge += bd
	}

	if c >= a {
		ge += ca
	}
	if c >= b {
		ge += cb
	}
	if c >= c {
		ge += cc
	}
	if c >= d {
		ge += cd
	}

	if d >= a {
		ge += da
	}
	if d >= b {
		ge += db
	}
	if d >= c {
		ge += dc
	}
	if d >= d {
		ge += dd
	}

	if a > a {
		gt += aa
	}
	if a > b {
		gt += ab
	}
	if a > c {
		gt += ac
	}
	if a > d {
		gt += ad
	}

	if b > a {
		gt += ba
	}
	if b > b {
		gt += bb
	}
	if b > c {
		gt += bc
	}
	if b > d {
		gt += bd
	}

	if c > a {
		gt += ca
	}
	if c > b {
		gt += cb
	}
	if c > c {
		gt += cc
	}
	if c > d {
		gt += cd
	}

	if d > a {
		gt += da
	}
	if d > b {
		gt += db
	}
	if d > c {
		gt += dc
	}
	if d > d {
		gt += dd
	}

	return
}

//go:noinline
func le64_ssa(x, y float64) bool {
	return x <= y
}

//go:noinline
func ge64_ssa(x, y float64) bool {
	return x >= y
}

//go:noinline
func lt64_ssa(x, y float64) bool {
	return x < y
}

//go:noinline
func gt64_ssa(x, y float64) bool {
	return x > y
}

//go:noinline
func eq64_ssa(x, y float64) bool {
	return x == y
}

//go:noinline
func ne64_ssa(x, y float64) bool {
	return x != y
}

//go:noinline
func eqbr64_ssa(x, y float64) float64 {
	if x == y {
		return 17
	}
	return 42
}

//go:noinline
func nebr64_ssa(x, y float64) float64 {
	if x != y {
		return 17
	}
	return 42
}

//go:noinline
func gebr64_ssa(x, y float64) float64 {
	if x >= y {
		return 17
	}
	return 42
}

//go:noinline
func lebr64_ssa(x, y float64) float64 {
	if x <= y {
		return 17
	}
	return 42
}

//go:noinline
func ltbr64_ssa(x, y float64) float64 {
	if x < y {
		return 17
	}
	return 42
}

//go:noinline
func gtbr64_ssa(x, y float64) float64 {
	if x > y {
		return 17
	}
	return 42
}

//go:noinline
func le32_ssa(x, y float32) bool {
	return x <= y
}

//go:noinline
func ge32_ssa(x, y float32) bool {
	return x >= y
}

//go:noinline
func lt32_ssa(x, y float32) bool {
	return x < y
}

//go:noinline
func gt32_ssa(x, y float32) bool {
	return x > y
}

//go:noinline
func eq32_ssa(x, y float32) bool {
	return x == y
}

//go:noinline
func ne32_ssa(x, y float32) bool {
	return x != y
}

//go:noinline
func eqbr32_ssa(x, y float32) float32 {
	if x == y {
		return 17
	}
	return 42
}

//go:noinline
func nebr32_ssa(x, y float32) float32 {
	if x != y {
		return 17
	}
	return 42
}

//go:noinline
func gebr32_ssa(x, y float32) float32 {
	if x >= y {
		return 17
	}
	return 42
}

//go:noinline
func lebr32_ssa(x, y float32) float32 {
	if x <= y {
		return 17
	}
	return 42
}

//go:noinline
func ltbr32_ssa(x, y float32) float32 {
	if x < y {
		return 17
	}
	return 42
}

//go:noinline
func gtbr32_ssa(x, y float32) float32 {
	if x > y {
		return 17
	}
	return 42
}

//go:noinline
func F32toU8_ssa(x float32) uint8 {
	return uint8(x)
}

//go:noinline
func F32toI8_ssa(x float32) int8 {
	return int8(x)
}

//go:noinline
func F32toU16_ssa(x float32) uint16 {
	return uint16(x)
}

//go:noinline
func F32toI16_ssa(x float32) int16 {
	return int16(x)
}

//go:noinline
func F32toU32_ssa(x float32) uint32 {
	return uint32(x)
}

//go:noinline
func F32toI32_ssa(x float32) int32 {
	return int32(x)
}

//go:noinline
func F32toU64_ssa(x float32) uint64 {
	return uint64(x)
}

//go:noinline
func F32toI64_ssa(x float32) int64 {
	return int64(x)
}

//go:noinline
func F64toU8_ssa(x float64) uint8 {
	return uint8(x)
}

//go:noinline
func F64toI8_ssa(x float64) int8 {
	return int8(x)
}

//go:noinline
func F64toU16_ssa(x float64) uint16 {
	return uint16(x)
}

//go:noinline
func F64toI16_ssa(x float64) int16 {
	return int16(x)
}

//go:noinline
func F64toU32_ssa(x float64) uint32 {
	return uint32(x)
}

//go:noinline
func F64toI32_ssa(x float64) int32 {
	return int32(x)
}

//go:noinline
func F64toU64_ssa(x float64) uint64 {
	return uint64(x)
}

//go:noinline
func F64toI64_ssa(x float64) int64 {
	return int64(x)
}

func floatsToInts(t *testing.T, x float64, expected int64) {
	y := float32(x)
	expectInt64(t, "F64toI8", int64(F64toI8_ssa(x)), expected)
	expectInt64(t, "F64toI16", int64(F64toI16_ssa(x)), expected)
	expectInt64(t, "F64toI32", int64(F64toI32_ssa(x)), expected)
	expectInt64(t, "F64toI64", int64(F64toI64_ssa(x)), expected)
	expectInt64(t, "F32toI8", int64(F32toI8_ssa(y)), expected)
	expectInt64(t, "F32toI16", int64(F32toI16_ssa(y)), expected)
	expectInt64(t, "F32toI32", int64(F32toI32_ssa(y)), expected)
	expectInt64(t, "F32toI64", int64(F32toI64_ssa(y)), expected)
}

func floatsToUints(t *testing.T, x float64, expected uint64) {
	y := float32(x)
	expectUint64(t, "F64toU8", uint64(F64toU8_ssa(x)), expected)
	expectUint64(t, "F64toU16", uint64(F64toU16_ssa(x)), expected)
	expectUint64(t, "F64toU32", uint64(F64toU32_ssa(x)), expected)
	expectUint64(t, "F64toU64", uint64(F64toU64_ssa(x)), expected)
	expectUint64(t, "F32toU8", uint64(F32toU8_ssa(y)), expected)
	expectUint64(t, "F32toU16", uint64(F32toU16_ssa(y)), expected)
	expectUint64(t, "F32toU32", uint64(F32toU32_ssa(y)), expected)
	expectUint64(t, "F32toU64", uint64(F32toU64_ssa(y)), expected)
}

func floatingToIntegerConversionsTest(t *testing.T) {
	floatsToInts(t, 0.0, 0)
	floatsToInts(t, 0.5, 0)
	floatsToInts(t, 0.9, 0)
	floatsToInts(t, 1.0, 1)
	floatsToInts(t, 1.5, 1)
	floatsToInts(t, 127.0, 127)
	floatsToInts(t, -1.0, -1)
	floatsToInts(t, -128.0, -128)

	floatsToUints(t, 0.0, 0)
	floatsToUints(t, 1.0, 1)
	floatsToUints(t, 255.0, 255)

	for j := uint(0); j < 24; j++ {
		// Avoid hard cases in the construction
		// of the test inputs.
		v := int64(1<<62) | int64(1<<(62-j))
		w := uint64(v)
		f := float32(v)
		d := float64(v)
		expectUint64(t, "2**62...", F32toU64_ssa(f), w)
		expectUint64(t, "2**62...", F64toU64_ssa(d), w)
		expectInt64(t, "2**62...", F32toI64_ssa(f), v)
		expectInt64(t, "2**62...", F64toI64_ssa(d), v)
		expectInt64(t, "2**62...", F32toI64_ssa(-f), -v)
		expectInt64(t, "2**62...", F64toI64_ssa(-d), -v)
		w += w
		f += f
		d += d
		expectUint64(t, "2**63...", F32toU64_ssa(f), w)
		expectUint64(t, "2**63...", F64toU64_ssa(d), w)
	}

	for j := uint(0); j < 16; j++ {
		// Avoid hard cases in the construction
		// of the test inputs.
		v := int32(1<<30) | int32(1<<(30-j))
		w := uint32(v)
		f := float32(v)
		d := float64(v)
		expectUint32(t, "2**30...", F32toU32_ssa(f), w)
		expectUint32(t, "2**30...", F64toU32_ssa(d), w)
		expectInt32(t, "2**30...", F32toI32_ssa(f), v)
		expectInt32(t, "2**30...", F64toI32_ssa(d), v)
		expectInt32(t, "2**30...", F32toI32_ssa(-f), -v)
		expectInt32(t, "2**30...", F64toI32_ssa(-d), -v)
		w += w
		f += f
		d += d
		expectUint32(t, "2**31...", F32toU32_ssa(f), w)
		expectUint32(t, "2**31...", F64toU32_ssa(d), w)
	}

	for j := uint(0); j < 15; j++ {
		// Avoid hard cases in the construction
		// of the test inputs.
		v := int16(1<<14) | int16(1<<(14-j))
		w := uint16(v)
		f := float32(v)
		d := float64(v)
		expectUint16(t, "2**14...", F32toU16_ssa(f), w)
		expectUint16(t, "2**14...", F64toU16_ssa(d), w)
		expectInt16(t, "2**14...", F32toI16_ssa(f), v)
		expectInt16(t, "2**14...", F64toI16_ssa(d), v)
		expectInt16(t, "2**14...", F32toI16_ssa(-f), -v)
		expectInt16(t, "2**14...", F64toI16_ssa(-d), -v)
		w += w
		f += f
		d += d
		expectUint16(t, "2**15...", F32toU16_ssa(f), w)
		expectUint16(t, "2**15...", F64toU16_ssa(d), w)
	}

	expectInt32(t, "-2147483648", F32toI32_ssa(-2147483648), -2147483648)

	expectInt32(t, "-2147483648", F64toI32_ssa(-2147483648), -2147483648)
	expectInt32(t, "-2147483647", F64toI32_ssa(-2147483647), -2147483647)
	expectUint32(t, "4294967295", F64toU32_ssa(4294967295), 4294967295)

	expectInt16(t, "-32768", F64toI16_ssa(-32768), -32768)
	expectInt16(t, "-32768", F32toI16_ssa(-32768), -32768)

	// NB more of a pain to do these for 32-bit because of lost bits in Float32 mantissa
	expectInt16(t, "32767", F64toI16_ssa(32767), 32767)
	expectInt16(t, "32767", F32toI16_ssa(32767), 32767)
	expectUint16(t, "32767", F64toU16_ssa(32767), 32767)
	expectUint16(t, "32767", F32toU16_ssa(32767), 32767)
	expectUint16(t, "65535", F64toU16_ssa(65535), 65535)
	expectUint16(t, "65535", F32toU16_ssa(65535), 65535)
}

func fail64(s string, f func(a, b float64) float64, a, b, e float64) {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float64) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
	}
}

func fail64bool(s string, f func(a, b float64) bool, a, b float64, e bool) {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float64) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
	}
}

func fail32(s string, f func(a, b float32) float32, a, b, e float32) {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float32) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
	}
}

func fail32bool(s string, f func(a, b float32) bool, a, b float32, e bool) {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float32) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
	}
}

func expect64(t *testing.T, s string, x, expected float64) {
	if x != expected {
		println("F64 Expected", expected, "for", s, ", got", x)
	}
}

func expect32(t *testing.T, s string, x, expected float32) {
	if x != expected {
		println("F32 Expected", expected, "for", s, ", got", x)
	}
}

func expectUint64(t *testing.T, s string, x, expected uint64) {
	if x != expected {
		fmt.Printf("U64 Expected 0x%016x for %s, got 0x%016x\n", expected, s, x)
	}
}

func expectInt64(t *testing.T, s string, x, expected int64) {
	if x != expected {
		fmt.Printf("%s: Expected 0x%016x, got 0x%016x\n", s, expected, x)
	}
}

func expectUint32(t *testing.T, s string, x, expected uint32) {
	if x != expected {
		fmt.Printf("U32 %s: Expected 0x%08x, got 0x%08x\n", s, expected, x)
	}
}

func expectInt32(t *testing.T, s string, x, expected int32) {
	if x != expected {
		fmt.Printf("I32 %s: Expected 0x%08x, got 0x%08x\n", s, expected, x)
	}
}

func expectUint16(t *testing.T, s string, x, expected uint16) {
	if x != expected {
		fmt.Printf("U16 %s: Expected 0x%04x, got 0x%04x\n", s, expected, x)
	}
}

func expectInt16(t *testing.T, s string, x, expected int16) {
	if x != expected {
		fmt.Printf("I16 %s: Expected 0x%04x, got 0x%04x\n", s, expected, x)
	}
}

func expectAll64(t *testing.T, s string, expected, a, b, c, d, e, f, g, h, i float64) {
	expect64(t, s+":a", a, expected)
	expect64(t, s+":b", b, expected)
	expect64(t, s+":c", c, expected)
	expect64(t, s+":d", d, expected)
	expect64(t, s+":e", e, expected)
	expect64(t, s+":f", f, expected)
	expect64(t, s+":g", g, expected)
}

func expectAll32(t *testing.T, s string, expected, a, b, c, d, e, f, g, h, i float32) {
	expect32(t, s+":a", a, expected)
	expect32(t, s+":b", b, expected)
	expect32(t, s+":c", c, expected)
	expect32(t, s+":d", d, expected)
	expect32(t, s+":e", e, expected)
	expect32(t, s+":f", f, expected)
	expect32(t, s+":g", g, expected)
}

var ev64 [2]float64 = [2]float64{42.0, 17.0}
var ev32 [2]float32 = [2]float32{42.0, 17.0}

func cmpOpTest(t *testing.T,
	s string,
	f func(a, b float64) bool,
	g func(a, b float64) float64,
	ff func(a, b float32) bool,
	gg func(a, b float32) float32,
	zero, one, inf, nan float64, result uint) {
	fail64bool(s, f, zero, zero, result>>16&1 == 1)
	fail64bool(s, f, zero, one, result>>12&1 == 1)
	fail64bool(s, f, zero, inf, result>>8&1 == 1)
	fail64bool(s, f, zero, nan, result>>4&1 == 1)
	fail64bool(s, f, nan, nan, result&1 == 1)

	fail64(s, g, zero, zero, ev64[result>>16&1])
	fail64(s, g, zero, one, ev64[result>>12&1])
	fail64(s, g, zero, inf, ev64[result>>8&1])
	fail64(s, g, zero, nan, ev64[result>>4&1])
	fail64(s, g, nan, nan, ev64[result>>0&1])

	{
		zero := float32(zero)
		one := float32(one)
		inf := float32(inf)
		nan := float32(nan)
		fail32bool(s, ff, zero, zero, (result>>16)&1 == 1)
		fail32bool(s, ff, zero, one, (result>>12)&1 == 1)
		fail32bool(s, ff, zero, inf, (result>>8)&1 == 1)
		fail32bool(s, ff, zero, nan, (result>>4)&1 == 1)
		fail32bool(s, ff, nan, nan, result&1 == 1)

		fail32(s, gg, zero, zero, ev32[(result>>16)&1])
		fail32(s, gg, zero, one, ev32[(result>>12)&1])
		fail32(s, gg, zero, inf, ev32[(result>>8)&1])
		fail32(s, gg, zero, nan, ev32[(result>>4)&1])
		fail32(s, gg, nan, nan, ev32[(result>>0)&1])
	}
}

func expectCx128(t *testing.T, s string, x, expected complex128) {
	if x != expected {
		t.Errorf("Cx 128 Expected %f for %s, got %f", expected, s, x)
	}
}

func expectCx64(t *testing.T, s string, x, expected complex64) {
	if x != expected {
		t.Errorf("Cx 64 Expected %f for %s, got %f", expected, s, x)
	}
}

//go:noinline
func cx128sum_ssa(a, b complex128) complex128 {
	return a + b
}

//go:noinline
func cx128diff_ssa(a, b complex128) complex128 {
	return a - b
}

//go:noinline
func cx128prod_ssa(a, b complex128) complex128 {
	return a * b
}

//go:noinline
func cx128quot_ssa(a, b complex128) complex128 {
	return a / b
}

//go:noinline
func cx128neg_ssa(a complex128) complex128 {
	return -a
}

//go:noinline
func cx128real_ssa(a complex128) float64 {
	return real(a)
}

//go:noinline
func cx128imag_ssa(a complex128) float64 {
	return imag(a)
}

//go:noinline
func cx128cnst_ssa(a complex128) complex128 {
	b := 2 + 3i
	return a * b
}

//go:noinline
func cx64sum_ssa(a, b complex64) complex64 {
	return a + b
}

//go:noinline
func cx64diff_ssa(a, b complex64) complex64 {
	return a - b
}

//go:noinline
func cx64prod_ssa(a, b complex64) complex64 {
	return a * b
}

//go:noinline
func cx64quot_ssa(a, b complex64) complex64 {
	return a / b
}

//go:noinline
func cx64neg_ssa(a complex64) complex64 {
	return -a
}

//go:noinline
func cx64real_ssa(a complex64) float32 {
	return real(a)
}

//go:noinline
func cx64imag_ssa(a complex64) float32 {
	return imag(a)
}

//go:noinline
func cx128eq_ssa(a, b complex128) bool {
	return a == b
}

//go:noinline
func cx128ne_ssa(a, b complex128) bool {
	return a != b
}

//go:noinline
func cx64eq_ssa(a, b complex64) bool {
	return a == b
}

//go:noinline
func cx64ne_ssa(a, b complex64) bool {
	return a != b
}

func expectTrue(t *testing.T, s string, b bool) {
	if !b {
		t.Errorf("expected true for %s, got false", s)
	}
}
func expectFalse(t *testing.T, s string, b bool) {
	if b {
		t.Errorf("expected false for %s, got true", s)
	}
}

func complexTest128(t *testing.T) {
	var a complex128 = 1 + 2i
	var b complex128 = 3 + 6i
	sum := cx128sum_ssa(b, a)
	diff := cx128diff_ssa(b, a)
	prod := cx128prod_ssa(b, a)
	quot := cx128quot_ssa(b, a)
	neg := cx128neg_ssa(a)
	r := cx128real_ssa(a)
	i := cx128imag_ssa(a)
	cnst := cx128cnst_ssa(a)
	c1 := cx128eq_ssa(a, a)
	c2 := cx128eq_ssa(a, b)
	c3 := cx128ne_ssa(a, a)
	c4 := cx128ne_ssa(a, b)

	expectCx128(t, "sum", sum, 4+8i)
	expectCx128(t, "diff", diff, 2+4i)
	expectCx128(t, "prod", prod, -9+12i)
	expectCx128(t, "quot", quot, 3+0i)
	expectCx128(t, "neg", neg, -1-2i)
	expect64(t, "real", r, 1)
	expect64(t, "imag", i, 2)
	expectCx128(t, "cnst", cnst, -4+7i)
	expectTrue(t, fmt.Sprintf("%v==%v", a, a), c1)
	expectFalse(t, fmt.Sprintf("%v==%v", a, b), c2)
	expectFalse(t, fmt.Sprintf("%v!=%v", a, a), c3)
	expectTrue(t, fmt.Sprintf("%v!=%v", a, b), c4)
}

func complexTest64(t *testing.T) {
	var a complex64 = 1 + 2i
	var b complex64 = 3 + 6i
	sum := cx64sum_ssa(b, a)
	diff := cx64diff_ssa(b, a)
	prod := cx64prod_ssa(b, a)
	quot := cx64quot_ssa(b, a)
	neg := cx64neg_ssa(a)
	r := cx64real_ssa(a)
	i := cx64imag_ssa(a)
	c1 := cx64eq_ssa(a, a)
	c2 := cx64eq_ssa(a, b)
	c3 := cx64ne_ssa(a, a)
	c4 := cx64ne_ssa(a, b)

	expectCx64(t, "sum", sum, 4+8i)
	expectCx64(t, "diff", diff, 2+4i)
	expectCx64(t, "prod", prod, -9+12i)
	expectCx64(t, "quot", quot, 3+0i)
	expectCx64(t, "neg", neg, -1-2i)
	expect32(t, "real", r, 1)
	expect32(t, "imag", i, 2)
	expectTrue(t, fmt.Sprintf("%v==%v", a, a), c1)
	expectFalse(t, fmt.Sprintf("%v==%v", a, b), c2)
	expectFalse(t, fmt.Sprintf("%v!=%v", a, a), c3)
	expectTrue(t, fmt.Sprintf("%v!=%v", a, b), c4)
}

// TestFP tests that we get the right answer for floating point expressions.
func TestFP(t *testing.T) {
	a := 3.0
	b := 4.0

	c := float32(3.0)
	d := float32(4.0)

	tiny := float32(1.5e-45) // smallest f32 denorm = 2**(-149)
	dtiny := float64(tiny)   // well within range of f64

	fail64("+", add64_ssa, a, b, 7.0)
	fail64("*", mul64_ssa, a, b, 12.0)
	fail64("-", sub64_ssa, a, b, -1.0)
	fail64("/", div64_ssa, a, b, 0.75)
	fail64("neg", neg64_ssa, a, b, -7)

	fail32("+", add32_ssa, c, d, 7.0)
	fail32("*", mul32_ssa, c, d, 12.0)
	fail32("-", sub32_ssa, c, d, -1.0)
	fail32("/", div32_ssa, c, d, 0.75)
	fail32("neg", neg32_ssa, c, d, -7)

	// denorm-squared should underflow to zero.
	fail32("*", mul32_ssa, tiny, tiny, 0)

	// but should not underflow in float and in fact is exactly representable.
	fail64("*", mul64_ssa, dtiny, dtiny, 1.9636373861190906e-90)

	// Intended to create register pressure which forces
	// asymmetric op into different code paths.
	aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd := manysub_ssa(1000.0, 100.0, 10.0, 1.0)

	expect64(t, "aa", aa, 11.0)
	expect64(t, "ab", ab, 900.0)
	expect64(t, "ac", ac, 990.0)
	expect64(t, "ad", ad, 999.0)

	expect64(t, "ba", ba, -900.0)
	expect64(t, "bb", bb, 22.0)
	expect64(t, "bc", bc, 90.0)
	expect64(t, "bd", bd, 99.0)

	expect64(t, "ca", ca, -990.0)
	expect64(t, "cb", cb, -90.0)
	expect64(t, "cc", cc, 33.0)
	expect64(t, "cd", cd, 9.0)

	expect64(t, "da", da, -999.0)
	expect64(t, "db", db, -99.0)
	expect64(t, "dc", dc, -9.0)
	expect64(t, "dd", dd, 44.0)

	integer2floatConversions(t)

	multiplyAdd(t)

	var zero64 float64 = 0.0
	var one64 float64 = 1.0
	var inf64 float64 = 1.0 / zero64
	var nan64 float64 = sub64_ssa(inf64, inf64)

	cmpOpTest(t, "!=", ne64_ssa, nebr64_ssa, ne32_ssa, nebr32_ssa, zero64, one64, inf64, nan64, 0x01111)
	cmpOpTest(t, "==", eq64_ssa, eqbr64_ssa, eq32_ssa, eqbr32_ssa, zero64, one64, inf64, nan64, 0x10000)
	cmpOpTest(t, "<=", le64_ssa, lebr64_ssa, le32_ssa, lebr32_ssa, zero64, one64, inf64, nan64, 0x11100)
	cmpOpTest(t, "<", lt64_ssa, ltbr64_ssa, lt32_ssa, ltbr32_ssa, zero64, one64, inf64, nan64, 0x01100)
	cmpOpTest(t, ">", gt64_ssa, gtbr64_ssa, gt32_ssa, gtbr32_ssa, zero64, one64, inf64, nan64, 0x00000)
	cmpOpTest(t, ">=", ge64_ssa, gebr64_ssa, ge32_ssa, gebr32_ssa, zero64, one64, inf64, nan64, 0x10000)

	{
		lt, le, eq, ne, ge, gt := compares64_ssa(0.0, 1.0, inf64, nan64)
		expectUint64(t, "lt", lt, 0x0110001000000000)
		expectUint64(t, "le", le, 0x1110011000100000)
		expectUint64(t, "eq", eq, 0x1000010000100000)
		expectUint64(t, "ne", ne, 0x0111101111011111)
		expectUint64(t, "ge", ge, 0x1000110011100000)
		expectUint64(t, "gt", gt, 0x0000100011000000)
		// fmt.Printf("lt=0x%016x, le=0x%016x, eq=0x%016x, ne=0x%016x, ge=0x%016x, gt=0x%016x\n",
		// 	lt, le, eq, ne, ge, gt)
	}
	{
		lt, le, eq, ne, ge, gt := compares32_ssa(0.0, 1.0, float32(inf64), float32(nan64))
		expectUint64(t, "lt", lt, 0x0110001000000000)
		expectUint64(t, "le", le, 0x1110011000100000)
		expectUint64(t, "eq", eq, 0x1000010000100000)
		expectUint64(t, "ne", ne, 0x0111101111011111)
		expectUint64(t, "ge", ge, 0x1000110011100000)
		expectUint64(t, "gt", gt, 0x0000100011000000)
	}

	floatingToIntegerConversionsTest(t)
	complexTest128(t)
	complexTest64(t)
}
