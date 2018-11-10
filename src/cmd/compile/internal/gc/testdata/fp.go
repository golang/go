// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests floating point arithmetic expressions

package main

import "fmt"

// manysub_ssa is designed to tickle bugs that depend on register
// pressure or unfriendly operand ordering in registers (and at
// least once it succeeded in this).
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

func integer2floatConversions() int {
	fails := 0
	{
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(0, 0, 0, 0, 0, 0, 0, 0, 0)
		fails += expectAll64("zero64", 0, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(1, 1, 1, 1, 1, 1, 1, 1, 1)
		fails += expectAll64("one64", 1, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(0, 0, 0, 0, 0, 0, 0, 0, 0)
		fails += expectAll32("zero32", 0, a, b, c, d, e, f, g, h, i)
	}
	{
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(1, 1, 1, 1, 1, 1, 1, 1, 1)
		fails += expectAll32("one32", 1, a, b, c, d, e, f, g, h, i)
	}
	{
		// Check maximum values
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(127, 255, 32767, 65535, 0x7fffffff, 0xffffffff, 0x7fffFFFFffffFFFF, 0xffffFFFFffffFFFF, 3.402823E38)
		fails += expect64("a", a, 127)
		fails += expect64("b", b, 255)
		fails += expect64("c", c, 32767)
		fails += expect64("d", d, 65535)
		fails += expect64("e", e, float64(int32(0x7fffffff)))
		fails += expect64("f", f, float64(uint32(0xffffffff)))
		fails += expect64("g", g, float64(int64(0x7fffffffffffffff)))
		fails += expect64("h", h, float64(uint64(0xffffffffffffffff)))
		fails += expect64("i", i, float64(float32(3.402823E38)))
	}
	{
		// Check minimum values (and tweaks for unsigned)
		a, b, c, d, e, f, g, h, i := conv2Float64_ssa(-128, 254, -32768, 65534, ^0x7fffffff, 0xfffffffe, ^0x7fffFFFFffffFFFF, 0xffffFFFFffffF401, 1.5E-45)
		fails += expect64("a", a, -128)
		fails += expect64("b", b, 254)
		fails += expect64("c", c, -32768)
		fails += expect64("d", d, 65534)
		fails += expect64("e", e, float64(^int32(0x7fffffff)))
		fails += expect64("f", f, float64(uint32(0xfffffffe)))
		fails += expect64("g", g, float64(^int64(0x7fffffffffffffff)))
		fails += expect64("h", h, float64(uint64(0xfffffffffffff401)))
		fails += expect64("i", i, float64(float32(1.5E-45)))
	}
	{
		// Check maximum values
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(127, 255, 32767, 65535, 0x7fffffff, 0xffffffff, 0x7fffFFFFffffFFFF, 0xffffFFFFffffFFFF, 3.402823E38)
		fails += expect32("a", a, 127)
		fails += expect32("b", b, 255)
		fails += expect32("c", c, 32767)
		fails += expect32("d", d, 65535)
		fails += expect32("e", e, float32(int32(0x7fffffff)))
		fails += expect32("f", f, float32(uint32(0xffffffff)))
		fails += expect32("g", g, float32(int64(0x7fffffffffffffff)))
		fails += expect32("h", h, float32(uint64(0xffffffffffffffff)))
		fails += expect32("i", i, float32(float64(3.402823E38)))
	}
	{
		// Check minimum values (and tweaks for unsigned)
		a, b, c, d, e, f, g, h, i := conv2Float32_ssa(-128, 254, -32768, 65534, ^0x7fffffff, 0xfffffffe, ^0x7fffFFFFffffFFFF, 0xffffFFFFffffF401, 1.5E-45)
		fails += expect32("a", a, -128)
		fails += expect32("b", b, 254)
		fails += expect32("c", c, -32768)
		fails += expect32("d", d, 65534)
		fails += expect32("e", e, float32(^int32(0x7fffffff)))
		fails += expect32("f", f, float32(uint32(0xfffffffe)))
		fails += expect32("g", g, float32(^int64(0x7fffffffffffffff)))
		fails += expect32("h", h, float32(uint64(0xfffffffffffff401)))
		fails += expect32("i", i, float32(float64(1.5E-45)))
	}
	return fails
}

func multiplyAdd() int {
	fails := 0
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
		check := func(s string, got, expected float32) int {
			if got != expected {
				fmt.Printf("multiplyAdd: %s, expected %g, got %g\n", s, expected, got)
				return 1
			}
			return 0
		}
		for _, t := range tests {
			fails += check(
				fmt.Sprintf("float32(%v * %v) + %v", t.x, t.y, t.z),
				func(x, y, z float32) float32 {
					return float32(x*y) + z
				}(t.x, t.y, t.z),
				t.res)

			fails += check(
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
		check := func(s string, got, expected float64) int {
			if got != expected {
				fmt.Printf("multiplyAdd: %s, expected %g, got %g\n", s, expected, got)
				return 1
			}
			return 0
		}
		for _, t := range tests {
			fails += check(
				fmt.Sprintf("float64(%v * %v) + %v", t.x, t.y, t.z),
				func(x, y, z float64) float64 {
					return float64(x*y) + z
				}(t.x, t.y, t.z),
				t.res)

			fails += check(
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
		check := func(s string, got, expected complex128) int {
			if got != expected {
				fmt.Printf("multiplyAdd: %s, expected %v, got %v\n", s, expected, got)
				return 1
			}
			return 0
		}
		for _, t := range tests {
			fails += check(
				fmt.Sprintf("complex128(complex(%v, 1)*3) + complex(%v, 0)", t.x, t.y),
				func(x, y float64) complex128 {
					return complex128(complex(x, 1)*3) + complex(y, 0)
				}(t.x, t.y),
				t.res)

			fails += check(
				fmt.Sprintf("z := complex(%v, 1); z += complex128(complex(%v, 1) * 3)", t.y, t.x),
				func(x, y float64) complex128 {
					z := complex(y, 0)
					z += complex128(complex(x, 1) * 3)
					return z
				}(t.x, t.y),
				t.res)
		}
	}
	return fails
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

func floatsToInts(x float64, expected int64) int {
	y := float32(x)
	fails := 0
	fails += expectInt64("F64toI8", int64(F64toI8_ssa(x)), expected)
	fails += expectInt64("F64toI16", int64(F64toI16_ssa(x)), expected)
	fails += expectInt64("F64toI32", int64(F64toI32_ssa(x)), expected)
	fails += expectInt64("F64toI64", int64(F64toI64_ssa(x)), expected)
	fails += expectInt64("F32toI8", int64(F32toI8_ssa(y)), expected)
	fails += expectInt64("F32toI16", int64(F32toI16_ssa(y)), expected)
	fails += expectInt64("F32toI32", int64(F32toI32_ssa(y)), expected)
	fails += expectInt64("F32toI64", int64(F32toI64_ssa(y)), expected)
	return fails
}

func floatsToUints(x float64, expected uint64) int {
	y := float32(x)
	fails := 0
	fails += expectUint64("F64toU8", uint64(F64toU8_ssa(x)), expected)
	fails += expectUint64("F64toU16", uint64(F64toU16_ssa(x)), expected)
	fails += expectUint64("F64toU32", uint64(F64toU32_ssa(x)), expected)
	fails += expectUint64("F64toU64", uint64(F64toU64_ssa(x)), expected)
	fails += expectUint64("F32toU8", uint64(F32toU8_ssa(y)), expected)
	fails += expectUint64("F32toU16", uint64(F32toU16_ssa(y)), expected)
	fails += expectUint64("F32toU32", uint64(F32toU32_ssa(y)), expected)
	fails += expectUint64("F32toU64", uint64(F32toU64_ssa(y)), expected)
	return fails
}

func floatingToIntegerConversionsTest() int {
	fails := 0
	fails += floatsToInts(0.0, 0)
	fails += floatsToInts(0.5, 0)
	fails += floatsToInts(0.9, 0)
	fails += floatsToInts(1.0, 1)
	fails += floatsToInts(1.5, 1)
	fails += floatsToInts(127.0, 127)
	fails += floatsToInts(-1.0, -1)
	fails += floatsToInts(-128.0, -128)

	fails += floatsToUints(0.0, 0)
	fails += floatsToUints(1.0, 1)
	fails += floatsToUints(255.0, 255)

	for j := uint(0); j < 24; j++ {
		// Avoid hard cases in the construction
		// of the test inputs.
		v := int64(1<<62) | int64(1<<(62-j))
		w := uint64(v)
		f := float32(v)
		d := float64(v)
		fails += expectUint64("2**62...", F32toU64_ssa(f), w)
		fails += expectUint64("2**62...", F64toU64_ssa(d), w)
		fails += expectInt64("2**62...", F32toI64_ssa(f), v)
		fails += expectInt64("2**62...", F64toI64_ssa(d), v)
		fails += expectInt64("2**62...", F32toI64_ssa(-f), -v)
		fails += expectInt64("2**62...", F64toI64_ssa(-d), -v)
		w += w
		f += f
		d += d
		fails += expectUint64("2**63...", F32toU64_ssa(f), w)
		fails += expectUint64("2**63...", F64toU64_ssa(d), w)
	}

	for j := uint(0); j < 16; j++ {
		// Avoid hard cases in the construction
		// of the test inputs.
		v := int32(1<<30) | int32(1<<(30-j))
		w := uint32(v)
		f := float32(v)
		d := float64(v)
		fails += expectUint32("2**30...", F32toU32_ssa(f), w)
		fails += expectUint32("2**30...", F64toU32_ssa(d), w)
		fails += expectInt32("2**30...", F32toI32_ssa(f), v)
		fails += expectInt32("2**30...", F64toI32_ssa(d), v)
		fails += expectInt32("2**30...", F32toI32_ssa(-f), -v)
		fails += expectInt32("2**30...", F64toI32_ssa(-d), -v)
		w += w
		f += f
		d += d
		fails += expectUint32("2**31...", F32toU32_ssa(f), w)
		fails += expectUint32("2**31...", F64toU32_ssa(d), w)
	}

	for j := uint(0); j < 15; j++ {
		// Avoid hard cases in the construction
		// of the test inputs.
		v := int16(1<<14) | int16(1<<(14-j))
		w := uint16(v)
		f := float32(v)
		d := float64(v)
		fails += expectUint16("2**14...", F32toU16_ssa(f), w)
		fails += expectUint16("2**14...", F64toU16_ssa(d), w)
		fails += expectInt16("2**14...", F32toI16_ssa(f), v)
		fails += expectInt16("2**14...", F64toI16_ssa(d), v)
		fails += expectInt16("2**14...", F32toI16_ssa(-f), -v)
		fails += expectInt16("2**14...", F64toI16_ssa(-d), -v)
		w += w
		f += f
		d += d
		fails += expectUint16("2**15...", F32toU16_ssa(f), w)
		fails += expectUint16("2**15...", F64toU16_ssa(d), w)
	}

	fails += expectInt32("-2147483648", F32toI32_ssa(-2147483648), -2147483648)

	fails += expectInt32("-2147483648", F64toI32_ssa(-2147483648), -2147483648)
	fails += expectInt32("-2147483647", F64toI32_ssa(-2147483647), -2147483647)
	fails += expectUint32("4294967295", F64toU32_ssa(4294967295), 4294967295)

	fails += expectInt16("-32768", F64toI16_ssa(-32768), -32768)
	fails += expectInt16("-32768", F32toI16_ssa(-32768), -32768)

	// NB more of a pain to do these for 32-bit because of lost bits in Float32 mantissa
	fails += expectInt16("32767", F64toI16_ssa(32767), 32767)
	fails += expectInt16("32767", F32toI16_ssa(32767), 32767)
	fails += expectUint16("32767", F64toU16_ssa(32767), 32767)
	fails += expectUint16("32767", F32toU16_ssa(32767), 32767)
	fails += expectUint16("65535", F64toU16_ssa(65535), 65535)
	fails += expectUint16("65535", F32toU16_ssa(65535), 65535)

	return fails
}

func fail64(s string, f func(a, b float64) float64, a, b, e float64) int {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float64) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
		return 1
	}
	return 0
}

func fail64bool(s string, f func(a, b float64) bool, a, b float64, e bool) int {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float64) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
		return 1
	}
	return 0
}

func fail32(s string, f func(a, b float32) float32, a, b, e float32) int {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float32) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
		return 1
	}
	return 0
}

func fail32bool(s string, f func(a, b float32) bool, a, b float32, e bool) int {
	d := f(a, b)
	if d != e {
		fmt.Printf("For (float32) %v %v %v, expected %v, got %v\n", a, s, b, e, d)
		return 1
	}
	return 0
}

func expect64(s string, x, expected float64) int {
	if x != expected {
		println("F64 Expected", expected, "for", s, ", got", x)
		return 1
	}
	return 0
}

func expect32(s string, x, expected float32) int {
	if x != expected {
		println("F32 Expected", expected, "for", s, ", got", x)
		return 1
	}
	return 0
}

func expectUint64(s string, x, expected uint64) int {
	if x != expected {
		fmt.Printf("U64 Expected 0x%016x for %s, got 0x%016x\n", expected, s, x)
		return 1
	}
	return 0
}

func expectInt64(s string, x, expected int64) int {
	if x != expected {
		fmt.Printf("%s: Expected 0x%016x, got 0x%016x\n", s, expected, x)
		return 1
	}
	return 0
}

func expectUint32(s string, x, expected uint32) int {
	if x != expected {
		fmt.Printf("U32 %s: Expected 0x%08x, got 0x%08x\n", s, expected, x)
		return 1
	}
	return 0
}

func expectInt32(s string, x, expected int32) int {
	if x != expected {
		fmt.Printf("I32 %s: Expected 0x%08x, got 0x%08x\n", s, expected, x)
		return 1
	}
	return 0
}

func expectUint16(s string, x, expected uint16) int {
	if x != expected {
		fmt.Printf("U16 %s: Expected 0x%04x, got 0x%04x\n", s, expected, x)
		return 1
	}
	return 0
}

func expectInt16(s string, x, expected int16) int {
	if x != expected {
		fmt.Printf("I16 %s: Expected 0x%04x, got 0x%04x\n", s, expected, x)
		return 1
	}
	return 0
}

func expectAll64(s string, expected, a, b, c, d, e, f, g, h, i float64) int {
	fails := 0
	fails += expect64(s+":a", a, expected)
	fails += expect64(s+":b", b, expected)
	fails += expect64(s+":c", c, expected)
	fails += expect64(s+":d", d, expected)
	fails += expect64(s+":e", e, expected)
	fails += expect64(s+":f", f, expected)
	fails += expect64(s+":g", g, expected)
	return fails
}

func expectAll32(s string, expected, a, b, c, d, e, f, g, h, i float32) int {
	fails := 0
	fails += expect32(s+":a", a, expected)
	fails += expect32(s+":b", b, expected)
	fails += expect32(s+":c", c, expected)
	fails += expect32(s+":d", d, expected)
	fails += expect32(s+":e", e, expected)
	fails += expect32(s+":f", f, expected)
	fails += expect32(s+":g", g, expected)
	return fails
}

var ev64 [2]float64 = [2]float64{42.0, 17.0}
var ev32 [2]float32 = [2]float32{42.0, 17.0}

func cmpOpTest(s string,
	f func(a, b float64) bool,
	g func(a, b float64) float64,
	ff func(a, b float32) bool,
	gg func(a, b float32) float32,
	zero, one, inf, nan float64, result uint) int {
	fails := 0
	fails += fail64bool(s, f, zero, zero, result>>16&1 == 1)
	fails += fail64bool(s, f, zero, one, result>>12&1 == 1)
	fails += fail64bool(s, f, zero, inf, result>>8&1 == 1)
	fails += fail64bool(s, f, zero, nan, result>>4&1 == 1)
	fails += fail64bool(s, f, nan, nan, result&1 == 1)

	fails += fail64(s, g, zero, zero, ev64[result>>16&1])
	fails += fail64(s, g, zero, one, ev64[result>>12&1])
	fails += fail64(s, g, zero, inf, ev64[result>>8&1])
	fails += fail64(s, g, zero, nan, ev64[result>>4&1])
	fails += fail64(s, g, nan, nan, ev64[result>>0&1])

	{
		zero := float32(zero)
		one := float32(one)
		inf := float32(inf)
		nan := float32(nan)
		fails += fail32bool(s, ff, zero, zero, (result>>16)&1 == 1)
		fails += fail32bool(s, ff, zero, one, (result>>12)&1 == 1)
		fails += fail32bool(s, ff, zero, inf, (result>>8)&1 == 1)
		fails += fail32bool(s, ff, zero, nan, (result>>4)&1 == 1)
		fails += fail32bool(s, ff, nan, nan, result&1 == 1)

		fails += fail32(s, gg, zero, zero, ev32[(result>>16)&1])
		fails += fail32(s, gg, zero, one, ev32[(result>>12)&1])
		fails += fail32(s, gg, zero, inf, ev32[(result>>8)&1])
		fails += fail32(s, gg, zero, nan, ev32[(result>>4)&1])
		fails += fail32(s, gg, nan, nan, ev32[(result>>0)&1])
	}

	return fails
}

func expectCx128(s string, x, expected complex128) int {
	if x != expected {
		println("Cx 128 Expected", expected, "for", s, ", got", x)
		return 1
	}
	return 0
}

func expectCx64(s string, x, expected complex64) int {
	if x != expected {
		println("Cx 64 Expected", expected, "for", s, ", got", x)
		return 1
	}
	return 0
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

func expectTrue(s string, b bool) int {
	if !b {
		println("expected true for", s, ", got false")
		return 1
	}
	return 0
}
func expectFalse(s string, b bool) int {
	if b {
		println("expected false for", s, ", got true")
		return 1
	}
	return 0
}

func complexTest128() int {
	fails := 0
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

	fails += expectCx128("sum", sum, 4+8i)
	fails += expectCx128("diff", diff, 2+4i)
	fails += expectCx128("prod", prod, -9+12i)
	fails += expectCx128("quot", quot, 3+0i)
	fails += expectCx128("neg", neg, -1-2i)
	fails += expect64("real", r, 1)
	fails += expect64("imag", i, 2)
	fails += expectCx128("cnst", cnst, -4+7i)
	fails += expectTrue(fmt.Sprintf("%v==%v", a, a), c1)
	fails += expectFalse(fmt.Sprintf("%v==%v", a, b), c2)
	fails += expectFalse(fmt.Sprintf("%v!=%v", a, a), c3)
	fails += expectTrue(fmt.Sprintf("%v!=%v", a, b), c4)

	return fails
}

func complexTest64() int {
	fails := 0
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

	fails += expectCx64("sum", sum, 4+8i)
	fails += expectCx64("diff", diff, 2+4i)
	fails += expectCx64("prod", prod, -9+12i)
	fails += expectCx64("quot", quot, 3+0i)
	fails += expectCx64("neg", neg, -1-2i)
	fails += expect32("real", r, 1)
	fails += expect32("imag", i, 2)
	fails += expectTrue(fmt.Sprintf("%v==%v", a, a), c1)
	fails += expectFalse(fmt.Sprintf("%v==%v", a, b), c2)
	fails += expectFalse(fmt.Sprintf("%v!=%v", a, a), c3)
	fails += expectTrue(fmt.Sprintf("%v!=%v", a, b), c4)

	return fails
}

func main() {

	a := 3.0
	b := 4.0

	c := float32(3.0)
	d := float32(4.0)

	tiny := float32(1.5E-45) // smallest f32 denorm = 2**(-149)
	dtiny := float64(tiny)   // well within range of f64

	fails := 0
	fails += fail64("+", add64_ssa, a, b, 7.0)
	fails += fail64("*", mul64_ssa, a, b, 12.0)
	fails += fail64("-", sub64_ssa, a, b, -1.0)
	fails += fail64("/", div64_ssa, a, b, 0.75)
	fails += fail64("neg", neg64_ssa, a, b, -7)

	fails += fail32("+", add32_ssa, c, d, 7.0)
	fails += fail32("*", mul32_ssa, c, d, 12.0)
	fails += fail32("-", sub32_ssa, c, d, -1.0)
	fails += fail32("/", div32_ssa, c, d, 0.75)
	fails += fail32("neg", neg32_ssa, c, d, -7)

	// denorm-squared should underflow to zero.
	fails += fail32("*", mul32_ssa, tiny, tiny, 0)

	// but should not underflow in float and in fact is exactly representable.
	fails += fail64("*", mul64_ssa, dtiny, dtiny, 1.9636373861190906e-90)

	// Intended to create register pressure which forces
	// asymmetric op into different code paths.
	aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd := manysub_ssa(1000.0, 100.0, 10.0, 1.0)

	fails += expect64("aa", aa, 11.0)
	fails += expect64("ab", ab, 900.0)
	fails += expect64("ac", ac, 990.0)
	fails += expect64("ad", ad, 999.0)

	fails += expect64("ba", ba, -900.0)
	fails += expect64("bb", bb, 22.0)
	fails += expect64("bc", bc, 90.0)
	fails += expect64("bd", bd, 99.0)

	fails += expect64("ca", ca, -990.0)
	fails += expect64("cb", cb, -90.0)
	fails += expect64("cc", cc, 33.0)
	fails += expect64("cd", cd, 9.0)

	fails += expect64("da", da, -999.0)
	fails += expect64("db", db, -99.0)
	fails += expect64("dc", dc, -9.0)
	fails += expect64("dd", dd, 44.0)

	fails += integer2floatConversions()

	fails += multiplyAdd()

	var zero64 float64 = 0.0
	var one64 float64 = 1.0
	var inf64 float64 = 1.0 / zero64
	var nan64 float64 = sub64_ssa(inf64, inf64)

	fails += cmpOpTest("!=", ne64_ssa, nebr64_ssa, ne32_ssa, nebr32_ssa, zero64, one64, inf64, nan64, 0x01111)
	fails += cmpOpTest("==", eq64_ssa, eqbr64_ssa, eq32_ssa, eqbr32_ssa, zero64, one64, inf64, nan64, 0x10000)
	fails += cmpOpTest("<=", le64_ssa, lebr64_ssa, le32_ssa, lebr32_ssa, zero64, one64, inf64, nan64, 0x11100)
	fails += cmpOpTest("<", lt64_ssa, ltbr64_ssa, lt32_ssa, ltbr32_ssa, zero64, one64, inf64, nan64, 0x01100)
	fails += cmpOpTest(">", gt64_ssa, gtbr64_ssa, gt32_ssa, gtbr32_ssa, zero64, one64, inf64, nan64, 0x00000)
	fails += cmpOpTest(">=", ge64_ssa, gebr64_ssa, ge32_ssa, gebr32_ssa, zero64, one64, inf64, nan64, 0x10000)

	{
		lt, le, eq, ne, ge, gt := compares64_ssa(0.0, 1.0, inf64, nan64)
		fails += expectUint64("lt", lt, 0x0110001000000000)
		fails += expectUint64("le", le, 0x1110011000100000)
		fails += expectUint64("eq", eq, 0x1000010000100000)
		fails += expectUint64("ne", ne, 0x0111101111011111)
		fails += expectUint64("ge", ge, 0x1000110011100000)
		fails += expectUint64("gt", gt, 0x0000100011000000)
		// fmt.Printf("lt=0x%016x, le=0x%016x, eq=0x%016x, ne=0x%016x, ge=0x%016x, gt=0x%016x\n",
		// 	lt, le, eq, ne, ge, gt)
	}
	{
		lt, le, eq, ne, ge, gt := compares32_ssa(0.0, 1.0, float32(inf64), float32(nan64))
		fails += expectUint64("lt", lt, 0x0110001000000000)
		fails += expectUint64("le", le, 0x1110011000100000)
		fails += expectUint64("eq", eq, 0x1000010000100000)
		fails += expectUint64("ne", ne, 0x0111101111011111)
		fails += expectUint64("ge", ge, 0x1000110011100000)
		fails += expectUint64("gt", gt, 0x0000100011000000)
	}

	fails += floatingToIntegerConversionsTest()
	fails += complexTest128()
	fails += complexTest64()

	if fails > 0 {
		fmt.Printf("Saw %v failures\n", fails)
		panic("Failed.")
	}
}
