// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests floating point arithmetic expressions

package main

import "fmt"

func fail64(s string, f func(a, b float64) float64, a, b, e float64) int {
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

func expect64(s string, x, expected float64) int {
	if x != expected {
		println("Expected", expected, "for", s, ", got", x)
		return 1
	}
	return 0
}

func expect32(s string, x, expected float32) int {
	if x != expected {
		println("Expected", expected, "for", s, ", got", x)
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

// manysub_ssa is designed to tickle bugs that depend on register
// pressure or unfriendly operand ordering in registers (and at
// least once it succeeded in this).
func manysub_ssa(a, b, c, d float64) (aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd float64) {
	switch {
	}
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

func add64_ssa(a, b float64) float64 {
	switch {
	}
	return a + b
}

func mul64_ssa(a, b float64) float64 {
	switch {
	}
	return a * b
}

func sub64_ssa(a, b float64) float64 {
	switch {
	}
	return a - b
}

func div64_ssa(a, b float64) float64 {
	switch {
	}
	return a / b
}

func add32_ssa(a, b float32) float32 {
	switch {
	}
	return a + b
}

func mul32_ssa(a, b float32) float32 {
	switch {
	}
	return a * b
}

func sub32_ssa(a, b float32) float32 {
	switch {
	}
	return a - b
}
func div32_ssa(a, b float32) float32 {
	switch {
	}
	return a / b
}

func conv2Float64_ssa(a int8, b uint8, c int16, d uint16,
	e int32, f uint32, g int64, h uint64, i float32) (aa, bb, cc, dd, ee, ff, gg, hh, ii float64) {
	switch {
	}
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

func conv2Float32_ssa(a int8, b uint8, c int16, d uint16,
	e int32, f uint32, g int64, h uint64, i float64) (aa, bb, cc, dd, ee, ff, gg, hh, ii float32) {
	switch {
	}
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

	fails += fail32("+", add32_ssa, c, d, 7.0)
	fails += fail32("*", mul32_ssa, c, d, 12.0)
	fails += fail32("-", sub32_ssa, c, d, -1.0)
	fails += fail32("/", div32_ssa, c, d, 0.75)

	// denorm-squared should underflow to zero.
	fails += fail32("*", mul32_ssa, tiny, tiny, 0)

	// but should not underflow in float and in fact is exactly representable.
	fails += fail64("*", mul64_ssa, dtiny, dtiny, 1.9636373861190906e-90)

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

	if fails > 0 {
		fmt.Printf("Saw %v failures\n", fails)
		panic("Failed.")
	}
}
