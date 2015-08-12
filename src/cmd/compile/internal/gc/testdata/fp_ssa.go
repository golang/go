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
	}
	return 0
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

	if fails > 0 {
		fmt.Printf("Saw %v failures\n", fails)
		panic("Failed.")
	}
}
