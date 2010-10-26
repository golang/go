// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math"
	"rand"
	. "runtime"
	"testing"
)

// turn uint64 op into float64 op
func fop(f func(x, y uint64) uint64) func(x, y float64) float64 {
	return func(x, y float64) float64 {
		bx := math.Float64bits(x)
		by := math.Float64bits(y)
		return math.Float64frombits(f(bx, by))
	}
}

func add(x, y float64) float64 { return x + y }
func sub(x, y float64) float64 { return x - y }
func mul(x, y float64) float64 { return x * y }
func div(x, y float64) float64 { return x / y }

func TestFloat64(t *testing.T) {
	base := []float64{
		0,
		math.Copysign(0, -1),
		-1,
		1,
		math.NaN(),
		math.Inf(+1),
		math.Inf(-1),
		0.1,
		1.5,
		1.9999999999999998,     // all 1s mantissa
		1.3333333333333333,     // 1.010101010101...
		1.1428571428571428,     // 1.001001001001...
		1.112536929253601e-308, // first normal
		2,
		4,
		8,
		16,
		32,
		64,
		128,
		256,
		3,
		12,
		1234,
		123456,
		-0.1,
		-1.5,
		-1.9999999999999998,
		-1.3333333333333333,
		-1.1428571428571428,
		-2,
		-3,
		1e-200,
		1e-300,
		1e-310,
		5e-324,
		1e-105,
		1e-305,
		1e+200,
		1e+306,
		1e+307,
		1e+308,
	}
	all := make([]float64, 200)
	copy(all, base)
	for i := len(base); i < len(all); i++ {
		all[i] = rand.NormFloat64()
	}

	test(t, "+", add, fop(Fadd64), all)
	test(t, "-", sub, fop(Fsub64), all)
	if GOARCH != "386" { // 386 is not precise!
		test(t, "*", mul, fop(Fmul64), all)
		test(t, "/", div, fop(Fdiv64), all)
	}
}

// 64 -hw-> 32 -hw-> 64
func trunc32(f float64) float64 {
	return float64(float32(f))
}

// 64 -sw->32 -hw-> 64
func to32sw(f float64) float64 {
	return float64(math.Float32frombits(F64to32(math.Float64bits(f))))
}

// 64 -hw->32 -sw-> 64
func to64sw(f float64) float64 {
	return math.Float64frombits(F32to64(math.Float32bits(float32(f))))
}

// float64 -hw-> int64 -hw-> float64
func hwint64(f float64) float64 {
	return float64(int64(f))
}

// float64 -hw-> int32 -hw-> float64
func hwint32(f float64) float64 {
	return float64(int32(f))
}

// float64 -sw-> int64 -hw-> float64
func toint64sw(f float64) float64 {
	i, ok := F64toint(math.Float64bits(f))
	if !ok {
		// There's no right answer for out of range.
		// Match the hardware to pass the test.
		i = int64(f)
	}
	return float64(i)
}

// float64 -hw-> int64 -sw-> float64
func fromint64sw(f float64) float64 {
	return math.Float64frombits(Fintto64(int64(f)))
}

var nerr int

func err(t *testing.T, format string, args ...interface{}) {
	t.Errorf(format, args...)

	// cut errors off after a while.
	// otherwise we spend all our time
	// allocating memory to hold the
	// formatted output.
	if nerr++; nerr >= 10 {
		t.Fatal("too many errors")
	}
}

func test(t *testing.T, op string, hw, sw func(float64, float64) float64, all []float64) {
	for _, f := range all {
		for _, g := range all {
			h := hw(f, g)
			s := sw(f, g)
			if !same(h, s) {
				err(t, "%g %s %g = sw %g, hw %g\n", f, op, g, s, h)
			}
			testu(t, "to32", trunc32, to32sw, h)
			testu(t, "to64", trunc32, to64sw, h)
			testu(t, "toint64", hwint64, toint64sw, h)
			testu(t, "fromint64", hwint64, fromint64sw, h)
			testcmp(t, f, h)
			testcmp(t, h, f)
			testcmp(t, g, h)
			testcmp(t, h, g)
		}
	}
}

func testu(t *testing.T, op string, hw, sw func(float64) float64, v float64) {
	h := hw(v)
	s := sw(v)
	if !same(h, s) {
		err(t, "%s %g = sw %g, hw %g\n", op, v, s, h)
	}
}

func hwcmp(f, g float64) (cmp int, isnan bool) {
	switch {
	case f < g:
		return -1, false
	case f > g:
		return +1, false
	case f == g:
		return 0, false
	}
	return 0, true // must be NaN
}

func testcmp(t *testing.T, f, g float64) {
	hcmp, hisnan := hwcmp(f, g)
	scmp, sisnan := Fcmp64(math.Float64bits(f), math.Float64bits(g))
	if hcmp != scmp || hisnan != sisnan {
		err(t, "cmp(%g, %g) = sw %v, %v, hw %v, %v\n", f, g, scmp, sisnan, hcmp, hisnan)
	}
}

func same(f, g float64) bool {
	if math.IsNaN(f) && math.IsNaN(g) {
		return true
	}
	if math.Copysign(1, f) != math.Copysign(1, g) {
		return false
	}
	return f == g
}
