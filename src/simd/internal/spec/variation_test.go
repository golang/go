// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

import (
	"math"
	"testing"
)

func f32(bits uint32) float32 { return math.Float32frombits(bits) }
func f64(bits uint64) float64 { return math.Float64frombits(bits) }

// Interesting float32 values.
var (
	negZero32  = f32(0x80000000)
	minSub32   = f32(0x00000001) // Smallest positive subnormal
	maxSub32   = f32(0x007fffff) // Largest subnormal
	minNorm32  = f32(0x00800000) // Smallest positive normal
	inf32      = f32(0x7f800000)
	qnan32     = f32(0x7fc00000) // Quiet NaN
	qnanPay32  = f32(0x7fc00123) // Quiet NaN with a payload
	snan32     = f32(0x7f800001) // Signaling NaN
	negQnan32  = f32(0xffc00000) // Quiet NaN with the sign bit set
	maxFloat32 = float32(math.MaxFloat32)
)

// Interesting float64 values.
var (
	negZero64 = f64(0x8000000000000000)
	minSub64  = f64(0x0000000000000001)
	maxSub64  = f64(0x000fffffffffffff)
	minNorm64 = f64(0x0010000000000000)
	inf64     = math.Inf(1)
	qnan64    = f64(0x7ff8000000000000)
	qnanPay64 = f64(0x7ff8000000000123)
	snan64    = f64(0x7ff0000000000001)
	negQnan64 = f64(0xfff8000000000000)
)

// allPlatforms is every concrete platform, plus PlatformAny.
func allPlatforms() []Platform {
	var ps []Platform
	for p := range numPlatforms {
		ps = append(ps, p)
	}
	return append(ps, PlatformAny)
}

// checkPermits checks that, on plat, v permits every value in yes and none of
// the values in no.
func checkPermits[E EltOrMask](t *testing.T, v Variation[E], plat Platform, yes, no []E) {
	t.Helper()
	for _, x := range yes {
		if !v.Permits(x, plat) {
			t.Errorf("%v on %v: %v not permitted, want permitted", v, plat, x)
		}
	}
	for _, x := range no {
		if v.Permits(x, plat) {
			t.Errorf("%v on %v: %v permitted, want not permitted", v, plat, x)
		}
	}
}

// checkPermitsAll is checkPermits on every platform.
func checkPermitsAll[E EltOrMask](t *testing.T, v Variation[E], yes, no []E) {
	t.Helper()
	for _, p := range allPlatforms() {
		checkPermits(t, v, p, yes, no)
	}
}

func checkCanon[E EltOrMask](t *testing.T, v Variation[E], want E) {
	t.Helper()
	if got := v.Canon(); !equalExact(got, want) {
		t.Errorf("%v: Canon() = %v, want %v", v, got, want)
	}
}

func checkString[E EltOrMask](t *testing.T, v Variation[E], want string) {
	t.Helper()
	if got := v.String(); got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
}

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s did not panic", name)
		}
	}()
	f()
}

type equalExactCase[E EltOrMask] struct {
	a, b E
	want bool
}

func checkEqualExact[E EltOrMask](t *testing.T, cases []equalExactCase[E]) {
	t.Helper()
	for _, c := range cases {
		// equalExact must be symmetric.
		if got := equalExact(c.a, c.b); got != c.want {
			t.Errorf("equalExact(%v, %v) = %v, want %v", c.a, c.b, got, c.want)
		}
		if got := equalExact(c.b, c.a); got != c.want {
			t.Errorf("equalExact(%v, %v) = %v, want %v", c.b, c.a, got, c.want)
		}
	}
}

func TestEqualExact(t *testing.T) {
	checkEqualExact(t, []equalExactCase[int8]{
		{-1, -1, true},
		{-1, 1, false},
		{math.MinInt8, math.MinInt8, true},
		{math.MinInt8, math.MaxInt8, false},
	})
	checkEqualExact(t, []equalExactCase[uint64]{
		{math.MaxUint64, math.MaxUint64, true},
		{math.MaxUint64, math.MaxUint64 - 1, false},
	})
	checkEqualExact(t, []equalExactCase[Mask32]{
		{^Mask32(0), ^Mask32(0), true},
		{0, 0, true},
		{0, ^Mask32(0), false},
	})

	checkEqualExact(t, []equalExactCase[float32]{
		{1, 1, true},
		{1, math.Nextafter32(1, 2), false},
		// Signed zeroes compare equal with ==, but are not identical.
		{0, negZero32, false},
		{negZero32, negZero32, true},
		// Subnormals are distinct from zero and from each other.
		{minSub32, minSub32, true},
		{minSub32, 0, false},
		{maxSub32, minNorm32, false},
		{inf32, inf32, true},
		{inf32, -inf32, false},
		{inf32, maxFloat32, false},
		// All NaNs are considered equal, regardless of payload, quiet vs.
		// signaling, or sign, even though NaN != NaN.
		{qnan32, qnan32, true},
		{qnan32, qnanPay32, true},
		{qnan32, snan32, true},
		{qnan32, negQnan32, true},
		{snan32, negQnan32, true},
		{qnan32, 0, false},
		{qnan32, inf32, false},
		{snan32, inf32, false}, // These differ in only the lowest bit.
	})

	checkEqualExact(t, []equalExactCase[float64]{
		{1, 1, true},
		{1, math.Nextafter(1, 2), false},
		{0, negZero64, false},
		{negZero64, negZero64, true},
		{minSub64, minSub64, true},
		{minSub64, 0, false},
		{maxSub64, minNorm64, false},
		{inf64, inf64, true},
		{inf64, -inf64, false},
		{inf64, math.MaxFloat64, false},
		{qnan64, qnan64, true},
		{qnan64, qnanPay64, true},
		{qnan64, snan64, true},
		{qnan64, negQnan64, true},
		{snan64, negQnan64, true},
		{qnan64, 0, false},
		{qnan64, inf64, false},
		{snan64, inf64, false},
	})
}

func TestExact(t *testing.T) {
	v := exact[int32](5)
	checkCanon(t, v, 5)
	checkString(t, v, "5")
	checkPermitsAll(t, v, []int32{5}, []int32{4, 6, 0})

	// exact compares like equalExact.
	checkPermitsAll(t, exact(qnan32), []float32{qnan32, snan32, negQnan32}, []float32{0, inf32})
	checkPermitsAll(t, exact(float32(0)), []float32{0}, []float32{negZero32})

	// The zero Variation is the exact zero value.
	var zero Variation[int16]
	checkCanon(t, zero, 0)
	checkPermitsAll(t, zero, []int16{0}, []int16{1, -1})
}

func TestPerPlatform(t *testing.T) {
	v := perPlatform(map[Platform]int32{
		PlatformARM64: 2,
		PlatformWASM:  3,
	})
	// The canonical value is the value for the smallest present platform.
	checkCanon(t, v, 2)
	checkString(t, v, "{arm64:2 wasm:3}")

	checkPermits(t, v, PlatformARM64, []int32{2}, []int32{3, 0})
	checkPermits(t, v, PlatformWASM, []int32{3}, []int32{2, 0})
	// A platform without a value permits nothing.
	checkPermits(t, v, PlatformAMD64, nil, []int32{2, 3, 0})
	// PlatformAny permits any platform's value.
	checkPermits(t, v, PlatformAny, []int32{2, 3}, []int32{0})
	// Unknown platforms permit nothing.
	checkPermits(t, v, numPlatforms, nil, []int32{2, 3})
	checkPermits(t, v, Platform(-2), nil, []int32{2, 3})

	// Values compare like equalExact.
	vf := perPlatform(map[Platform]float32{
		PlatformAMD64: qnan32,
		PlatformARM64: 0,
	})
	checkPermits(t, vf, PlatformAMD64, []float32{snan32}, []float32{0})
	checkPermits(t, vf, PlatformARM64, []float32{0}, []float32{negZero32, qnan32})

	mustPanic(t, "perPlatform(empty)", func() { perPlatform(map[Platform]int32{}) })
	mustPanic(t, "perPlatform(no recognized platforms)", func() {
		perPlatform(map[Platform]int32{PlatformAny: 1})
	})
}

func TestAnyOf(t *testing.T) {
	// A single value is just exact.
	v1 := anyOf[int32](7)
	checkCanon(t, v1, 7)
	checkString(t, v1, "7")
	checkPermitsAll(t, v1, []int32{7}, []int32{6, 8})

	v := anyOf[int32](1, 2, 3)
	// The canonical value is the first value.
	checkCanon(t, v, 1)
	checkString(t, v, "{1|2|3}")
	checkPermitsAll(t, v, []int32{1, 2, 3}, []int32{0, 4})

	// Values compare like equalExact.
	checkPermitsAll(t, anyOf(float32(0), qnan32), []float32{0, snan32}, []float32{negZero32, 1})

	mustPanic(t, "anyOf()", func() { anyOf[int32]() })
}

func TestUndef(t *testing.T) {
	v := undef[int8]()
	checkCanon(t, v, 0)
	checkString(t, v, "undef")
	checkPermitsAll(t, v, []int8{0, 1, -1, math.MinInt8, math.MaxInt8}, nil)

	checkPermitsAll(t, undef[float32](), []float32{0, negZero32, inf32, -inf32, qnan32, snan32}, nil)
}

func TestULPs(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		// up returns the float32 n representable values above x (below if n is
		// negative), assuming no sign change.
		up := func(x float32, n int32) float32 {
			return f32(uint32(int32(math.Float32bits(x)) + n))
		}

		// ulps(x, 0) is exact.
		v0 := ulps(float32(1), 0)
		checkString(t, v0, "1")
		checkPermitsAll(t, v0, []float32{1}, []float32{up(1, 1), up(1, -1)})

		v := ulps(float32(1), 2)
		checkCanon(t, v, 1)
		checkString(t, v, "1±2ulp")
		// Distance is in representable values, not magnitude: the ULP below
		// 1 is half the ULP above 1. The platform is ignored.
		checkPermitsAll(t, v,
			[]float32{1, up(1, 1), up(1, 2), up(1, -1), up(1, -2)},
			[]float32{up(1, 3), up(1, -3), -1, 0})

		// Crossing a binade boundary: the largest float32 below 2 is 1 ULP
		// from 2.
		checkPermitsAll(t, ulps(float32(2), 1), []float32{up(2, -1)}, []float32{up(2, -2)})

		// Crossing the subnormal/normal boundary, and within subnormals.
		checkPermitsAll(t, ulps(minNorm32, 1), []float32{maxSub32}, []float32{up(maxSub32, -1)})
		checkPermitsAll(t, ulps(minSub32, 1), []float32{0, up(minSub32, 1)}, []float32{up(minSub32, 2)})

		// The sign must match, even across zero: ulps(+0, 1) permits the
		// smallest positive subnormal, but not -0 or the smallest negative
		// subnormal.
		checkPermitsAll(t, ulps(float32(0), 1), []float32{0, minSub32}, []float32{negZero32, -minSub32})
		checkPermitsAll(t, ulps(minSub32, 1), nil, []float32{negZero32, -minSub32})
		checkPermitsAll(t, ulps(negZero32, 1), []float32{negZero32, -minSub32}, []float32{0, minSub32})

		// Negative values.
		checkPermitsAll(t, ulps(float32(-1), 1),
			[]float32{-1, up(-1, 1), up(-1, -1)},
			[]float32{up(-1, 2), 1})

		// Infinities must match exactly, even though +Inf's bit pattern is
		// adjacent to MaxFloat32's.
		checkPermitsAll(t, ulps(maxFloat32, 1), []float32{up(maxFloat32, -1)}, []float32{inf32})
		checkPermitsAll(t, ulps(inf32, 1), []float32{inf32}, []float32{maxFloat32, -inf32, qnan32})
		checkPermitsAll(t, ulps(-inf32, 1), []float32{-inf32}, []float32{-maxFloat32, inf32})

		// NaN permits any NaN, and nothing else, and no finite value permits
		// NaN. (snan32 differs from +Inf in only the lowest bit.)
		checkPermitsAll(t, ulps(qnan32, 1), []float32{qnan32, snan32, negQnan32}, []float32{0, inf32})
		checkPermitsAll(t, ulps(float32(1), 1<<30), nil, []float32{qnan32, snan32, negQnan32})
		checkPermitsAll(t, ulps(maxFloat32, 1<<30), nil, []float32{snan32})

		// Large distances: 1 and 2 are 1<<23 representable values apart.
		checkPermitsAll(t, ulps(float32(1), 1<<23), []float32{2}, []float32{up(2, 1)})

		// Very large distances. The bit patterns of values with opposite signs
		// are at least 1<<31 apart, so this is where the sign check matters.
		// Also, the distance must not be truncated to 32 bits.
		checkPermitsAll(t, ulps(minSub32, 1<<31), []float32{maxFloat32}, []float32{negZero32, -minSub32})
		checkPermitsAll(t, ulps(float32(1), 1<<32), []float32{0, 2, maxFloat32}, []float32{-1, inf32, qnan32})
		checkPermitsAll(t, ulps(float32(1), math.MaxUint64), []float32{0, minSub32, maxFloat32}, []float32{negZero32, -1, inf32, qnan32})
	})

	t.Run("float64", func(t *testing.T) {
		up := func(x float64, n int64) float64 {
			return f64(uint64(int64(math.Float64bits(x)) + n))
		}

		checkPermitsAll(t, ulps(1.0, 0), []float64{1}, []float64{up(1, 1)})

		v := ulps(1.0, 2)
		checkCanon(t, v, 1)
		checkString(t, v, "1±2ulp")
		checkPermitsAll(t, v,
			[]float64{1, up(1, 1), up(1, 2), up(1, -1), up(1, -2)},
			[]float64{up(1, 3), up(1, -3), -1, 0})

		checkPermitsAll(t, ulps(2.0, 1), []float64{up(2, -1)}, []float64{up(2, -2)})

		checkPermitsAll(t, ulps(minNorm64, 1), []float64{maxSub64}, []float64{up(maxSub64, -1)})
		checkPermitsAll(t, ulps(minSub64, 1), []float64{0, up(minSub64, 1)}, []float64{up(minSub64, 2)})

		checkPermitsAll(t, ulps(0.0, 1), []float64{0, minSub64}, []float64{negZero64, -minSub64})
		checkPermitsAll(t, ulps(negZero64, 1), []float64{negZero64, -minSub64}, []float64{0, minSub64})

		checkPermitsAll(t, ulps(-1.0, 1),
			[]float64{-1, up(-1, 1), up(-1, -1)},
			[]float64{up(-1, 2), 1})

		checkPermitsAll(t, ulps(math.MaxFloat64, 1), []float64{up(math.MaxFloat64, -1)}, []float64{inf64})
		checkPermitsAll(t, ulps(inf64, 1), []float64{inf64}, []float64{math.MaxFloat64, -inf64, qnan64})
		checkPermitsAll(t, ulps(-inf64, 1), []float64{-inf64}, []float64{-math.MaxFloat64, inf64})

		checkPermitsAll(t, ulps(qnan64, 1), []float64{qnan64, snan64, negQnan64}, []float64{0, inf64})
		checkPermitsAll(t, ulps(1.0, 1<<30), nil, []float64{qnan64, snan64, negQnan64})
		checkPermitsAll(t, ulps(math.MaxFloat64, 1<<30), nil, []float64{snan64})

		// 1 and 2 are 1<<52 representable values apart.
		checkPermitsAll(t, ulps(1.0, 1<<52), []float64{2}, []float64{up(2, 1)})

		// The bit patterns of values with opposite signs are at least 1<<63
		// apart, so this is where the sign check matters.
		checkPermitsAll(t, ulps(minSub64, 1<<63), []float64{math.MaxFloat64}, []float64{negZero64, -minSub64})
		checkPermitsAll(t, ulps(1.0, math.MaxUint64), []float64{0, minSub64, math.MaxFloat64}, []float64{negZero64, -1, inf64, qnan64})
	})
}
