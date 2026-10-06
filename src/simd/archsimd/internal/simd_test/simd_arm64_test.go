// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"fmt"
	"simd/archsimd"
	"strings"
	"testing"
)

func TestLookupOrZero(t *testing.T) {
	// Out-of-range indices produce zero lane value.
	x := []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	indices := []uint8{7, 6, 5, 4, 3, 2, 1, 0, 0xff, 8, 16, 9, 128, 10, 20, 11}
	want := []uint8{8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 0, 10, 0, 11, 0, 12}
	got := make([]uint8, len(x))
	archsimd.LoadUint8x16(x).LookupOrZero(archsimd.LoadUint8x16(indices)).StorePart(got)
	checkSlices(t, got, want)
}

func TestClMul(t *testing.T) {
	if !archsimd.ARM64.PMULL() {
		t.Skip("no carryless multiply")
	}
	var x = archsimd.LoadUint64x2([]uint64{1, 5})
	var y = archsimd.LoadUint64x2([]uint64{3, 9})

	foo := func(v archsimd.Uint64x2, s []uint64) {
		r := make([]uint64, 2, 2)
		v.StorePart(r)
		checkSlices[uint64](t, r, s)
	}

	foo(x.CarrylessMultiplyEven(y), []uint64{3, 0})
	foo(x.CarrylessMultiplyEvenOdd(y), []uint64{9, 0})
	foo(x.CarrylessMultiplyOddEven(y), []uint64{15, 0})
	foo(x.CarrylessMultiplyOdd(y), []uint64{45, 0})
	foo(y.CarrylessMultiplyEven(y), []uint64{5, 0})
}

//go:noinline
func addInt8sNoinline(a, b archsimd.Int8s) archsimd.Int8s { return a.Add(b) }

//go:noinline
func blackholeSVE() {}

// TestAddSVEAcrossCall passes scalable vectors across a real (non-inlined) ABI
// boundary, exercising the register/stack passing that size.go's simdify decides
// for SVE types.
func TestAddSVEAcrossCall(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	var a, b, got [32]int8
	for i := range a {
		a[i] = int8(i)
		b[i] = int8(2*i + 1)
	}
	x := archsimd.LoadInt8s(a[:])
	addInt8sNoinline(x, archsimd.LoadInt8s(b[:])).Store(got[:])
	for i := 0; i < x.Len(); i++ {
		if want := a[i] + b[i]; got[i] != want {
			t.Errorf("lane %d: got %d, want %d", i, got[i], want)
		}
	}
}

//go:noinline
func greaterInt8sNoinline(a, b archsimd.Int8s) archsimd.Mask8s { return a.Greater(b) }

// TestGreaterSVEMaskRoundTrip returns a mask across a non-inlined call, exercising
// the predicate memory round-trip (PSTR to return it, PLDR to reload it) that the
// mask ABI relies on, and checks it against a > b lane by lane.
func TestGreaterSVEMaskRoundTrip(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	var a, b [32]int8
	for i := range a {
		a[i] = int8(i - 8)
		b[i] = int8(2*i - 20)
	}
	var z archsimd.Int8s
	m := greaterInt8sNoinline(archsimd.LoadInt8s(a[:]), archsimd.LoadInt8s(b[:]))
	bits := maskBits(m)
	for i := 0; i < z.Len(); i++ {
		want := a[i] > b[i]
		got := bits>>i&1 == 1
		if got != want {
			t.Errorf("lane %d: got %v, want %v (a=%d b=%d)", i, got, want, a[i], b[i])
		}
	}
	if bits>>z.Len() != 0 {
		t.Errorf("bits beyond the vector length are set: %#x", bits)
	}
}

// TestAddSVESpill keeps a scalable vector live across a call, forcing the
// register allocator to spill and reload it (ZSTR/ZLDR).
func TestAddSVESpill(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	var a, b, got [32]int8
	for i := range a {
		a[i] = int8(i)
		b[i] = int8(100 - i)
	}
	sum := archsimd.LoadInt8s(a[:]).Add(archsimd.LoadInt8s(b[:]))
	blackholeSVE() // clobbers caller-saved regs; sum must survive via a spill
	sum.Store(got[:])
	for i := 0; i < sum.Len(); i++ {
		if want := a[i] + b[i]; got[i] != want {
			t.Errorf("lane %d: got %d, want %d", i, got[i], want)
		}
	}
}

// TestAddSaturatedSVE checks the generated saturating add: an explicit
// boundary case, then every integer shape against the generic emulation.
func TestAddSaturatedSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	var si, gi [32]int8
	for i := range si {
		si[i] = 100 // 100+100 saturates to +127
	}
	vi := archsimd.LoadInt8s(si[:])
	vi.AddSaturated(vi).Store(gi[:])
	for i := 0; i < vi.Len(); i++ {
		if gi[i] != 127 {
			t.Errorf("int8 lane %d: got %d, want 127", i, gi[i])
		}
	}
	testInt8sBinary(t, archsimd.Int8s.AddSaturated, addSaturatedSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.AddSaturated, addSaturatedSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.AddSaturated, addSaturatedSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.AddSaturated, addSaturatedSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.AddSaturated, addSaturatedSlice[uint8])
	testUint16sBinary(t, archsimd.Uint16s.AddSaturated, addSaturatedSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.AddSaturated, addSaturatedSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.AddSaturated, addSaturatedSlice[uint64])
}

// TestSubSaturatedSVE checks the generated saturating subtract: an explicit
// boundary case, then every integer shape against the generic emulation.
func TestSubSaturatedSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	var sx, sy, gi [32]int8
	for i := range sx {
		sx[i] = 100 // 100 - (-100) saturates to +127
		sy[i] = -100
	}
	x, y := archsimd.LoadInt8s(sx[:]), archsimd.LoadInt8s(sy[:])
	x.SubSaturated(y).Store(gi[:])
	for i := 0; i < x.Len(); i++ {
		if gi[i] != 127 {
			t.Errorf("int8 lane %d: got %d, want 127", i, gi[i])
		}
	}
	testInt8sBinary(t, archsimd.Int8s.SubSaturated, subSaturatedSlice[int8])
	testInt16sBinary(t, archsimd.Int16s.SubSaturated, subSaturatedSlice[int16])
	testInt32sBinary(t, archsimd.Int32s.SubSaturated, subSaturatedSlice[int32])
	testInt64sBinary(t, archsimd.Int64s.SubSaturated, subSaturatedSlice[int64])
	testUint8sBinary(t, archsimd.Uint8s.SubSaturated, subSaturatedSlice[uint8])
	testUint16sBinary(t, archsimd.Uint16s.SubSaturated, subSaturatedSlice[uint16])
	testUint32sBinary(t, archsimd.Uint32s.SubSaturated, subSaturatedSlice[uint32])
	testUint64sBinary(t, archsimd.Uint64s.SubSaturated, subSaturatedSlice[uint64])
}

func TestStringSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	want := func(v any) string {
		return "{" + strings.ReplaceAll(strings.Trim(fmt.Sprint(v), "[]"), " ", ",") + "}"
	}

	xs := make([]int8, archsimd.Int8s{}.Len())
	ys := make([]int64, archsimd.Int64s{}.Len())
	for i := range xs {
		xs[i] = int8(i % 2)
	}
	for i := range ys {
		ys[i] = int64(i % 2)
	}
	x := archsimd.LoadInt8s(xs)
	y := archsimd.LoadInt64s(ys)
	mx := x.Greater(archsimd.LoadInt8s(make([]int8, len(xs))))
	my := y.Greater(archsimd.LoadInt64s(make([]int64, len(ys))))

	if x.String() != want(xs) {
		t.Errorf("x=%s wanted %s", x, want(xs))
	}
	if y.String() != want(ys) {
		t.Errorf("y=%s wanted %s", y, want(ys))
	}
	if mx.String() != want(xs) {
		t.Errorf("mx=%s wanted %s", mx, want(xs))
	}
	if my.String() != want(ys) {
		t.Errorf("my=%s wanted %s", my, want(ys))
	}
	t.Logf("x=%s", x)
	t.Logf("y=%s", y)
	t.Logf("mx=%s", mx)
	t.Logf("my=%s", my)
}

//go:noinline
func keepAliveInt8s(archsimd.Int8s) {}

// namedMask8s and maxVia pass masks through types other than archsimd.Mask8s
// itself: a defined type, and the GC shape of a generic instantiation. Both
// must use the memory ABI of the mask they are made from, to agree with the
// concrete functions they call and to reach a P register when used.
type namedMask8s archsimd.Mask8s

//go:noinline
func greaterNamed(x, y archsimd.Int8s) namedMask8s { return namedMask8s(x.Greater(y)) }

//go:noinline
func maxVia[M archsimd.Mask8s](greater func(x, y archsimd.Int8s) M, x, y archsimd.Int8s) archsimd.Int8s {
	return x.IfElse(archsimd.Mask8s(greater(x, y)), y)
}

func TestMaskABISVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	n := archsimd.Int8s{}.Len()
	xs, ys := make([]int8, n), make([]int8, n)
	for i := range xs {
		xs[i], ys[i] = int8(i%5), int8(i%3)
	}
	x, y := archsimd.LoadInt8s(xs), archsimd.LoadInt8s(ys)
	check := func(name string, v archsimd.Int8s) {
		t.Helper()
		got := make([]int8, n)
		v.Store(got)
		for i := range got {
			if want := max(xs[i], ys[i]); got[i] != want {
				t.Errorf("%s: lane %d = %d, want %d", name, i, got[i], want)
			}
		}
	}
	check("generic", maxVia(archsimd.Int8s.Greater, x, y))
	check("named", x.IfElse(archsimd.Mask8s(greaterNamed(x, y)), y))
}

// TestIfElseSVE checks IfElse and Masked, and that the merging peephole keeps
// the same semantics whether or not it fires: x.Add(y).IfElse(m, x) folds into a
// predicated add, x.Add(y).IfElse(m, z) does not, and both must agree with a
// lane-by-lane reference.
func TestIfElseSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	n := archsimd.Int8s{}.Len()
	xs, ys, zs := make([]int8, n), make([]int8, n), make([]int8, n)
	for i := range xs {
		xs[i] = int8(i + 1)
		ys[i] = int8(i % 3) // active where xs[i] > ys[i], which alternates early on
		zs[i] = int8(-i - 1)
	}
	x, y, z := archsimd.LoadInt8s(xs), archsimd.LoadInt8s(ys), archsimd.LoadInt8s(zs)
	m := x.Greater(y)

	got := make([]int8, n)
	check := func(name string, v archsimd.Int8s, want func(i int, active bool) int8) {
		t.Helper()
		v.Store(got)
		for i := 0; i < n; i++ {
			if w := want(i, xs[i] > ys[i]); got[i] != w {
				t.Errorf("%s: lane %d = %d, want %d (x=%d y=%d)", name, i, got[i], w, xs[i], ys[i])
			}
		}
	}

	check("IfElse", x.IfElse(m, y), func(i int, active bool) int8 {
		if active {
			return xs[i]
		}
		return ys[i]
	})
	check("Masked", x.Masked(m), func(i int, active bool) int8 {
		if active {
			return xs[i]
		}
		return 0
	})
	// Folds into the merging-predicated add.
	check("Add.IfElse(x)", x.Add(y).IfElse(m, x), func(i int, active bool) int8 {
		if active {
			return xs[i] + ys[i]
		}
		return xs[i]
	})
	// Folds via commutativity.
	check("Add.IfElse(y)", x.Add(y).IfElse(m, y), func(i int, active bool) int8 {
		if active {
			return xs[i] + ys[i]
		}
		return ys[i]
	})
	// Folds behind a merging MOVPRFX: the else operand is neither source.
	check("Add.IfElse(z)", x.Add(y).IfElse(m, z), func(i int, active bool) int8 {
		if active {
			return xs[i] + ys[i]
		}
		return zs[i]
	})
	// ADD has no zeroing-predicated form, so Masked folds into the merging one
	// with the zero vector as its else operand.
	check("Add.Masked", x.Add(y).Masked(m), func(i int, active bool) int8 {
		if active {
			return xs[i] + ys[i]
		}
		return 0
	})

	// The two merges below differ only in which source the inactive lanes
	// keep. The merging machine op pins its result to that source and is not
	// commutative, or CSE — which canonicalizes the args of a commutative op —
	// would conflate the two values and give one of them the wrong else
	// operand.
	keepX := x.Add(y).IfElse(m, x)
	keepY := y.Add(x).IfElse(m, y)
	check("Add.IfElse cse keepX", keepX, func(i int, active bool) int8 {
		if active {
			return xs[i] + ys[i]
		}
		return xs[i]
	})
	check("Add.IfElse cse keepY", keepY, func(i int, active bool) int8 {
		if active {
			return xs[i] + ys[i]
		}
		return ys[i]
	})

	// SUB is not commutative, so only an "else" operand that is the destructive
	// one — the minuend — folds into the merging-predicated instruction.
	check("Sub.IfElse(x)", x.Sub(y).IfElse(m, x), func(i int, active bool) int8 {
		if active {
			return xs[i] - ys[i]
		}
		return xs[i]
	})
	// Does not fold, and must not silently become y-x.
	check("Sub.IfElse(y)", x.Sub(y).IfElse(m, y), func(i int, active bool) int8 {
		if active {
			return xs[i] - ys[i]
		}
		return ys[i]
	})
	// Does not fold: there is no prefixed form for a non-commutative operation.
	check("Sub.IfElse(z)", x.Sub(y).IfElse(m, z), func(i int, active bool) int8 {
		if active {
			return xs[i] - ys[i]
		}
		return zs[i]
	})
	check("Sub.Masked", x.Sub(y).Masked(m), func(i int, active bool) int8 {
		if active {
			return xs[i] - ys[i]
		}
		return 0
	})

	// Abs is predicated-only, so its unpredicated API runs under an all-true
	// predicate that a select can simply replace: IfElse becomes the merging
	// form and Masked the zeroing one, each a single instruction.
	absLane := func(v int8) int8 {
		if v < 0 {
			return -v
		}
		return v
	}
	check("Abs.IfElse(z)", x.Abs().ConvertToInt8().IfElse(m, z), func(i int, active bool) int8 {
		if active {
			return absLane(xs[i])
		}
		return zs[i]
	})
	check("Abs.IfElse(x)", x.Abs().ConvertToInt8().IfElse(m, x), func(i int, active bool) int8 {
		if active {
			return absLane(xs[i])
		}
		return xs[i]
	})
	check("Abs.Masked", x.Abs().Masked(m).ConvertToInt8(), func(i int, active bool) int8 {
		if active {
			return absLane(xs[i])
		}
		return 0
	})

	// The prefixed path with every operand still live afterwards, so the
	// destination can be none of them and the merging MOVPRFX has to place the
	// else operand itself.
	rz := x.Add(y).IfElse(m, z)
	keepAliveInt8s(x)
	keepAliveInt8s(y)
	keepAliveInt8s(z)
	check("Add.IfElse(z) with all live", rz, func(i int, active bool) int8 {
		if active {
			return xs[i] + ys[i]
		}
		return zs[i]
	})

	// The MOVPRFX path: x must survive the destructive predicated add.
	r := x.Add(y).IfElse(m, x)
	keepAliveInt8s(x)
	check("Add.IfElse(x) with x live", r, func(i int, active bool) int8 {
		if active {
			return xs[i] + ys[i]
		}
		return xs[i]
	})
	x.Store(got)
	for i := 0; i < n; i++ {
		if got[i] != xs[i] {
			t.Errorf("x clobbered by MOVPRFX: lane %d = %d, want %d", i, got[i], xs[i])
		}
	}
}

// TestMaskAllTrueSVE checks the all-true mask constructors: a select under an
// all-true mask returns its first operand in every lane, at each lane width.
func TestMaskAllTrueSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	check := func(name string, n int, sel func(i int) (got, want any)) {
		t.Helper()
		for i := 0; i < n; i++ {
			if got, want := sel(i); got != want {
				t.Errorf("%s: lane %d = %v, want %v", name, i, got, want)
			}
		}
	}

	xs8, ys8 := make([]int8, archsimd.Int8s{}.Len()), make([]int8, archsimd.Int8s{}.Len())
	for i := range xs8 {
		xs8[i], ys8[i] = int8(i+1), int8(-i-1)
	}
	g8 := make([]int8, len(xs8))
	archsimd.LoadInt8s(xs8).IfElse(archsimd.Mask8sAllTrue(), archsimd.LoadInt8s(ys8)).Store(g8)
	check("Mask8sAllTrue", len(xs8), func(i int) (any, any) { return g8[i], xs8[i] })

	xs16, ys16 := make([]int16, archsimd.Int16s{}.Len()), make([]int16, archsimd.Int16s{}.Len())
	for i := range xs16 {
		xs16[i], ys16[i] = int16(i+1), int16(-i-1)
	}
	g16 := make([]int16, len(xs16))
	archsimd.LoadInt16s(xs16).IfElse(archsimd.Mask16sAllTrue(), archsimd.LoadInt16s(ys16)).Store(g16)
	check("Mask16sAllTrue", len(xs16), func(i int) (any, any) { return g16[i], xs16[i] })

	xs32, ys32 := make([]int32, archsimd.Int32s{}.Len()), make([]int32, archsimd.Int32s{}.Len())
	for i := range xs32 {
		xs32[i], ys32[i] = int32(i+1), int32(-i-1)
	}
	g32 := make([]int32, len(xs32))
	archsimd.LoadInt32s(xs32).IfElse(archsimd.Mask32sAllTrue(), archsimd.LoadInt32s(ys32)).Store(g32)
	check("Mask32sAllTrue", len(xs32), func(i int) (any, any) { return g32[i], xs32[i] })

	xs64, ys64 := make([]int64, archsimd.Int64s{}.Len()), make([]int64, archsimd.Int64s{}.Len())
	for i := range xs64 {
		xs64[i], ys64[i] = int64(i+1), int64(-i-1)
	}
	g64 := make([]int64, len(xs64))
	archsimd.LoadInt64s(xs64).IfElse(archsimd.Mask64sAllTrue(), archsimd.LoadInt64s(ys64)).Store(g64)
	check("Mask64sAllTrue", len(xs64), func(i int) (any, any) { return g64[i], xs64[i] })
}
