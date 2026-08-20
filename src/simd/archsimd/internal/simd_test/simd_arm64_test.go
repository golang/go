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
// mask ABI relies on. It then stores the mask, reloads it with LoadMask8s, and
// checks both agree with a > b lane by lane.
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
	bits := make([]uint16, sveMaskUint16s)
	m.Store(bits)

	reloaded := make([]uint16, sveMaskUint16s)
	archsimd.LoadMask8s(bits).Store(reloaded)

	for i := 0; i < z.Len(); i++ {
		want := a[i] > b[i]
		got := bits[i/16]>>uint(i%16)&1 == 1
		if got != want {
			t.Errorf("lane %d: got %v, want %v (a=%d b=%d)", i, got, want, a[i], b[i])
		}
		if reloaded[i/16] != bits[i/16] {
			t.Errorf("LoadMask8s round-trip mismatch at uint16 %d: %#x vs %#x", i/16, reloaded[i/16], bits[i/16])
		}
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

// TestAddSaturatedSVE checks that the generated saturating add saturates.
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
	var su, gu [32]uint8
	for i := range su {
		su[i] = 200 // 200+200 saturates to 255
	}
	vu := archsimd.LoadUint8s(su[:])
	vu.AddSaturated(vu).Store(gu[:])
	for i := 0; i < vu.Len(); i++ {
		if gu[i] != 255 {
			t.Errorf("uint8 lane %d: got %d, want 255", i, gu[i])
		}
	}
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
