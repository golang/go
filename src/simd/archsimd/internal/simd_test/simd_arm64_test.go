// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
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
