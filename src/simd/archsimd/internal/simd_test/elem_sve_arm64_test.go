// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"fmt"
	"math"
	"simd/archsimd"
	"testing"
	"unsafe"
)

// sameBits reports whether a and b have the same bits, so NaN payloads and
// the sign of zero count.
func sameBits[T any](a, b T) bool {
	n := unsafe.Sizeof(a)
	return string(unsafe.Slice((*byte)(unsafe.Pointer(&a)), n)) == string(unsafe.Slice((*byte)(unsafe.Pointer(&b)), n))
}

// testSVEElem checks get(x, i) against element i of the slice x was loaded
// from, for every lane i by a variable index and for vectors filled from pool
// at every offset, then checks that the first indexes past the runtime length
// panic. Elements are compared bit for bit.
func testSVEElem[T any, V interface{ Len() int }](t *testing.T, name string, pool []T, load func([]T) V, get func(V, uint8) T) {
	t.Helper()
	var s [sveMaxBytes]T
	var x V
	for start := range pool {
		for k := range s {
			s[k] = pool[(start+k)%len(pool)]
		}
		x = load(s[:])
		for i := range x.Len() {
			if got := get(x, uint8(i)); !sameBits(got, s[i]) {
				t.Errorf("%s(%v, %d) = %v, want %v", name, s[:x.Len()], i, got, s[i])
				return
			}
		}
	}
	for _, i := range []int{x.Len(), x.Len() + 1, 255} {
		mustPanic(t, fmt.Sprintf("%s(x, %d)", name, i), func() { get(x, uint8(i)) })
	}
}

func TestElemSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	testSVEElem(t, "Int8s.Elem", append(int8s, math.MinInt8, math.MaxInt8), archsimd.LoadInt8s, archsimd.Int8s.Elem)
	testSVEElem(t, "Uint8s.Elem", append(uint8s, math.MaxUint8), archsimd.LoadUint8s, archsimd.Uint8s.Elem)
	testSVEElem(t, "Int16s.Elem", append(int16s, math.MinInt16, math.MaxInt16), archsimd.LoadInt16s, archsimd.Int16s.Elem)
	testSVEElem(t, "Uint16s.Elem", append(uint16s, math.MaxUint16), archsimd.LoadUint16s, archsimd.Uint16s.Elem)
	testSVEElem(t, "Int32s.Elem", append(int32s, math.MinInt32, math.MaxInt32), archsimd.LoadInt32s, archsimd.Int32s.Elem)
	testSVEElem(t, "Uint32s.Elem", append(uint32s, math.MaxUint32), archsimd.LoadUint32s, archsimd.Uint32s.Elem)
	testSVEElem(t, "Float32s.Elem", float32s, archsimd.LoadFloat32s, archsimd.Float32s.Elem)
	testSVEElem(t, "Int64s.Elem", append(int64s, math.MinInt64, math.MaxInt64), archsimd.LoadInt64s, archsimd.Int64s.Elem)
	testSVEElem(t, "Uint64s.Elem", append(uint64s, math.MaxUint64), archsimd.LoadUint64s, archsimd.Uint64s.Elem)
	testSVEElem(t, "Float64s.Elem", float64s, archsimd.LoadFloat64s, archsimd.Float64s.Elem)
}

// TestElemConstSVE checks constant indexes, which take the same path as
// variable ones, and that a signed element is sign-extended where it is
// widened.
func TestElemConstSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no SVE")
	}
	var s [sveMaxBytes]int8
	for k := range s {
		s[k] = int8(-1 - k)
	}
	x := archsimd.LoadInt8s(s[:])
	if got := int64(x.Elem(0)); got != -1 {
		t.Errorf("Int8s.Elem(0) = %d, want -1", got)
	}
	if got := int64(x.Elem(15)); got != -16 {
		t.Errorf("Int8s.Elem(15) = %d, want -16", got)
	}
	// Lane 31 exists only when the vector is 256 bits or longer.
	if x.Len() > 31 {
		if got := int64(x.Elem(31)); got != -32 {
			t.Errorf("Int8s.Elem(31) = %d, want -32", got)
		}
	} else {
		mustPanic(t, "Int8s.Elem(31)", func() { x.Elem(31) })
	}
	var s16 [sveMaxBytes]int16
	s16[7] = math.MinInt16
	if got := int64(archsimd.LoadInt16s(s16[:]).Elem(7)); got != math.MinInt16 {
		t.Errorf("Int16s.Elem(7) = %d, want %d", got, math.MinInt16)
	}
	var s32 [sveMaxBytes]int32
	s32[3] = -5
	if got := int64(archsimd.LoadInt32s(s32[:]).Elem(3)); got != -5 {
		t.Errorf("Int32s.Elem(3) = %d, want -5", got)
	}
}
