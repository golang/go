// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"simd"
	"testing"
)

func assertEQ[T number](t *testing.T, x, y T, what string) {
	t.Helper()
	if x != y {
		t.Errorf("%v != %v, %s", x, y, what)
	}
}

func makeSlice[T number](l int) []T {
	x := make([]T, l)
	var i T
	for ; int(i) < len(x); i++ {
		x[int(i)] = i + 1
	}
	return x
}

func TestLen(t *testing.T) {
	var U64s simd.Uint64s
	var U32s simd.Uint32s
	var U16s simd.Uint16s
	var U8s simd.Uint8s

	assertEQ(t, 2*U64s.Len(), U32s.Len(), "2*U64s.Len()==U32s.Len()")
	assertEQ(t, 2*U32s.Len(), U16s.Len(), "2*U32.Len()==U16s.Len()")
	assertEQ(t, 2*U16s.Len(), U8s.Len(), "2*U16s.Len()==U8s.Len()")
}

func getElem64(x simd.Uint64s, i int) uint64 {
	s := make([]uint64, x.Len())
	x.Store(s)
	return s[i]
}

func getElem32(x simd.Uint32s, i int) uint32 {
	s := make([]uint32, x.Len())
	x.Store(s)
	return s[i]
}

func getElem16(x simd.Uint16s, i int) uint16 {
	s := make([]uint16, x.Len())
	x.Store(s)
	return s[i]
}

func getElem8(x simd.Uint8s, i int) uint8 {
	s := make([]uint8, x.Len())
	x.Store(s)
	return s[i]
}

func TestEndian(t *testing.T) {
	var U64s simd.Uint64s
	var U32s simd.Uint32s
	var U16s simd.Uint16s
	var U8s simd.Uint8s

	u64s := makeSlice[uint64](U64s.Len())
	u32s := makeSlice[uint32](U32s.Len())
	u16s := makeSlice[uint16](U16s.Len())
	u8s := makeSlice[uint8](U8s.Len())

	U64s = simd.LoadUint64s(u64s)
	U32s = simd.LoadUint32s(u32s)
	U16s = simd.LoadUint16s(u16s)
	U8s = simd.LoadUint8s(u8s)

	assertEQ(t, getElem64(U64s, 1), 2, "U64s[1] == 2")
	assertEQ(t, getElem32(U32s, 1), 2, "U32s[1] == 2")
	assertEQ(t, getElem16(U16s, 1), 2, "U16s[1] == 2")
	assertEQ(t, getElem8(U8s, 1), 2, "U8s[1] == 2")
}

func TestReshape(t *testing.T) {
	var U64s simd.Uint64s
	var U32s simd.Uint32s
	var U16s simd.Uint16s
	var U8s simd.Uint8s

	u64s := makeSlice[uint64](U64s.Len())
	u32s := makeSlice[uint32](U32s.Len())
	u16s := makeSlice[uint16](U16s.Len())
	u8s := makeSlice[uint8](U8s.Len())

	U64s = simd.LoadUint64s(u64s)
	U32s = simd.LoadUint32s(u32s)
	U16s = simd.LoadUint16s(u16s)
	U8s = simd.LoadUint8s(u8s)

	assertEQ(t, getElem8(U8s, 0), 1, "U8s[0] == 1")
	assertEQ(t, getElem8(U16s.ReshapeToUint8s(), 0), 1, "U16s.ReshapeToUint8s()[0] == 1")
	assertEQ(t, getElem8(U32s.ReshapeToUint8s(), 0), 1, "U32s.ReshapeToUint8s()[0] == 1")
	assertEQ(t, getElem8(U64s.ReshapeToUint8s(), 0), 1, "U64s.ReshapeToUint8s()[0] == 1")

	assertEQ(t, getElem16(U8s.ReshapeToUint16s(), 0), 0x0201, "U8s.ReshapeToUint16s()[0] == 0x0201")
	assertEQ(t, getElem16(U16s, 0), 1, "U16s[0] == 1")
	assertEQ(t, getElem16(U32s.ReshapeToUint16s(), 0), 1, "U32s.ReshapeToUint16s()[0] == 1")
	assertEQ(t, getElem16(U64s.ReshapeToUint16s(), 0), 1, "U64s.ReshapeToUint16s()[0] == 1")

	assertEQ(t, getElem32(U8s.ReshapeToUint32s(), 0), 0x04030201, "U8s.ReshapeToUint16s()[0] == 0x04030201")
	assertEQ(t, getElem32(U16s.ReshapeToUint32s(), 0), 0x00020001, "U16s.ReshapeToUint32s()[0] == 0x00020001")
	assertEQ(t, getElem32(U32s, 0), 1, "U32s[0] == 1")
	assertEQ(t, getElem32(U64s.ReshapeToUint32s(), 0), 1, "U64s.ReshapeToUint32s()[0] == 1")

	assertEQ(t, getElem64(U8s.ReshapeToUint64s(), 0), 0x0807060504030201, "U8s.ReshapeToUint64s()[0] == 0x0807060504030201")
	assertEQ(t, getElem64(U16s.ReshapeToUint64s(), 0), 0x0004000300020001, "U16s.ReshapeToUint64s()[0] == 0x0004000300020001")
	assertEQ(t, getElem64(U32s.ReshapeToUint64s(), 0), 0x0000000200000001, "U32s.ReshapeToUint64s()[0] == 0x0000000200000001")
	assertEQ(t, getElem64(U64s, 0), 1, "U64s[0] == 1")

	t.Logf("U8s=%v", U8s)
	t.Logf("U16s=%v", U16s)
	t.Logf("U32s=%v", U32s)
	t.Logf("U64s=%v", U64s)

}
