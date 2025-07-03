// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests arithmetic expressions

package main

import (
	"math"
	"runtime"
	"testing"
)

const (
	y = 0x0fffFFFF
)

var (
	g8  int8
	g16 int16
	g32 int32
	g64 int64
)

//go:noinline
func lshNop1(x uint64) uint64 {
	// two outer shifts should be removed
	return (((x << 5) >> 2) << 2)
}

//go:noinline
func lshNop2(x uint64) uint64 {
	return (((x << 5) >> 2) << 3)
}

//go:noinline
func lshNop3(x uint64) uint64 {
	return (((x << 5) >> 2) << 6)
}

//go:noinline
func lshNotNop(x uint64) uint64 {
	// outer shift can't be removed
	return (((x << 5) >> 2) << 1)
}

//go:noinline
func rshNop1(x uint64) uint64 {
	return (((x >> 5) << 2) >> 2)
}

//go:noinline
func rshNop2(x uint64) uint64 {
	return (((x >> 5) << 2) >> 3)
}

//go:noinline
func rshNop3(x uint64) uint64 {
	return (((x >> 5) << 2) >> 6)
}

//go:noinline
func rshNotNop(x uint64) uint64 {
	return (((x >> 5) << 2) >> 1)
}

func testShiftRemoval(t *testing.T) {
	allSet := ^uint64(0)
	if want, got := uint64(0x7ffffffffffffff), rshNop1(allSet); want != got {
		t.Errorf("testShiftRemoval rshNop1 failed, wanted %d got %d", want, got)
	}
	if want, got := uint64(0x3ffffffffffffff), rshNop2(allSet); want != got {
		t.Errorf("testShiftRemoval rshNop2 failed, wanted %d got %d", want, got)
	}
	if want, got := uint64(0x7fffffffffffff), rshNop3(allSet); want != got {
		t.Errorf("testShiftRemoval rshNop3 failed, wanted %d got %d", want, got)
	}
	if want, got := uint64(0xffffffffffffffe), rshNotNop(allSet); want != got {
		t.Errorf("testShiftRemoval rshNotNop failed, wanted %d got %d", want, got)
	}
	if want, got := uint64(0xffffffffffffffe0), lshNop1(allSet); want != got {
		t.Errorf("testShiftRemoval lshNop1 failed, wanted %d got %d", want, got)
	}
	if want, got := uint64(0xffffffffffffffc0), lshNop2(allSet); want != got {
		t.Errorf("testShiftRemoval lshNop2 failed, wanted %d got %d", want, got)
	}
	if want, got := uint64(0xfffffffffffffe00), lshNop3(allSet); want != got {
		t.Errorf("testShiftRemoval lshNop3 failed, wanted %d got %d", want, got)
	}
	if want, got := uint64(0x7ffffffffffffff0), lshNotNop(allSet); want != got {
		t.Errorf("testShiftRemoval lshNotNop failed, wanted %d got %d", want, got)
	}
}

//go:noinline
func parseLE64(b []byte) uint64 {
	// skip the first two bytes, and parse the remaining 8 as a uint64
	return uint64(b[2]) | uint64(b[3])<<8 | uint64(b[4])<<16 | uint64(b[5])<<24 |
		uint64(b[6])<<32 | uint64(b[7])<<40 | uint64(b[8])<<48 | uint64(b[9])<<56
}

//go:noinline
func parseLE32(b []byte) uint32 {
	return uint32(b[2]) | uint32(b[3])<<8 | uint32(b[4])<<16 | uint32(b[5])<<24
}

//go:noinline
func parseLE16(b []byte) uint16 {
	return uint16(b[2]) | uint16(b[3])<<8
}

// testLoadCombine tests for issue #14694 where load combining didn't respect the pointer offset.
func testLoadCombine(t *testing.T) {
	testData := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09}
	if want, got := uint64(0x0908070605040302), parseLE64(testData); want != got {
		t.Errorf("testLoadCombine failed, wanted %d got %d", want, got)
	}
	if want, got := uint32(0x05040302), parseLE32(testData); want != got {
		t.Errorf("testLoadCombine failed, wanted %d got %d", want, got)
	}
	if want, got := uint16(0x0302), parseLE16(testData); want != got {
		t.Errorf("testLoadCombine failed, wanted %d got %d", want, got)
	}
}

var loadSymData = [...]byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}

func testLoadSymCombine(t *testing.T) {
	w2 := uint16(0x0201)
	g2 := uint16(loadSymData[0]) | uint16(loadSymData[1])<<8
	if g2 != w2 {
		t.Errorf("testLoadSymCombine failed, wanted %d got %d", w2, g2)
	}
	w4 := uint32(0x04030201)
	g4 := uint32(loadSymData[0]) | uint32(loadSymData[1])<<8 |
		uint32(loadSymData[2])<<16 | uint32(loadSymData[3])<<24
	if g4 != w4 {
		t.Errorf("testLoadSymCombine failed, wanted %d got %d", w4, g4)
	}
	w8 := uint64(0x0807060504030201)
	g8 := uint64(loadSymData[0]) | uint64(loadSymData[1])<<8 |
		uint64(loadSymData[2])<<16 | uint64(loadSymData[3])<<24 |
		uint64(loadSymData[4])<<32 | uint64(loadSymData[5])<<40 |
		uint64(loadSymData[6])<<48 | uint64(loadSymData[7])<<56
	if g8 != w8 {
		t.Errorf("testLoadSymCombine failed, wanted %d got %d", w8, g8)
	}
}

//go:noinline
func invalidAdd_ssa(x uint32) uint32 {
	return x + y + y + y + y + y + y + y + y + y + y + y + y + y + y + y + y + y
}

//go:noinline
func invalidSub_ssa(x uint32) uint32 {
	return x - y - y - y - y - y - y - y - y - y - y - y - y - y - y - y - y - y
}

//go:noinline
func invalidMul_ssa(x uint32) uint32 {
	return x * y * y * y * y * y * y * y * y * y * y * y * y * y * y * y * y * y
}

// testLargeConst tests a situation where larger than 32 bit consts were passed to ADDL
// causing an invalid instruction error.
func testLargeConst(t *testing.T) {
	if want, got := uint32(268435440), invalidAdd_ssa(1); want != got {
		t.Errorf("testLargeConst add failed, wanted %d got %d", want, got)
	}
	if want, got := uint32(4026531858), invalidSub_ssa(1); want != got {
		t.Errorf("testLargeConst sub failed, wanted %d got %d", want, got)
	}
	if want, got := uint32(268435455), invalidMul_ssa(1); want != got {
		t.Errorf("testLargeConst mul failed, wanted %d got %d", want, got)
	}
}

// testArithRshConst ensures that "const >> const" right shifts correctly perform
// sign extension on the lhs constant
func testArithRshConst(t *testing.T) {
	wantu := uint64(0x4000000000000000)
	if got := arithRshuConst_ssa(); got != wantu {
		t.Errorf("arithRshuConst failed, wanted %d got %d", wantu, got)
	}

	wants := int64(-0x4000000000000000)
	if got := arithRshConst_ssa(); got != wants {
		t.Errorf("arithRshConst failed, wanted %d got %d", wants, got)
	}
}

//go:noinline
func arithRshuConst_ssa() uint64 {
	y := uint64(0x8000000000000001)
	z := uint64(1)
	return uint64(y >> z)
}

//go:noinline
func arithRshConst_ssa() int64 {
	y := int64(-0x8000000000000000)
	z := uint64(1)
	return int64(y >> z)
}

//go:noinline
func arithConstShift_ssa(x int64) int64 {
	return x >> 100
}

// testArithConstShift tests that right shift by large constants preserve
// the sign of the input.
func testArithConstShift(t *testing.T) {
	want := int64(-1)
	if got := arithConstShift_ssa(-1); want != got {
		t.Errorf("arithConstShift_ssa(-1) failed, wanted %d got %d", want, got)
	}
	want = 0
	if got := arithConstShift_ssa(1); want != got {
		t.Errorf("arithConstShift_ssa(1) failed, wanted %d got %d", want, got)
	}
}

// overflowConstShift_ssa verifies that constant folding for shift
// doesn't wrap (i.e. x << MAX_INT << 1 doesn't get folded to x << 0).
//
//go:noinline
func overflowConstShift64_ssa(x int64) int64 {
	return x << uint64(0xffffffffffffffff) << uint64(1)
}

//go:noinline
func overflowConstShift32_ssa(x int64) int32 {
	return int32(x) << uint32(0xffffffff) << uint32(1)
}

//go:noinline
func overflowConstShift16_ssa(x int64) int16 {
	return int16(x) << uint16(0xffff) << uint16(1)
}

//go:noinline
func overflowConstShift8_ssa(x int64) int8 {
	return int8(x) << uint8(0xff) << uint8(1)
}

func testOverflowConstShift(t *testing.T) {
	want := int64(0)
	for x := int64(-127); x < int64(127); x++ {
		got := overflowConstShift64_ssa(x)
		if want != got {
			t.Errorf("overflowShift64 failed, wanted %d got %d", want, got)
		}
		got = int64(overflowConstShift32_ssa(x))
		if want != got {
			t.Errorf("overflowShift32 failed, wanted %d got %d", want, got)
		}
		got = int64(overflowConstShift16_ssa(x))
		if want != got {
			t.Errorf("overflowShift16 failed, wanted %d got %d", want, got)
		}
		got = int64(overflowConstShift8_ssa(x))
		if want != got {
			t.Errorf("overflowShift8 failed, wanted %d got %d", want, got)
		}
	}
}

//go:noinline
func rsh64x64ConstOverflow8(x int8) int64 {
	return int64(x) >> 9
}

//go:noinline
func rsh64x64ConstOverflow16(x int16) int64 {
	return int64(x) >> 17
}

//go:noinline
func rsh64x64ConstOverflow32(x int32) int64 {
	return int64(x) >> 33
}

func testArithRightShiftConstOverflow(t *testing.T) {
	allSet := int64(-1)
	if got, want := rsh64x64ConstOverflow8(0x7f), int64(0); got != want {
		t.Errorf("rsh64x64ConstOverflow8 failed: got %v, want %v", got, want)
	}
	if got, want := rsh64x64ConstOverflow16(0x7fff), int64(0); got != want {
		t.Errorf("rsh64x64ConstOverflow16 failed: got %v, want %v", got, want)
	}
	if got, want := rsh64x64ConstOverflow32(0x7ffffff), int64(0); got != want {
		t.Errorf("rsh64x64ConstOverflow32 failed: got %v, want %v", got, want)
	}
	if got, want := rsh64x64ConstOverflow8(int8(-1)), allSet; got != want {
		t.Errorf("rsh64x64ConstOverflow8 failed: got %v, want %v", got, want)
	}
	if got, want := rsh64x64ConstOverflow16(int16(-1)), allSet; got != want {
		t.Errorf("rsh64x64ConstOverflow16 failed: got %v, want %v", got, want)
	}
	if got, want := rsh64x64ConstOverflow32(int32(-1)), allSet; got != want {
		t.Errorf("rsh64x64ConstOverflow32 failed: got %v, want %v", got, want)
	}
}

//go:noinline
func rsh64Ux64ConstOverflow8(x uint8) uint64 {
	return uint64(x) >> 9
}

//go:noinline
func rsh64Ux64ConstOverflow16(x uint16) uint64 {
	return uint64(x) >> 17
}

//go:noinline
func rsh64Ux64ConstOverflow32(x uint32) uint64 {
	return uint64(x) >> 33
}

func testRightShiftConstOverflow(t *testing.T) {
	if got, want := rsh64Ux64ConstOverflow8(0xff), uint64(0); got != want {
		t.Errorf("rsh64Ux64ConstOverflow8 failed: got %v, want %v", got, want)
	}
	if got, want := rsh64Ux64ConstOverflow16(0xffff), uint64(0); got != want {
		t.Errorf("rsh64Ux64ConstOverflow16 failed: got %v, want %v", got, want)
	}
	if got, want := rsh64Ux64ConstOverflow32(0xffffffff), uint64(0); got != want {
		t.Errorf("rsh64Ux64ConstOverflow32 failed: got %v, want %v", got, want)
	}
}

// test64BitConstMult tests that rewrite rules don't fold 64 bit constants
// into multiply instructions.
func test64BitConstMult(t *testing.T) {
	want := int64(103079215109)
	if got := test64BitConstMult_ssa(1, 2); want != got {
		t.Errorf("test64BitConstMult failed, wanted %d got %d", want, got)
	}
}

//go:noinline
func test64BitConstMult_ssa(a, b int64) int64 {
	return 34359738369*a + b*34359738370
}

// test64BitConstAdd tests that rewrite rules don't fold 64 bit constants
// into add instructions.
func test64BitConstAdd(t *testing.T) {
	want := int64(3567671782835376650)
	if got := test64BitConstAdd_ssa(1, 2); want != got {
		t.Errorf("test64BitConstAdd failed, wanted %d got %d", want, got)
	}
}

//go:noinline
func test64BitConstAdd_ssa(a, b int64) int64 {
	return a + 575815584948629622 + b + 2991856197886747025
}

// testRegallocCVSpill tests that regalloc spills a value whose last use is the
// current value.
func testRegallocCVSpill(t *testing.T) {
	want := int8(-9)
	if got := testRegallocCVSpill_ssa(1, 2, 3, 4); want != got {
		t.Errorf("testRegallocCVSpill failed, wanted %d got %d", want, got)
	}
}

//go:noinline
func testRegallocCVSpill_ssa(a, b, c, d int8) int8 {
	return a + -32 + b + 63*c*-87*d
}

func testBitwiseLogic(t *testing.T) {
	a, b := uint32(57623283), uint32(1314713839)
	if want, got := uint32(38551779), testBitwiseAnd_ssa(a, b); want != got {
		t.Errorf("testBitwiseAnd failed, wanted %d got %d", want, got)
	}
	if want, got := uint32(1333785343), testBitwiseOr_ssa(a, b); want != got {
		t.Errorf("testBitwiseOr failed, wanted %d got %d", want, got)
	}
	if want, got := uint32(1295233564), testBitwiseXor_ssa(a, b); want != got {
		t.Errorf("testBitwiseXor failed, wanted %d got %d", want, got)
	}
	if want, got := int32(832), testBitwiseLsh_ssa(13, 4, 2); want != got {
		t.Errorf("testBitwiseLsh failed, wanted %d got %d", want, got)
	}
	if want, got := int32(0), testBitwiseLsh_ssa(13, 25, 15); want != got {
		t.Errorf("testBitwiseLsh failed, wanted %d got %d", want, got)
	}
	if want, got := int32(0), testBitwiseLsh_ssa(-13, 25, 15); want != got {
		t.Errorf("testBitwiseLsh failed, wanted %d got %d", want, got)
	}
	if want, got := int32(-13), testBitwiseRsh_ssa(-832, 4, 2); want != got {
		t.Errorf("testBitwiseRsh failed, wanted %d got %d", want, got)
	}
	if want, got := int32(0), testBitwiseRsh_ssa(13, 25, 15); want != got {
		t.Errorf("testBitwiseRsh failed, wanted %d got %d", want, got)
	}
	if want, got := int32(-1), testBitwiseRsh_ssa(-13, 25, 15); want != got {
		t.Errorf("testBitwiseRsh failed, wanted %d got %d", want, got)
	}
	if want, got := uint32(0x3ffffff), testBitwiseRshU_ssa(0xffffffff, 4, 2); want != got {
		t.Errorf("testBitwiseRshU failed, wanted %d got %d", want, got)
	}
	if want, got := uint32(0), testBitwiseRshU_ssa(13, 25, 15); want != got {
		t.Errorf("testBitwiseRshU failed, wanted %d got %d", want, got)
	}
	if want, got := uint32(0), testBitwiseRshU_ssa(0x8aaaaaaa, 25, 15); want != got {
		t.Errorf("testBitwiseRshU failed, wanted %d got %d", want, got)
	}
}

//go:noinline
func testBitwiseAnd_ssa(a, b uint32) uint32 {
	return a & b
}

//go:noinline
func testBitwiseOr_ssa(a, b uint32) uint32 {
	return a | b
}

//go:noinline
func testBitwiseXor_ssa(a, b uint32) uint32 {
	return a ^ b
}

//go:noinline
func testBitwiseLsh_ssa(a int32, b, c uint32) int32 {
	return a << b << c
}

//go:noinline
func testBitwiseRsh_ssa(a int32, b, c uint32) int32 {
	return a >> b >> c
}

//go:noinline
func testBitwiseRshU_ssa(a uint32, b, c uint32) uint32 {
	return a >> b >> c
}

//go:noinline
func testShiftCX_ssa() int {
	v1 := uint8(3)
	v4 := (v1 * v1) ^ v1 | v1 - v1 - v1&v1 ^ uint8(3+2) + v1*1>>0 - v1 | 1 | v1<<(2*3|0-0*0^1)
	v5 := v4>>(3-0-uint(3)) | v1 | v1 + v1 ^ v4<<(0+1|3&1)<<(uint64(1)<<0*2*0<<0) ^ v1
	v6 := v5 ^ (v1+v1)*v1 | v1 | v1*v1>>(v1&v1)>>(uint(1)<<0*uint(3)>>1)*v1<<2*v1<<v1 - v1>>2 | (v4 - v1) ^ v1 + v1 ^ v1>>1 | v1 + v1 - v1 ^ v1
	v7 := v6 & v5 << 0
	v1++
	v11 := 2&1 ^ 0 + 3 | int(0^0)<<1>>(1*0*3) ^ 0*0 ^ 3&0*3&3 ^ 3*3 ^ 1 ^ int(2)<<(2*3) + 2 | 2 | 2 ^ 2 + 1 | 3 | 0 ^ int(1)>>1 ^ 2 // int
	v7--
	return int(uint64(2*1)<<(3-2)<<uint(3>>v7)-2)&v11 | v11 - int(2)<<0>>(2-1)*(v11*0&v11<<1<<(uint8(2)+v4))
}

func testShiftCX(t *testing.T) {
	want := 141
	if got := testShiftCX_ssa(); want != got {
		t.Errorf("testShiftCX failed, wanted %d got %d", want, got)
	}
}

// testSubqToNegq ensures that the SUBQ -> NEGQ translation works correctly.
func testSubqToNegq(t *testing.T) {
	want := int64(-318294940372190156)
	if got := testSubqToNegq_ssa(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2); want != got {
		t.Errorf("testSubqToNegq failed, wanted %d got %d", want, got)
	}
}

//go:noinline
func testSubqToNegq_ssa(a, b, c, d, e, f, g, h, i, j, k int64) int64 {
	return a + 8207351403619448057 - b - 1779494519303207690 + c*8810076340510052032*d - 4465874067674546219 - e*4361839741470334295 - f + 8688847565426072650*g*8065564729145417479
}

func testOcom(t *testing.T) {
	want1, want2 := int32(0x55555555), int32(-0x55555556)
	if got1, got2 := testOcom_ssa(0x55555555, 0x55555555); want1 != got1 || want2 != got2 {
		t.Errorf("testOcom failed, wanted %d and %d got %d and %d", want1, want2, got1, got2)
	}
}

//go:noinline
func testOcom_ssa(a, b int32) (int32, int32) {
	return ^^^^a, ^^^^^b
}

func lrot1_ssa(w uint8, x uint16, y uint32, z uint64) (a uint8, b uint16, c uint32, d uint64) {
	a = (w << 5) | (w >> 3)
	b = (x << 13) | (x >> 3)
	c = (y << 29) | (y >> 3)
	d = (z << 61) | (z >> 3)
	return
}

//go:noinline
func lrot2_ssa(w, n uint32) uint32 {
	// Want to be sure that a "rotate by 32" which
	// is really 0 | (w >> 0) == w
	// is correctly compiled.
	return (w << n) | (w >> (32 - n))
}

//go:noinline
func lrot3_ssa(w uint32) uint32 {
	// Want to be sure that a "rotate by 32" which
	// is really 0 | (w >> 0) == w
	// is correctly compiled.
	return (w << 32) | (w >> (32 - 32))
}

func testLrot(t *testing.T) {
	wantA, wantB, wantC, wantD := uint8(0xe1), uint16(0xe001),
		uint32(0xe0000001), uint64(0xe000000000000001)
	a, b, c, d := lrot1_ssa(0xf, 0xf, 0xf, 0xf)
	if a != wantA || b != wantB || c != wantC || d != wantD {
		t.Errorf("lrot1_ssa(0xf, 0xf, 0xf, 0xf)=%d %d %d %d, got %d %d %d %d", wantA, wantB, wantC, wantD, a, b, c, d)
	}
	x := lrot2_ssa(0xb0000001, 32)
	wantX := uint32(0xb0000001)
	if x != wantX {
		t.Errorf("lrot2_ssa(0xb0000001, 32)=%d, got %d", wantX, x)
	}
	x = lrot3_ssa(0xb0000001)
	if x != wantX {
		t.Errorf("lrot3_ssa(0xb0000001)=%d, got %d", wantX, x)
	}

}

//go:noinline
func sub1_ssa() uint64 {
	v1 := uint64(3) // uint64
	return v1*v1 - (v1&v1)&v1
}

//go:noinline
func sub2_ssa() uint8 {
	v1 := uint8(0)
	v3 := v1 + v1 + v1 ^ v1 | 3 + v1 ^ v1 | v1 ^ v1
	v1-- // dev.ssa doesn't see this one
	return v1 ^ v1*v1 - v3
}

func testSubConst(t *testing.T) {
	x1 := sub1_ssa()
	want1 := uint64(6)
	if x1 != want1 {
		t.Errorf("sub1_ssa()=%d, got %d", want1, x1)
	}
	x2 := sub2_ssa()
	want2 := uint8(251)
	if x2 != want2 {
		t.Errorf("sub2_ssa()=%d, got %d", want2, x2)
	}
}

//go:noinline
func orPhi_ssa(a bool, x int) int {
	v := 0
	if a {
		v = -1
	} else {
		v = -1
	}
	return x | v
}

func testOrPhi(t *testing.T) {
	if want, got := -1, orPhi_ssa(true, 4); got != want {
		t.Errorf("orPhi_ssa(true, 4)=%d, want %d", got, want)
	}
	if want, got := -1, orPhi_ssa(false, 0); got != want {
		t.Errorf("orPhi_ssa(false, 0)=%d, want %d", got, want)
	}
}

//go:noinline
func addshiftLL_ssa(a, b uint32) uint32 {
	return a + b<<3
}

//go:noinline
func subshiftLL_ssa(a, b uint32) uint32 {
	return a - b<<3
}

//go:noinline
func rsbshiftLL_ssa(a, b uint32) uint32 {
	return a<<3 - b
}

//go:noinline
func andshiftLL_ssa(a, b uint32) uint32 {
	return a & (b << 3)
}

//go:noinline
func orshiftLL_ssa(a, b uint32) uint32 {
	return a | b<<3
}

//go:noinline
func xorshiftLL_ssa(a, b uint32) uint32 {
	return a ^ b<<3
}

//go:noinline
func bicshiftLL_ssa(a, b uint32) uint32 {
	return a &^ (b << 3)
}

//go:noinline
func notshiftLL_ssa(a uint32) uint32 {
	return ^(a << 3)
}

//go:noinline
func addshiftRL_ssa(a, b uint32) uint32 {
	return a + b>>3
}

//go:noinline
func subshiftRL_ssa(a, b uint32) uint32 {
	return a - b>>3
}

//go:noinline
func rsbshiftRL_ssa(a, b uint32) uint32 {
	return a>>3 - b
}

//go:noinline
func andshiftRL_ssa(a, b uint32) uint32 {
	return a & (b >> 3)
}

//go:noinline
func orshiftRL_ssa(a, b uint32) uint32 {
	return a | b>>3
}

//go:noinline
func xorshiftRL_ssa(a, b uint32) uint32 {
	return a ^ b>>3
}

//go:noinline
func bicshiftRL_ssa(a, b uint32) uint32 {
	return a &^ (b >> 3)
}

//go:noinline
func notshiftRL_ssa(a uint32) uint32 {
	return ^(a >> 3)
}

//go:noinline
func addshiftRA_ssa(a, b int32) int32 {
	return a + b>>3
}

//go:noinline
func subshiftRA_ssa(a, b int32) int32 {
	return a - b>>3
}

//go:noinline
func rsbshiftRA_ssa(a, b int32) int32 {
	return a>>3 - b
}

//go:noinline
func andshiftRA_ssa(a, b int32) int32 {
	return a & (b >> 3)
}

//go:noinline
func orshiftRA_ssa(a, b int32) int32 {
	return a | b>>3
}

//go:noinline
func xorshiftRA_ssa(a, b int32) int32 {
	return a ^ b>>3
}

//go:noinline
func bicshiftRA_ssa(a, b int32) int32 {
	return a &^ (b >> 3)
}

//go:noinline
func notshiftRA_ssa(a int32) int32 {
	return ^(a >> 3)
}

//go:noinline
func addshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a + b<<s
}

//go:noinline
func subshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a - b<<s
}

//go:noinline
func rsbshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a<<s - b
}

//go:noinline
func andshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a & (b << s)
}

//go:noinline
func orshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a | b<<s
}

//go:noinline
func xorshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a ^ b<<s
}

//go:noinline
func bicshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a &^ (b << s)
}

//go:noinline
func notshiftLLreg_ssa(a uint32, s uint8) uint32 {
	return ^(a << s)
}

//go:noinline
func addshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a + b>>s
}

//go:noinline
func subshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a - b>>s
}

//go:noinline
func rsbshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a>>s - b
}

//go:noinline
func andshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a & (b >> s)
}

//go:noinline
func orshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a | b>>s
}

//go:noinline
func xorshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a ^ b>>s
}

//go:noinline
func bicshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a &^ (b >> s)
}

//go:noinline
func notshiftRLreg_ssa(a uint32, s uint8) uint32 {
	return ^(a >> s)
}

//go:noinline
func addshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a + b>>s
}

//go:noinline
func subshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a - b>>s
}

//go:noinline
func rsbshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a>>s - b
}

//go:noinline
func andshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a & (b >> s)
}

//go:noinline
func orshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a | b>>s
}

//go:noinline
func xorshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a ^ b>>s
}

//go:noinline
func bicshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a &^ (b >> s)
}

//go:noinline
func notshiftRAreg_ssa(a int32, s uint8) int32 {
	return ^(a >> s)
}

// test ARM shifted ops
func testShiftedOps(t *testing.T) {
	a, b := uint32(10), uint32(42)
	if want, got := a+b<<3, addshiftLL_ssa(a, b); got != want {
		t.Errorf("addshiftLL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a-b<<3, subshiftLL_ssa(a, b); got != want {
		t.Errorf("subshiftLL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a<<3-b, rsbshiftLL_ssa(a, b); got != want {
		t.Errorf("rsbshiftLL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a&(b<<3), andshiftLL_ssa(a, b); got != want {
		t.Errorf("andshiftLL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a|b<<3, orshiftLL_ssa(a, b); got != want {
		t.Errorf("orshiftLL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a^b<<3, xorshiftLL_ssa(a, b); got != want {
		t.Errorf("xorshiftLL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a&^(b<<3), bicshiftLL_ssa(a, b); got != want {
		t.Errorf("bicshiftLL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := ^(a << 3), notshiftLL_ssa(a); got != want {
		t.Errorf("notshiftLL_ssa(10) = %d want %d", got, want)
	}
	if want, got := a+b>>3, addshiftRL_ssa(a, b); got != want {
		t.Errorf("addshiftRL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a-b>>3, subshiftRL_ssa(a, b); got != want {
		t.Errorf("subshiftRL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a>>3-b, rsbshiftRL_ssa(a, b); got != want {
		t.Errorf("rsbshiftRL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a&(b>>3), andshiftRL_ssa(a, b); got != want {
		t.Errorf("andshiftRL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a|b>>3, orshiftRL_ssa(a, b); got != want {
		t.Errorf("orshiftRL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a^b>>3, xorshiftRL_ssa(a, b); got != want {
		t.Errorf("xorshiftRL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := a&^(b>>3), bicshiftRL_ssa(a, b); got != want {
		t.Errorf("bicshiftRL_ssa(10, 42) = %d want %d", got, want)
	}
	if want, got := ^(a >> 3), notshiftRL_ssa(a); got != want {
		t.Errorf("notshiftRL_ssa(10) = %d want %d", got, want)
	}
	c, d := int32(10), int32(-42)
	if want, got := c+d>>3, addshiftRA_ssa(c, d); got != want {
		t.Errorf("addshiftRA_ssa(10, -42) = %d want %d", got, want)
	}
	if want, got := c-d>>3, subshiftRA_ssa(c, d); got != want {
		t.Errorf("subshiftRA_ssa(10, -42) = %d want %d", got, want)
	}
	if want, got := c>>3-d, rsbshiftRA_ssa(c, d); got != want {
		t.Errorf("rsbshiftRA_ssa(10, -42) = %d want %d", got, want)
	}
	if want, got := c&(d>>3), andshiftRA_ssa(c, d); got != want {
		t.Errorf("andshiftRA_ssa(10, -42) = %d want %d", got, want)
	}
	if want, got := c|d>>3, orshiftRA_ssa(c, d); got != want {
		t.Errorf("orshiftRA_ssa(10, -42) = %d want %d", got, want)
	}
	if want, got := c^d>>3, xorshiftRA_ssa(c, d); got != want {
		t.Errorf("xorshiftRA_ssa(10, -42) = %d want %d", got, want)
	}
	if want, got := c&^(d>>3), bicshiftRA_ssa(c, d); got != want {
		t.Errorf("bicshiftRA_ssa(10, -42) = %d want %d", got, want)
	}
	if want, got := ^(d >> 3), notshiftRA_ssa(d); got != want {
		t.Errorf("notshiftRA_ssa(-42) = %d want %d", got, want)
	}
	s := uint8(3)
	if want, got := a+b<<s, addshiftLLreg_ssa(a, b, s); got != want {
		t.Errorf("addshiftLLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a-b<<s, subshiftLLreg_ssa(a, b, s); got != want {
		t.Errorf("subshiftLLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a<<s-b, rsbshiftLLreg_ssa(a, b, s); got != want {
		t.Errorf("rsbshiftLLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a&(b<<s), andshiftLLreg_ssa(a, b, s); got != want {
		t.Errorf("andshiftLLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a|b<<s, orshiftLLreg_ssa(a, b, s); got != want {
		t.Errorf("orshiftLLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a^b<<s, xorshiftLLreg_ssa(a, b, s); got != want {
		t.Errorf("xorshiftLLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a&^(b<<s), bicshiftLLreg_ssa(a, b, s); got != want {
		t.Errorf("bicshiftLLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := ^(a << s), notshiftLLreg_ssa(a, s); got != want {
		t.Errorf("notshiftLLreg_ssa(10) = %d want %d", got, want)
	}
	if want, got := a+b>>s, addshiftRLreg_ssa(a, b, s); got != want {
		t.Errorf("addshiftRLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a-b>>s, subshiftRLreg_ssa(a, b, s); got != want {
		t.Errorf("subshiftRLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a>>s-b, rsbshiftRLreg_ssa(a, b, s); got != want {
		t.Errorf("rsbshiftRLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a&(b>>s), andshiftRLreg_ssa(a, b, s); got != want {
		t.Errorf("andshiftRLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a|b>>s, orshiftRLreg_ssa(a, b, s); got != want {
		t.Errorf("orshiftRLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a^b>>s, xorshiftRLreg_ssa(a, b, s); got != want {
		t.Errorf("xorshiftRLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := a&^(b>>s), bicshiftRLreg_ssa(a, b, s); got != want {
		t.Errorf("bicshiftRLreg_ssa(10, 42, 3) = %d want %d", got, want)
	}
	if want, got := ^(a >> s), notshiftRLreg_ssa(a, s); got != want {
		t.Errorf("notshiftRLreg_ssa(10) = %d want %d", got, want)
	}
	if want, got := c+d>>s, addshiftRAreg_ssa(c, d, s); got != want {
		t.Errorf("addshiftRAreg_ssa(10, -42, 3) = %d want %d", got, want)
	}
	if want, got := c-d>>s, subshiftRAreg_ssa(c, d, s); got != want {
		t.Errorf("subshiftRAreg_ssa(10, -42, 3) = %d want %d", got, want)
	}
	if want, got := c>>s-d, rsbshiftRAreg_ssa(c, d, s); got != want {
		t.Errorf("rsbshiftRAreg_ssa(10, -42, 3) = %d want %d", got, want)
	}
	if want, got := c&(d>>s), andshiftRAreg_ssa(c, d, s); got != want {
		t.Errorf("andshiftRAreg_ssa(10, -42, 3) = %d want %d", got, want)
	}
	if want, got := c|d>>s, orshiftRAreg_ssa(c, d, s); got != want {
		t.Errorf("orshiftRAreg_ssa(10, -42, 3) = %d want %d", got, want)
	}
	if want, got := c^d>>s, xorshiftRAreg_ssa(c, d, s); got != want {
		t.Errorf("xorshiftRAreg_ssa(10, -42, 3) = %d want %d", got, want)
	}
	if want, got := c&^(d>>s), bicshiftRAreg_ssa(c, d, s); got != want {
		t.Errorf("bicshiftRAreg_ssa(10, -42, 3) = %d want %d", got, want)
	}
	if want, got := ^(d >> s), notshiftRAreg_ssa(d, s); got != want {
		t.Errorf("notshiftRAreg_ssa(-42, 3) = %d want %d", got, want)
	}
}

// TestArithmetic tests that both backends have the same result for arithmetic expressions.
func TestArithmetic(t *testing.T) {
	test64BitConstMult(t)
	test64BitConstAdd(t)
	testRegallocCVSpill(t)
	testSubqToNegq(t)
	testBitwiseLogic(t)
	testOcom(t)
	testLrot(t)
	testShiftCX(t)
	testSubConst(t)
	testOverflowConstShift(t)
	testArithRightShiftConstOverflow(t)
	testRightShiftConstOverflow(t)
	testArithConstShift(t)
	testArithRshConst(t)
	testLargeConst(t)
	testLoadCombine(t)
	testLoadSymCombine(t)
	testShiftRemoval(t)
	testShiftedOps(t)
	testDivFixUp(t)
	testDivisibleSignedPow2(t)
	testDivisibility(t)
}

// testDivFixUp ensures that signed division fix-ups are being generated.
func testDivFixUp(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Error("testDivFixUp failed")
			if e, ok := r.(runtime.Error); ok {
				t.Logf("%v\n", e.Error())
			}
		}
	}()
	var w int8 = -128
	var x int16 = -32768
	var y int32 = -2147483648
	var z int64 = -9223372036854775808

	for i := -5; i < 0; i++ {
		g8 = w / int8(i)
		g16 = x / int16(i)
		g32 = y / int32(i)
		g64 = z / int64(i)
		g8 = w % int8(i)
		g16 = x % int16(i)
		g32 = y % int32(i)
		g64 = z % int64(i)
	}
}

//go:noinline
func divisible_int8_2to1(x int8) bool {
	return x%(1<<1) == 0
}

//go:noinline
func divisible_int8_2to2(x int8) bool {
	return x%(1<<2) == 0
}

//go:noinline
func divisible_int8_2to3(x int8) bool {
	return x%(1<<3) == 0
}

//go:noinline
func divisible_int8_2to4(x int8) bool {
	return x%(1<<4) == 0
}

//go:noinline
func divisible_int8_2to5(x int8) bool {
	return x%(1<<5) == 0
}

//go:noinline
func divisible_int8_2to6(x int8) bool {
	return x%(1<<6) == 0
}

//go:noinline
func divisible_int16_2to1(x int16) bool {
	return x%(1<<1) == 0
}

//go:noinline
func divisible_int16_2to2(x int16) bool {
	return x%(1<<2) == 0
}

//go:noinline
func divisible_int16_2to3(x int16) bool {
	return x%(1<<3) == 0
}

//go:noinline
func divisible_int16_2to4(x int16) bool {
	return x%(1<<4) == 0
}

//go:noinline
func divisible_int16_2to5(x int16) bool {
	return x%(1<<5) == 0
}

//go:noinline
func divisible_int16_2to6(x int16) bool {
	return x%(1<<6) == 0
}

//go:noinline
func divisible_int16_2to7(x int16) bool {
	return x%(1<<7) == 0
}

//go:noinline
func divisible_int16_2to8(x int16) bool {
	return x%(1<<8) == 0
}

//go:noinline
func divisible_int16_2to9(x int16) bool {
	return x%(1<<9) == 0
}

//go:noinline
func divisible_int16_2to10(x int16) bool {
	return x%(1<<10) == 0
}

//go:noinline
func divisible_int16_2to11(x int16) bool {
	return x%(1<<11) == 0
}

//go:noinline
func divisible_int16_2to12(x int16) bool {
	return x%(1<<12) == 0
}

//go:noinline
func divisible_int16_2to13(x int16) bool {
	return x%(1<<13) == 0
}

//go:noinline
func divisible_int16_2to14(x int16) bool {
	return x%(1<<14) == 0
}

//go:noinline
func divisible_int32_2to4(x int32) bool {
	return x%(1<<4) == 0
}

//go:noinline
func divisible_int32_2to15(x int32) bool {
	return x%(1<<15) == 0
}

//go:noinline
func divisible_int32_2to26(x int32) bool {
	return x%(1<<26) == 0
}

//go:noinline
func divisible_int64_2to4(x int64) bool {
	return x%(1<<4) == 0
}

//go:noinline
func divisible_int64_2to15(x int64) bool {
	return x%(1<<15) == 0
}

//go:noinline
func divisible_int64_2to26(x int64) bool {
	return x%(1<<26) == 0
}

//go:noinline
func divisible_int64_2to34(x int64) bool {
	return x%(1<<34) == 0
}

//go:noinline
func divisible_int64_2to48(x int64) bool {
	return x%(1<<48) == 0
}

//go:noinline
func divisible_int64_2to57(x int64) bool {
	return x%(1<<57) == 0
}

// testDivisibleSignedPow2 confirms that x%(1<<k)==0 is rewritten correctly
func testDivisibleSignedPow2(t *testing.T) {
	var i int64
	var pow2 = []int64{
		1,
		1 << 1,
		1 << 2,
		1 << 3,
		1 << 4,
		1 << 5,
		1 << 6,
		1 << 7,
		1 << 8,
		1 << 9,
		1 << 10,
		1 << 11,
		1 << 12,
		1 << 13,
		1 << 14,
	}
	// exhaustive test for int8
	for i = math.MinInt8; i <= math.MaxInt8; i++ {
		if want, got := int8(i)%int8(pow2[1]) == 0, divisible_int8_2to1(int8(i)); got != want {
			t.Errorf("divisible_int8_2to1(%d) = %v want %v", i, got, want)
		}
		if want, got := int8(i)%int8(pow2[2]) == 0, divisible_int8_2to2(int8(i)); got != want {
			t.Errorf("divisible_int8_2to2(%d) = %v want %v", i, got, want)
		}
		if want, got := int8(i)%int8(pow2[3]) == 0, divisible_int8_2to3(int8(i)); got != want {
			t.Errorf("divisible_int8_2to3(%d) = %v want %v", i, got, want)
		}
		if want, got := int8(i)%int8(pow2[4]) == 0, divisible_int8_2to4(int8(i)); got != want {
			t.Errorf("divisible_int8_2to4(%d) = %v want %v", i, got, want)
		}
		if want, got := int8(i)%int8(pow2[5]) == 0, divisible_int8_2to5(int8(i)); got != want {
			t.Errorf("divisible_int8_2to5(%d) = %v want %v", i, got, want)
		}
		if want, got := int8(i)%int8(pow2[6]) == 0, divisible_int8_2to6(int8(i)); got != want {
			t.Errorf("divisible_int8_2to6(%d) = %v want %v", i, got, want)
		}
	}
	// exhaustive test for int16
	for i = math.MinInt16; i <= math.MaxInt16; i++ {
		if want, got := int16(i)%int16(pow2[1]) == 0, divisible_int16_2to1(int16(i)); got != want {
			t.Errorf("divisible_int16_2to1(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[2]) == 0, divisible_int16_2to2(int16(i)); got != want {
			t.Errorf("divisible_int16_2to2(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[3]) == 0, divisible_int16_2to3(int16(i)); got != want {
			t.Errorf("divisible_int16_2to3(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[4]) == 0, divisible_int16_2to4(int16(i)); got != want {
			t.Errorf("divisible_int16_2to4(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[5]) == 0, divisible_int16_2to5(int16(i)); got != want {
			t.Errorf("divisible_int16_2to5(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[6]) == 0, divisible_int16_2to6(int16(i)); got != want {
			t.Errorf("divisible_int16_2to6(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[7]) == 0, divisible_int16_2to7(int16(i)); got != want {
			t.Errorf("divisible_int16_2to7(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[8]) == 0, divisible_int16_2to8(int16(i)); got != want {
			t.Errorf("divisible_int16_2to8(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[9]) == 0, divisible_int16_2to9(int16(i)); got != want {
			t.Errorf("divisible_int16_2to9(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[10]) == 0, divisible_int16_2to10(int16(i)); got != want {
			t.Errorf("divisible_int16_2to10(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[11]) == 0, divisible_int16_2to11(int16(i)); got != want {
			t.Errorf("divisible_int16_2to11(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[12]) == 0, divisible_int16_2to12(int16(i)); got != want {
			t.Errorf("divisible_int16_2to12(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[13]) == 0, divisible_int16_2to13(int16(i)); got != want {
			t.Errorf("divisible_int16_2to13(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(pow2[14]) == 0, divisible_int16_2to14(int16(i)); got != want {
			t.Errorf("divisible_int16_2to14(%d) = %v want %v", i, got, want)
		}
	}
	// spot check for int32 and int64
	var (
		two4  int64 = 1 << 4
		two15 int64 = 1 << 15
		two26 int64 = 1 << 26
		two34 int64 = 1 << 34
		two48 int64 = 1 << 48
		two57 int64 = 1 << 57
	)
	var xs = []int64{two4, two4 + 3, -3 * two4, -3*two4 + 1,
		two15, two15 + 3, -3 * two15, -3*two15 + 1,
		two26, two26 + 37, -5 * two26, -5*two26 + 2,
		two34, two34 + 356, -7 * two34, -7*two34 + 13,
		two48, two48 + 3000, -12 * two48, -12*two48 + 1111,
		two57, two57 + 397654, -15 * two57, -15*two57 + 11234,
	}
	for _, x := range xs {
		if int64(int32(x)) == x {
			if want, got := int32(x)%int32(two4) == 0, divisible_int32_2to4(int32(x)); got != want {
				t.Errorf("divisible_int32_2to4(%d) = %v want %v", x, got, want)
			}

			if want, got := int32(x)%int32(two15) == 0, divisible_int32_2to15(int32(x)); got != want {
				t.Errorf("divisible_int32_2to15(%d) = %v want %v", x, got, want)
			}

			if want, got := int32(x)%int32(two26) == 0, divisible_int32_2to26(int32(x)); got != want {
				t.Errorf("divisible_int32_2to26(%d) = %v want %v", x, got, want)
			}
		}
		// spot check for int64
		if want, got := x%two4 == 0, divisible_int64_2to4(x); got != want {
			t.Errorf("divisible_int64_2to4(%d) = %v want %v", x, got, want)
		}

		if want, got := x%two15 == 0, divisible_int64_2to15(x); got != want {
			t.Errorf("divisible_int64_2to15(%d) = %v want %v", x, got, want)
		}

		if want, got := x%two26 == 0, divisible_int64_2to26(x); got != want {
			t.Errorf("divisible_int64_2to26(%d) = %v want %v", x, got, want)
		}

		if want, got := x%two34 == 0, divisible_int64_2to34(x); got != want {
			t.Errorf("divisible_int64_2to34(%d) = %v want %v", x, got, want)
		}

		if want, got := x%two48 == 0, divisible_int64_2to48(x); got != want {
			t.Errorf("divisible_int64_2to48(%d) = %v want %v", x, got, want)
		}

		if want, got := x%two57 == 0, divisible_int64_2to57(x); got != want {
			t.Errorf("divisible_int64_2to57(%d) = %v want %v", x, got, want)
		}
	}
}

func div6_uint8(n uint8) bool {
	return n%6 == 0
}

//go:noinline
func div6_uint16(n uint16) bool {
	return n%6 == 0
}

//go:noinline
func div6_uint32(n uint32) bool {
	return n%6 == 0
}

//go:noinline
func div6_uint64(n uint64) bool {
	return n%6 == 0
}

//go:noinline
func div19_uint8(n uint8) bool {
	return n%19 == 0
}

//go:noinline
func div19_uint16(n uint16) bool {
	return n%19 == 0
}

//go:noinline
func div19_uint32(n uint32) bool {
	return n%19 == 0
}

//go:noinline
func div19_uint64(n uint64) bool {
	return n%19 == 0
}

//go:noinline
func div6_int8(n int8) bool {
	return n%6 == 0
}

//go:noinline
func div6_int16(n int16) bool {
	return n%6 == 0
}

//go:noinline
func div6_int32(n int32) bool {
	return n%6 == 0
}

//go:noinline
func div6_int64(n int64) bool {
	return n%6 == 0
}

//go:noinline
func div19_int8(n int8) bool {
	return n%19 == 0
}

//go:noinline
func div19_int16(n int16) bool {
	return n%19 == 0
}

//go:noinline
func div19_int32(n int32) bool {
	return n%19 == 0
}

//go:noinline
func div19_int64(n int64) bool {
	return n%19 == 0
}

// testDivisibility confirms that rewrite rules x%c ==0 for c constant are correct.
func testDivisibility(t *testing.T) {
	// unsigned tests
	// test an even and an odd divisor
	var sixU, nineteenU uint64 = 6, 19
	// test all inputs for uint8, uint16
	for i := uint64(0); i <= math.MaxUint16; i++ {
		if i <= math.MaxUint8 {
			if want, got := uint8(i)%uint8(sixU) == 0, div6_uint8(uint8(i)); got != want {
				t.Errorf("div6_uint8(%d) = %v want %v", i, got, want)
			}
			if want, got := uint8(i)%uint8(nineteenU) == 0, div19_uint8(uint8(i)); got != want {
				t.Errorf("div6_uint19(%d) = %v want %v", i, got, want)
			}
		}
		if want, got := uint16(i)%uint16(sixU) == 0, div6_uint16(uint16(i)); got != want {
			t.Errorf("div6_uint16(%d) = %v want %v", i, got, want)
		}
		if want, got := uint16(i)%uint16(nineteenU) == 0, div19_uint16(uint16(i)); got != want {
			t.Errorf("div19_uint16(%d) = %v want %v", i, got, want)
		}
	}
	var maxU32, maxU64 uint64 = math.MaxUint32, math.MaxUint64
	// spot check inputs for uint32 and uint64
	xu := []uint64{
		0, 1, 2, 3, 4, 5,
		sixU, 2 * sixU, 3 * sixU, 5 * sixU, 12345 * sixU,
		sixU + 1, 2*sixU - 5, 3*sixU + 3, 5*sixU + 4, 12345*sixU - 2,
		nineteenU, 2 * nineteenU, 3 * nineteenU, 5 * nineteenU, 12345 * nineteenU,
		nineteenU + 1, 2*nineteenU - 5, 3*nineteenU + 3, 5*nineteenU + 4, 12345*nineteenU - 2,
		maxU32, maxU32 - 1, maxU32 - 2, maxU32 - 3, maxU32 - 4,
		maxU32 - 5, maxU32 - 6, maxU32 - 7, maxU32 - 8,
		maxU32 - 9, maxU32 - 10, maxU32 - 11, maxU32 - 12,
		maxU32 - 13, maxU32 - 14, maxU32 - 15, maxU32 - 16,
		maxU32 - 17, maxU32 - 18, maxU32 - 19, maxU32 - 20,
		maxU64, maxU64 - 1, maxU64 - 2, maxU64 - 3, maxU64 - 4,
		maxU64 - 5, maxU64 - 6, maxU64 - 7, maxU64 - 8,
		maxU64 - 9, maxU64 - 10, maxU64 - 11, maxU64 - 12,
		maxU64 - 13, maxU64 - 14, maxU64 - 15, maxU64 - 16,
		maxU64 - 17, maxU64 - 18, maxU64 - 19, maxU64 - 20,
	}
	for _, x := range xu {
		if x <= maxU32 {
			if want, got := uint32(x)%uint32(sixU) == 0, div6_uint32(uint32(x)); got != want {
				t.Errorf("div6_uint32(%d) = %v want %v", x, got, want)
			}
			if want, got := uint32(x)%uint32(nineteenU) == 0, div19_uint32(uint32(x)); got != want {
				t.Errorf("div19_uint32(%d) = %v want %v", x, got, want)
			}
		}
		if want, got := x%sixU == 0, div6_uint64(x); got != want {
			t.Errorf("div6_uint64(%d) = %v want %v", x, got, want)
		}
		if want, got := x%nineteenU == 0, div19_uint64(x); got != want {
			t.Errorf("div19_uint64(%d) = %v want %v", x, got, want)
		}
	}

	// signed tests
	// test an even and an odd divisor
	var sixS, nineteenS int64 = 6, 19
	// test all inputs for int8, int16
	for i := int64(math.MinInt16); i <= math.MaxInt16; i++ {
		if math.MinInt8 <= i && i <= math.MaxInt8 {
			if want, got := int8(i)%int8(sixS) == 0, div6_int8(int8(i)); got != want {
				t.Errorf("div6_int8(%d) = %v want %v", i, got, want)
			}
			if want, got := int8(i)%int8(nineteenS) == 0, div19_int8(int8(i)); got != want {
				t.Errorf("div6_int19(%d) = %v want %v", i, got, want)
			}
		}
		if want, got := int16(i)%int16(sixS) == 0, div6_int16(int16(i)); got != want {
			t.Errorf("div6_int16(%d) = %v want %v", i, got, want)
		}
		if want, got := int16(i)%int16(nineteenS) == 0, div19_int16(int16(i)); got != want {
			t.Errorf("div19_int16(%d) = %v want %v", i, got, want)
		}
	}
	var minI32, maxI32, minI64, maxI64 int64 = math.MinInt32, math.MaxInt32, math.MinInt64, math.MaxInt64
	// spot check inputs for int32 and int64
	xs := []int64{
		0, 1, 2, 3, 4, 5,
		-1, -2, -3, -4, -5,
		sixS, 2 * sixS, 3 * sixS, 5 * sixS, 12345 * sixS,
		sixS + 1, 2*sixS - 5, 3*sixS + 3, 5*sixS + 4, 12345*sixS - 2,
		-sixS, -2 * sixS, -3 * sixS, -5 * sixS, -12345 * sixS,
		-sixS + 1, -2*sixS - 5, -3*sixS + 3, -5*sixS + 4, -12345*sixS - 2,
		nineteenS, 2 * nineteenS, 3 * nineteenS, 5 * nineteenS, 12345 * nineteenS,
		nineteenS + 1, 2*nineteenS - 5, 3*nineteenS + 3, 5*nineteenS + 4, 12345*nineteenS - 2,
		-nineteenS, -2 * nineteenS, -3 * nineteenS, -5 * nineteenS, -12345 * nineteenS,
		-nineteenS + 1, -2*nineteenS - 5, -3*nineteenS + 3, -5*nineteenS + 4, -12345*nineteenS - 2,
		minI32, minI32 + 1, minI32 + 2, minI32 + 3, minI32 + 4,
		minI32 + 5, minI32 + 6, minI32 + 7, minI32 + 8,
		minI32 + 9, minI32 + 10, minI32 + 11, minI32 + 12,
		minI32 + 13, minI32 + 14, minI32 + 15, minI32 + 16,
		minI32 + 17, minI32 + 18, minI32 + 19, minI32 + 20,
		maxI32, maxI32 - 1, maxI32 - 2, maxI32 - 3, maxI32 - 4,
		maxI32 - 5, maxI32 - 6, maxI32 - 7, maxI32 - 8,
		maxI32 - 9, maxI32 - 10, maxI32 - 11, maxI32 - 12,
		maxI32 - 13, maxI32 - 14, maxI32 - 15, maxI32 - 16,
		maxI32 - 17, maxI32 - 18, maxI32 - 19, maxI32 - 20,
		minI64, minI64 + 1, minI64 + 2, minI64 + 3, minI64 + 4,
		minI64 + 5, minI64 + 6, minI64 + 7, minI64 + 8,
		minI64 + 9, minI64 + 10, minI64 + 11, minI64 + 12,
		minI64 + 13, minI64 + 14, minI64 + 15, minI64 + 16,
		minI64 + 17, minI64 + 18, minI64 + 19, minI64 + 20,
		maxI64, maxI64 - 1, maxI64 - 2, maxI64 - 3, maxI64 - 4,
		maxI64 - 5, maxI64 - 6, maxI64 - 7, maxI64 - 8,
		maxI64 - 9, maxI64 - 10, maxI64 - 11, maxI64 - 12,
		maxI64 - 13, maxI64 - 14, maxI64 - 15, maxI64 - 16,
		maxI64 - 17, maxI64 - 18, maxI64 - 19, maxI64 - 20,
	}
	for _, x := range xs {
		if minI32 <= x && x <= maxI32 {
			if want, got := int32(x)%int32(sixS) == 0, div6_int32(int32(x)); got != want {
				t.Errorf("div6_int32(%d) = %v want %v", x, got, want)
			}
			if want, got := int32(x)%int32(nineteenS) == 0, div19_int32(int32(x)); got != want {
				t.Errorf("div19_int32(%d) = %v want %v", x, got, want)
			}
		}
		if want, got := x%sixS == 0, div6_int64(x); got != want {
			t.Errorf("div6_int64(%d) = %v want %v", x, got, want)
		}
		if want, got := x%nineteenS == 0, div19_int64(x); got != want {
			t.Errorf("div19_int64(%d) = %v want %v", x, got, want)
		}
	}
}

//go:noinline
func genREV16_1(c uint64) uint64 {
	b := ((c & 0xff00ff00ff00ff00) >> 8) | ((c & 0x00ff00ff00ff00ff) << 8)
	return b
}

//go:noinline
func genREV16_2(c uint64) uint64 {
	b := ((c & 0xff00ff00) >> 8) | ((c & 0x00ff00ff) << 8)
	return b
}

//go:noinline
func genREV16W(c uint32) uint32 {
	b := ((c & 0xff00ff00) >> 8) | ((c & 0x00ff00ff) << 8)
	return b
}

func TestREV16(t *testing.T) {
	x := uint64(0x8f7f6f5f4f3f2f1f)
	want1 := uint64(0x7f8f5f6f3f4f1f2f)
	want2 := uint64(0x3f4f1f2f)

	got1 := genREV16_1(x)
	if got1 != want1 {
		t.Errorf("genREV16_1(%#x) = %#x want %#x", x, got1, want1)
	}
	got2 := genREV16_2(x)
	if got2 != want2 {
		t.Errorf("genREV16_2(%#x) = %#x want %#x", x, got2, want2)
	}
}

func TestREV16W(t *testing.T) {
	x := uint32(0x4f3f2f1f)
	want := uint32(0x3f4f1f2f)

	got := genREV16W(x)
	if got != want {
		t.Errorf("genREV16W(%#x) = %#x want %#x", x, got, want)
	}
}
